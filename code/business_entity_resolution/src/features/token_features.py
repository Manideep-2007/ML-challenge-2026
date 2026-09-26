"""
Token-set evidence via sparse matrices (no per-pair Python loop).

Every record is tokenized ONCE per view into a binary sparse row over a
vocabulary shared by S1 and the reference universe (TokenIndex). A batch of
pairs just selects rows (C-level indexing); the row-wise product gives the
shared tokens, from which Jaccard, containment, overlap counts and
IDF-weighted evidence follow. IDF comes from document frequencies over the
whole reference universe (unsupervised, no labels).
"""

from array import array

import numpy as np
import scipy.sparse as sp

RARE_DF = 100


def build_matrices(texts_by_side: list, vocabulary: dict) -> list[sp.csr_matrix]:
    """
    Binary CSR per side over a shared, growing vocabulary. Deterministic:
    tokens are visited in sorted order (set order depends on per-process hash
    randomization) and row indices are sorted, so sums never change order.
    """
    raw = []
    for texts in texts_by_side:
        indices, indptr = array("i"), array("q", [0])
        for text in texts:
            if text:
                indices.extend(sorted(vocabulary.setdefault(t, len(vocabulary)) for t in set(text.split())))
            indptr.append(len(indices))
        raw.append((np.frombuffer(indices, dtype=np.int32), np.frombuffer(indptr, dtype=np.int64)))
    n_vocab = len(vocabulary)
    return [
        sp.csr_matrix((np.ones(len(idx), dtype=np.float32), idx, ptr), shape=(len(ptr) - 1, n_vocab))
        for idx, ptr in raw
    ]


def document_frequency_counts(texts, counts: dict) -> dict:
    """Add each text's distinct tokens to counts (token -> number of documents)."""
    for text in texts:
        if text:
            for token in set(text.split()):
                counts[token] = counts.get(token, 0) + 1
    return counts


class TokenIndex:
    """
    One view: S1 matrix, reference matrix and (optionally) IDF over the reference universe.
    The vocabulary is built from the reference side first, so token ids (and therefore
    summation order) do not depend on which S1 records are being processed.
    """

    def __init__(self, s1_texts, ref_texts, with_idf: bool, global_df: tuple[dict, int] | None = None):
        """
        global_df: (token -> document frequency, number of documents) over the whole
        reference universe, when ref_texts is only part of it (per-country inference).
        Without it, document frequencies come from ref_texts.
        """
        vocabulary: dict = {}
        self.ref = build_matrices([ref_texts], vocabulary)[0]
        self.s1 = build_matrices([s1_texts], vocabulary)[0]
        n_vocab = len(vocabulary)
        self.ref.resize((self.ref.shape[0], n_vocab))
        self.idf = None
        self.rare = None
        if with_idf:
            if global_df is None:
                df = np.bincount(self.ref.indices, minlength=n_vocab).astype(np.float64)
                n_docs = self.ref.shape[0]
            else:
                counts, n_docs = global_df
                tokens = np.empty(n_vocab, dtype=object)
                for token, i in vocabulary.items():
                    tokens[i] = token
                df = np.array([counts.get(t, 0) for t in tokens], dtype=np.float64)
            self.idf = np.log((n_docs + 1) / (df + 1)) + 1
            self.rare = (df <= RARE_DF).astype(np.float64)

    def pairs(self, q: np.ndarray, r: np.ndarray) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        return self.s1[q], self.ref[r]


def overlap(a: sp.csr_matrix, b: sp.csr_matrix, prefix: str, index: TokenIndex | None = None) -> dict:
    """
    Token-set features between aligned rows of a (S1) and b (candidate).
    Empty side -> NaN for ratios (no evidence), counts stay 0.
    """
    shared = a.multiply(b).tocsr()
    n_a = np.asarray(a.sum(axis=1)).ravel()
    n_b = np.asarray(b.sum(axis=1)).ravel()
    n_shared = np.asarray(shared.sum(axis=1)).ravel()
    union = n_a + n_b - n_shared
    missing = (n_a == 0) | (n_b == 0)

    def ratio(num, den):
        out = np.divide(num, den, out=np.zeros(len(num), dtype=np.float32), where=den > 0).astype(np.float32)
        out[missing] = np.nan
        return out

    features = {
        f"{prefix}_jaccard": ratio(n_shared, union),
        f"{prefix}_containment_s1": ratio(n_shared, n_a),
        f"{prefix}_containment_cand": ratio(n_shared, n_b),
        f"{prefix}_overlap_count": n_shared.astype(np.float32),
        f"{prefix}_count_s1": n_a.astype(np.float32),
        f"{prefix}_count_cand": n_b.astype(np.float32),
    }

    if index is not None and index.idf is not None:
        idf = index.idf   # float64 accumulation, stored as float32
        shared_idf = shared.astype(np.float64) @ idf
        a_idf, b_idf = a.astype(np.float64) @ idf, b.astype(np.float64) @ idf
        max_shared = shared.multiply(idf).tocsr().max(axis=1).toarray().ravel().astype(np.float32)
        features.update({
            f"{prefix}_shared_idf_sum": shared_idf.astype(np.float32),
            f"{prefix}_shared_idf_max": max_shared,
            f"{prefix}_idf_jaccard": ratio(shared_idf, a_idf + b_idf - shared_idf),
            f"{prefix}_s1_only_idf": (a_idf - shared_idf).astype(np.float32),
            f"{prefix}_cand_only_idf": (b_idf - shared_idf).astype(np.float32),
            f"{prefix}_shared_rare_count": (shared.astype(np.float64) @ index.rare).astype(np.float32),
        })
    return features
