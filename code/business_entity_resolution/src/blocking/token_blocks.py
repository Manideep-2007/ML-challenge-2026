"""
Rare-token retrieval: records sharing informative tokens.

Each record is a binary bag of tokens; reference tokens are weighted by IDF
and tokens with document frequency > max_df are dropped (they would only
create huge blocks). A query's score against a reference record is the sum
of IDF over shared tokens; the top_k references per query are kept.
Computed per country, as sparse matrix products in query chunks.
"""

from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


def split_tokens(text: str) -> list[str]:
    return text.split()


LEET = str.maketrans({"0": "o", "1": "l", "5": "s", "6": "g", "8": "b"})


def deleet(text: str) -> str:
    """
    Undo leetspeak inside letter+digit tokens of length >= 4 ("6roup" -> "group",
    "hea1th" -> "health"). Pure numbers and short tokens ("5th", "b2b") are kept.
    Stage 3 rule R1: these substitutions occur only in S2/S3.
    """
    out = []
    for token in text.split():
        if len(token) >= 4 and not token.isdigit() and any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
            token = token.translate(LEET)
        out.append(token)
    return " ".join(out)


def _chunk_top_k(query_chunk: sp.csr_matrix, ref_t: sp.csr_matrix, top_k: int, offset: int):
    """Vectorized per-row top-k of one chunk's score matrix (ties broken by column index)."""
    scores = (query_chunk @ ref_t).tocsr()
    counts = np.diff(scores.indptr)
    rows = np.repeat(np.arange(scores.shape[0]), counts)
    order = np.lexsort((scores.indices, -scores.data, rows))
    starts = np.repeat(scores.indptr[:-1], counts)
    rank = np.arange(len(order)) - starts
    keep = rank < top_k
    kept = order[keep]
    return (rows[kept] + offset).astype(np.int64), scores.indices[kept].astype(np.int64), rank[keep].astype(np.int32)


# Measured on the real reference matrices: 8 threads beat 24 (95s vs 198s per
# 20k-query chunk); the products are memory-bandwidth bound.
DEFAULT_WORKERS = min(8, os.cpu_count() or 8)


def top_k_products(query: sp.csr_matrix, ref: sp.csr_matrix, top_k: int, chunk: int = 500,
                   workers: int = DEFAULT_WORKERS):
    """
    For each query row, the top_k reference rows by dot product (> 0).
    Returns (query_pos, ref_pos, rank) with rank 0 = best score. Chunks run
    on a thread pool (scipy's sparse product releases the GIL).
    """
    ref_t = ref.T.tocsr()
    starts = range(0, query.shape[0], chunk)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(lambda s: _chunk_top_k(query[s:s + chunk], ref_t, top_k, s), starts))
    if not parts:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int32)
    return tuple(np.concatenate(p) for p in zip(*parts))


class RareTokenRetriever:
    """
    Fit once on a reference set, query many times. Several views are combined
    into one bag with a view prefix, so name and address tokens never match
    each other (hybrid retrieval when >1 view).

    transforms: column -> str function applied to queries and references.
    ref_transforms: column -> str function applied to references only
    (e.g. the learned S2/S3 -> S1 token translation).
    """

    def __init__(self, text_columns: list[str], max_df: int, top_k: int,
                 transforms: dict | None = None, ref_transforms: dict | None = None):
        self.text_columns = text_columns
        self.max_df = max_df
        self.top_k = top_k
        self.transforms = transforms or {}
        self.ref_transforms = ref_transforms or {}

    def bag(self, frame: pd.DataFrame, is_reference: bool) -> pd.Series:
        parts = []
        for i, c in enumerate(self.text_columns):
            text = frame[c]
            if c in self.transforms:
                text = text.map(self.transforms[c])
            if is_reference and c in self.ref_transforms:
                text = text.map(self.ref_transforms[c])
            if len(self.text_columns) > 1:
                text = text.map(lambda s, p=i: " ".join(f"{p}:{t}" for t in s.split()))
            parts.append(text)
        out = parts[0]
        for part in parts[1:]:
            out = out + " " + part
        return out

    def fit(self, reference: pd.DataFrame):
        self.vectorizer = CountVectorizer(analyzer=split_tokens, binary=True, dtype=np.float32)
        ref = self.vectorizer.fit_transform(self.bag(reference, True).tolist()).tocsc()
        df = np.diff(ref.indptr)
        self.keep = np.flatnonzero((df > 0) & (df <= self.max_df))
        idf = np.log(ref.shape[0] / df[self.keep]).astype(np.float32)
        self.ref = (ref[:, self.keep] @ sp.diags(idf)).tocsr()
        dropped = np.flatnonzero(df > self.max_df)
        top_dropped = dropped[np.argsort(df[dropped])[::-1][:15]]
        names = self.vectorizer.get_feature_names_out()
        self.stats = {
            "vocabulary": int(len(df)),
            "kept_tokens": int(len(self.keep)),
            "dropped_common_tokens": int(len(dropped)),
            "most_common_dropped": {str(names[i]): int(df[i]) for i in top_dropped},
        }
        return self

    def query(self, queries: pd.DataFrame):
        query_m = self.vectorizer.transform(self.bag(queries, False).tolist()).tocsc()[:, self.keep].tocsr()
        return top_k_products(query_m, self.ref, self.top_k)


def token_channel(text_columns: list[str], max_df: int, top_k: int,
                  transforms: dict | None = None, ref_transforms: dict | None = None):
    """Channel wrapper for generate_candidates: fit on the partition's references, then query."""
    def channel(queries: pd.DataFrame, reference: pd.DataFrame):
        retriever = RareTokenRetriever(text_columns, max_df, top_k, transforms, ref_transforms).fit(reference)
        q, r, rank = retriever.query(queries)
        return q, r, {**retriever.stats, "rank": rank}

    channel.columns = text_columns
    return channel
