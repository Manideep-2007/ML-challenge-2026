"""
Character n-gram TF-IDF retrieval (typos, spacing, partial spellings).
Cosine similarity via sparse products, top_k per query, per country.
N-grams with document frequency > max_df are dropped to keep products sparse.

only_empty_address=True (Stage 8 / E003): search only references whose
address is empty. Those records carry nothing but a (noisy) name, so they
rarely rank in the name+address hybrid channel; 42% of missed true pairs are
retrieval misses, concentrated on exactly these records. The pool is ~3% of
references, so character n-grams are affordable here.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .token_blocks import deleet, top_k_products


def char_channel(view: str, ngram_range: tuple[int, int], max_df: int, top_k: int, min_df: int = 2,
                 only_empty_address: bool = False, deleet_reference: bool = False):
    def channel(queries: pd.DataFrame, reference: pd.DataFrame):
        positions = np.arange(len(reference))
        if only_empty_address:
            positions = np.flatnonzero(reference["address_basic"].eq("").to_numpy(dtype=bool))
        if len(positions) == 0:
            empty = np.empty(0, np.int64)
            return empty, empty, {"reference_pool": 0, "rank": np.empty(0, np.int32)}
        ref_text = reference[view].iloc[positions]
        if deleet_reference:
            ref_text = ref_text.map(deleet)
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=min(min_df, len(positions)),
            max_df=max_df,
            lowercase=False,
            sublinear_tf=True,
            dtype=np.float32,
        )
        ref_m = vectorizer.fit_transform(ref_text.tolist())
        query_m = vectorizer.transform(queries[view].tolist())
        q, r, rank = top_k_products(query_m, ref_m, top_k)
        return q, positions[r], {"kept_ngrams": int(len(vectorizer.vocabulary_)),
                                 "reference_pool": int(len(positions)), "rank": rank}

    channel.columns = [view, "address_basic"] if only_empty_address else [view]
    return channel
