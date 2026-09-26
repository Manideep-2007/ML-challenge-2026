"""
Exact-key blocking: a query and a reference record are candidates when a
normalized view is identical. Implemented as an integer block join; blocks
larger than max_block are skipped (a key shared by thousands of records
carries no identity information on its own).
"""

import numpy as np
import pandas as pd

from .indexes import expand_ranges, sorted_block_index


def exact_join(query_keys: pd.Series, ref_keys: pd.Series, max_block: int):
    """
    Returns (query_pos, ref_pos, stats) for equal non-empty keys.
    Positions are local to the given series.
    """
    n_ref = len(ref_keys)
    keys = pd.concat([ref_keys, query_keys], ignore_index=True)
    codes, _ = pd.factorize(keys)
    codes[keys.eq("").to_numpy(dtype=bool)] = -1
    ref_codes, query_codes = codes[:n_ref], codes[n_ref:]

    order, sorted_codes = sorted_block_index(ref_codes)
    left = np.searchsorted(sorted_codes, query_codes, "left")
    right = np.searchsorted(sorted_codes, query_codes, "right")
    sizes = right - left
    sizes[query_codes < 0] = 0

    oversized = sizes > max_block
    sizes[oversized] = 0

    query_pos = np.repeat(np.arange(len(query_keys)), sizes)
    ref_pos = order[expand_ranges(left, sizes)]
    return query_pos, ref_pos, {"queries_with_oversized_block": int(oversized.sum())}


class ExactIndex:
    """Reference-side block structure built once; queried per chunk (same result as exact_join)."""

    def __init__(self, ref_keys: pd.Series, max_block: int):
        codes, uniques = pd.factorize(ref_keys)
        self.uniques = pd.Index(uniques)
        codes = np.where(ref_keys.eq("").to_numpy(dtype=bool), -1, codes)
        self.order, self.sorted_codes = sorted_block_index(codes)
        self.max_block = max_block

    def query(self, query_keys: pd.Series):
        codes = self.uniques.get_indexer(query_keys)
        codes[query_keys.eq("").to_numpy(dtype=bool)] = -1
        left = np.searchsorted(self.sorted_codes, codes, "left")
        right = np.searchsorted(self.sorted_codes, codes, "right")
        sizes = right - left
        sizes[codes < 0] = 0
        sizes[sizes > self.max_block] = 0
        return np.repeat(np.arange(len(query_keys)), sizes), self.order[expand_ranges(left, sizes)]


def exact_view_channel(view: str, max_block: int):
    def channel(queries: pd.DataFrame, reference: pd.DataFrame):
        return exact_join(queries[view], reference[view], max_block)
    channel.columns = [view]
    return channel
