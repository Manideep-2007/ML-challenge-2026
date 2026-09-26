"""
Address-number blocking: candidates share at least one address number
(canonical integers from Stage 3, so "00930" == "930"). Numbers shared by
more than max_block reference records (e.g. "1", "2") are skipped.
"""

import numpy as np
import pandas as pd

from .exact_blocks import exact_join


def explode_tokens(text: pd.Series, min_length: int) -> tuple[np.ndarray, pd.Series]:
    """(row position, token) for each distinct token of length >= min_length."""
    rows, tokens = [], []
    for position, value in enumerate(text.tolist()):
        for token in set(value.split()):
            if len(token) >= min_length:
                rows.append(position)
                tokens.append(token)
    return np.asarray(rows, dtype=np.int64), pd.Series(tokens, dtype="str")


def number_channel(view: str, max_block: int, min_digits: int = 1):
    def channel(queries: pd.DataFrame, reference: pd.DataFrame):
        q_rows, q_tokens = explode_tokens(queries[view], min_digits)
        r_rows, r_tokens = explode_tokens(reference[view], min_digits)
        q, r, stats = exact_join(q_tokens, r_tokens, max_block)
        return q_rows[q], r_rows[r], stats
    channel.columns = [view]
    return channel
