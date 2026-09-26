"""
Candidate ranking per S1: pairs sorted by (S1, -probability) with rank,
best / second-best probability and rank within (S1, source).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Scored:
    code: np.ndarray         # S1 index per pair
    cand: np.ndarray         # candidate record index (for exclusivity)
    is_s2: np.ndarray        # bool
    prob: np.ndarray
    label: np.ndarray        # int8 (0 when unknown, e.g. test)
    country: np.ndarray      # per-S1 country label, indexed by code
    rank: np.ndarray         # rank within S1 (0 = best)
    best: np.ndarray         # best probability of the pair's S1
    second: np.ndarray       # second-best probability of the pair's S1 (0 if none)
    source_rank: np.ndarray  # rank within (S1, source)
    row: np.ndarray          # position of the pair in the original (unfiltered) arrays


def _group_rank(code: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]]) if len(code) else np.empty(0, np.int64)
    lengths = np.diff(np.r_[starts, len(code)])
    rank = np.arange(len(code)) - np.repeat(starts, lengths)
    return rank, starts, lengths


def prepare(code, cand, is_s2, prob, label, country, floor: float = 0.02) -> Scored:
    """Keep pairs with prob >= floor (no decision threshold goes lower) and rank them."""
    row = np.flatnonzero(prob >= floor)
    code, cand, is_s2, prob, label = code[row], cand[row], is_s2[row], prob[row], label[row]
    order = np.lexsort((-prob, code))
    code, cand, is_s2, prob, label, row = code[order], cand[order], is_s2[order], prob[order], label[order], row[order]

    rank, starts, lengths = _group_rank(code)
    best = np.repeat(prob[starts], lengths)
    second_per_group = np.where(lengths > 1, prob[np.minimum(starts + 1, len(prob) - 1)], 0.0)
    second = np.repeat(second_per_group, lengths)

    source_rank = np.empty(len(code), dtype=np.int64)
    for flag in (True, False):
        idx = np.flatnonzero(is_s2 == flag)
        source_rank[idx] = _group_rank(code[idx])[0]
    return Scored(code, cand, is_s2, prob, label.astype(np.int8), country, rank, best, second.astype(np.float32),
                  source_rank, row)
