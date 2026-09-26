"""
Per-S1 confidence profile (top / second / third probability, margins) and a
diagnostic decision type for each S1. Decision types are for error analysis
only; they are not part of the submission.
"""

import numpy as np

from .ranking import Scored

STRONG_MARGIN = 0.2   # descriptive cut for the labels below, not a tuned rule


def s1_confidence(s: Scored, n_s1: int) -> dict:
    top = np.zeros(n_s1, dtype=np.float32)
    second = np.zeros(n_s1, dtype=np.float32)
    third = np.zeros(n_s1, dtype=np.float32)
    for r, target in ((0, top), (1, second), (2, third)):
        m = s.rank == r
        target[s.code[m]] = s.prob[m]
    rest = s.rank > 0
    total = np.bincount(s.code[rest], weights=s.prob[rest], minlength=n_s1)
    count = np.bincount(s.code[rest], minlength=n_s1)
    mean_rest = np.divide(total, count, out=np.zeros(n_s1), where=count > 0)
    return {
        "top_probability": top, "second_probability": second, "third_probability": third,
        "top_second_margin": top - second, "top_third_margin": top - third,
        "top_over_second": np.divide(top, second, out=np.full(n_s1, np.inf, dtype=np.float32), where=second > 0),
        "top_minus_mean_rest": (top - mean_rest).astype(np.float32),
    }


def decision_types(s: Scored, keep: np.ndarray, n_s1: int) -> np.ndarray:
    n_pred = np.bincount(s.code, weights=keep, minlength=n_s1)
    top_kept = np.zeros(n_s1)
    np.maximum.at(top_kept, s.code[keep], s.prob[keep])
    min_kept = np.ones(n_s1)
    np.minimum.at(min_kept, s.code[keep], s.prob[keep])
    # strongest rejected probability per S1 (gap between accepted and rejected)
    max_rejected = np.zeros(n_s1)
    np.maximum.at(max_rejected, s.code[~keep], s.prob[~keep])
    gap = min_kept - max_rejected

    labels = np.full(n_s1, "NO_MATCH", dtype=object)
    single = n_pred == 1
    multi = n_pred >= 2
    labels[single & (gap >= STRONG_MARGIN)] = "SINGLE_STRONG"
    labels[single & (gap < STRONG_MARGIN)] = "SINGLE_AMBIGUOUS"
    labels[multi & (gap >= STRONG_MARGIN)] = "MULTI_STRONG"
    labels[multi & (gap < STRONG_MARGIN)] = "MULTI_AMBIGUOUS"
    return labels
