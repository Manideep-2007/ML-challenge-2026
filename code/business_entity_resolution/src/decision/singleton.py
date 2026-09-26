"""
Empty-ground-truth protection analysis: one false match on an S1 whose truth
is empty turns its score from 1.0 to 0.0. How well does the best candidate
probability separate truly empty S1 from matched S1?
"""

import numpy as np
import pandas as pd

QUANTILES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def best_score_profile(top_probability: np.ndarray, true_counts: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, mask in [("empty_truth", true_counts == 0), ("single_truth", true_counts == 1),
                       ("multi_truth", true_counts >= 2)]:
        q = np.quantile(top_probability[mask], QUANTILES)
        rows.append({"group": name, "s1": int(mask.sum()), **{f"best_p{int(x * 100)}": round(float(v), 5) for x, v in zip(QUANTILES, q)}})
    return pd.DataFrame(rows)


def empty_truth_false_merge_rate(top_probability: np.ndarray, true_counts: np.ndarray, thresholds) -> pd.DataFrame:
    """For each no-match threshold: share of empty-truth S1 that would get a false match
    and share of matched S1 that would be (wrongly) left empty."""
    empty = true_counts == 0
    return pd.DataFrame([{
        "no_match_threshold": float(t),
        "empty_truth_false_merge_rate": round(float((top_probability[empty] >= t).mean()), 6),
        "matched_left_empty_rate": round(float((top_probability[~empty] < t).mean()), 6),
    } for t in thresholds])
