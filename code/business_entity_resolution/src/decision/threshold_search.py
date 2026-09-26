"""
Global threshold sweep at entity level (never pair level) with a stability
check: the chosen operating point must sit on a plateau, not a spike.
"""

import numpy as np
import pandas as pd

from .decision_engine import DecisionParams, entity_outcomes, select

COARSE = np.round(np.arange(0.50, 0.90, 0.05), 3)
FINE = np.round(np.arange(0.900, 0.9995, 0.001), 4)
THRESHOLDS = np.unique(np.r_[COARSE, FINE])


def sweep(s, true_counts, thresholds=THRESHOLDS, base: DecisionParams | None = None) -> pd.DataFrame:
    base = base or DecisionParams()
    rows = []
    for t in thresholds:
        params = DecisionParams(**{**base.__dict__, "t_first": float(t), "t_rest": float(t)})
        keep = select(s, params)
        o = entity_outcomes(s, keep, true_counts)
        empty_truth = true_counts == 0
        tp = keep & (s.label == 1)
        rows.append({
            "threshold": float(t),
            "macro_f05": o["f05"].mean(),
            "macro_precision": o["precision"].mean(),
            "macro_recall": o["recall"].mean(),
            "empty_truth_f05": o["f05"][empty_truth].mean(),
            "matched_f05": o["f05"][~empty_truth].mean(),
            "tp": int(tp.sum()),
            "fp": int((keep & (s.label == 0)).sum()),
            "fn": int(true_counts.sum() - tp.sum()),
            "predicted_matches": int(keep.sum()),
            "empty_predictions": int((o["n_pred"] == 0).sum()),
        })
    return pd.DataFrame(rows)


def plateau(table: pd.DataFrame, tolerance: float = 0.0005) -> dict:
    """Width of the region within `tolerance` of the best macro F0.5."""
    best = table["macro_f05"].max()
    near = table[table["macro_f05"] >= best - tolerance]
    return {
        "best_threshold": float(table.loc[table["macro_f05"].idxmax(), "threshold"]),
        "best_macro_f05": float(best),
        "plateau_tolerance": tolerance,
        "plateau_low": float(near["threshold"].min()),
        "plateau_high": float(near["threshold"].max()),
        "plateau_points": int(len(near)),
        "plateau_center": float(np.median(near["threshold"])),
    }
