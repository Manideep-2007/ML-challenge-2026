"""
Ambiguity analysis: where is the score lost? Groups S1 entities by the
top-vs-second probability margin and by how many candidates clear the
threshold, and reports macro F0.5 / precision / recall per group.
"""

import numpy as np
import pandas as pd

MARGIN_BINS = [-0.001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]


def by_margin(margin: np.ndarray, outcomes: dict, true_counts: np.ndarray) -> pd.DataFrame:
    bins = pd.cut(margin, MARGIN_BINS)
    frame = pd.DataFrame({"margin_bin": bins, "f05": outcomes["f05"], "precision": outcomes["precision"],
                          "recall": outcomes["recall"], "n_pred": outcomes["n_pred"], "n_true": true_counts})
    out = frame.groupby("margin_bin", observed=True).agg(
        s1=("f05", "size"), macro_f05=("f05", "mean"), macro_precision=("precision", "mean"),
        macro_recall=("recall", "mean"), mean_predicted=("n_pred", "mean"), mean_true=("n_true", "mean"))
    return out.reset_index().assign(margin_bin=lambda d: d["margin_bin"].astype(str))


def by_group(labels: np.ndarray, outcomes: dict, true_counts: np.ndarray, name: str) -> pd.DataFrame:
    frame = pd.DataFrame({name: labels, "f05": outcomes["f05"], "precision": outcomes["precision"],
                          "recall": outcomes["recall"], "n_pred": outcomes["n_pred"], "n_true": true_counts})
    return frame.groupby(name).agg(
        s1=("f05", "size"), macro_f05=("f05", "mean"), macro_precision=("precision", "mean"),
        macro_recall=("recall", "mean"), mean_predicted=("n_pred", "mean"), mean_true=("n_true", "mean"),
        lost_f05_points=("f05", lambda f: float((1 - f).sum()))).reset_index()
