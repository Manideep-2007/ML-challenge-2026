"""
Model evaluation at the competition level.

threshold_sweep turns pair probabilities into per-S1 match sets for each
threshold and computes the exact macro F0.5 of Stage 2: recall uses each
S1's FULL ground-truth count (true matches that blocking never retrieved
count as misses), singletons score 1.0 only when nothing is predicted.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

THRESHOLDS = np.round(np.concatenate([np.arange(0.05, 0.9, 0.05), np.arange(0.9, 0.995, 0.01)]), 3)


def entity_scores(code: np.ndarray, label: np.ndarray, predicted: np.ndarray, true_counts: np.ndarray) -> dict:
    n = len(true_counts)
    n_pred = np.bincount(code, weights=predicted, minlength=n)
    tp = np.bincount(code, weights=predicted & (label == 1), minlength=n)
    precision = np.divide(tp, n_pred, out=np.zeros(n), where=n_pred > 0)
    recall = np.divide(tp, true_counts, out=np.zeros(n), where=true_counts > 0)
    denom = 0.25 * precision + recall
    f = np.divide(1.25 * precision * recall, denom, out=np.zeros(n), where=denom > 0)
    singleton = true_counts == 0
    f = np.where(singleton, (n_pred == 0).astype(float), f)
    precision = np.where(singleton & (n_pred == 0), 1.0, precision)
    recall = np.where(singleton, np.where(n_pred == 0, 1.0, 0.0), recall)
    return {
        "macro_f05": float(f.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "singleton_f05": float(f[singleton].mean()) if singleton.any() else np.nan,
        "matched_f05": float(f[~singleton].mean()),
        "mean_predicted": float(n_pred.mean()),
        "per_entity": f,
    }


def threshold_sweep(pred: dict, true_counts: np.ndarray, thresholds=THRESHOLDS, groups: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        s = entity_scores(pred["code"], pred["label"], pred["prob"] >= t, true_counts)
        row = {"threshold": float(t), **{k: round(v, 6) for k, v in s.items() if k != "per_entity"}}
        if groups is not None:
            for g in np.unique(groups):
                row[f"macro_f05_{g}"] = round(float(s["per_entity"][groups == g].mean()), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def pair_metrics(pred: dict, sample: int = 20_000_000, seed: int = 0) -> dict:
    y, p = pred["label"], pred["prob"]
    if len(y) > sample:
        idx = np.random.default_rng(seed).choice(len(y), size=sample, replace=False)
        y, p = y[idx], p[idx]
    return {
        "pair_auc": round(float(roc_auc_score(y, p)), 6),
        "pair_logloss": round(float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))), 6),
    }


def probability_separation(pred: dict) -> dict:
    q = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    out = {}
    for name, mask in [("positive", pred["label"] == 1), ("negative", pred["label"] == 0)]:
        out[name] = {f"p{int(x * 100)}": round(float(v), 5) for x, v in zip(q, np.quantile(pred["prob"][mask], q))}
    out["negatives_above_0.5"] = int(((pred["label"] == 0) & (pred["prob"] >= 0.5)).sum())
    out["positives_below_0.5"] = int(((pred["label"] == 1) & (pred["prob"] < 0.5)).sum())
    return out
