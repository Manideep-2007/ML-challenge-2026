"""
Feature quality gate: missing / infinite / constant / duplicate features,
positive vs negative distributions and single-feature separability (AUC).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

NON_FEATURES = {"source1_entity_id", "candidate_entity_id", "candidate_source", "label"}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in NON_FEATURES]


def feature_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in feature_columns(frame):
        v = frame[col].to_numpy(dtype=np.float64)
        finite = v[np.isfinite(v)]
        rows.append({
            "feature": col,
            "missing_pct": round(100 * float(np.isnan(v).mean()), 3),
            "inf_count": int(np.isinf(v).sum()),
            "zero_pct": round(100 * float((finite == 0).mean()), 3) if len(finite) else np.nan,
            "distinct_values": int(len(np.unique(finite))),
            "constant": bool(len(np.unique(finite)) <= 1),
            "mean": float(finite.mean()) if len(finite) else np.nan,
            "std": float(finite.std()) if len(finite) else np.nan,
            "min": float(finite.min()) if len(finite) else np.nan,
            "max": float(finite.max()) if len(finite) else np.nan,
        })
    return pd.DataFrame(rows)


def label_profile(frame: pd.DataFrame, label: int) -> pd.DataFrame:
    part = frame[frame["label"] == label]
    rows = []
    for col in feature_columns(frame):
        v = part[col].to_numpy(dtype=np.float64)
        finite = v[np.isfinite(v)]
        rows.append({
            "feature": col,
            "pairs": int(len(v)),
            "missing_pct": round(100 * float(np.isnan(v).mean()), 3),
            "mean": float(finite.mean()) if len(finite) else np.nan,
            "p10": float(np.percentile(finite, 10)) if len(finite) else np.nan,
            "median": float(np.median(finite)) if len(finite) else np.nan,
            "p90": float(np.percentile(finite, 90)) if len(finite) else np.nan,
        })
    return pd.DataFrame(rows)


def separability(frame: pd.DataFrame) -> pd.DataFrame:
    """Single-feature ROC AUC (NaN treated as its own lowest value) and positive/negative means."""
    y = frame["label"].to_numpy()
    rows = []
    for col in feature_columns(frame):
        v = frame[col].to_numpy(dtype=np.float64)
        filled = np.where(np.isnan(v), np.nanmin(v) - 1 if np.isfinite(np.nanmin(v)) else -1, v)
        auc = roc_auc_score(y, filled) if len(np.unique(filled)) > 1 else 0.5
        rows.append({
            "feature": col,
            "auc": round(float(auc), 5),
            "separation": round(abs(float(auc) - 0.5) * 2, 5),
            "direction": "higher=match" if auc >= 0.5 else "lower=match",
            "positive_mean": float(np.nanmean(v[y == 1])) if np.isfinite(v[y == 1]).any() else np.nan,
            "negative_mean": float(np.nanmean(v[y == 0])) if np.isfinite(v[y == 0]).any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values("separation", ascending=False)


def correlated_pairs(frame: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    cols = feature_columns(frame)
    values = frame[cols].astype(np.float64).fillna(-1.0)
    values = values.loc[:, values.std() > 0]
    corr = values.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("abs_pearson").reset_index()
    pairs.columns = ["feature_a", "feature_b", "abs_pearson"]
    return pairs[pairs["abs_pearson"] >= threshold].sort_values("abs_pearson", ascending=False)


def exact_duplicates(frame: pd.DataFrame) -> list[tuple[str, str]]:
    cols = feature_columns(frame)
    seen, dupes = {}, []
    for col in cols:
        key = frame[col].fillna(-999.0).to_numpy().tobytes()
        if key in seen:
            dupes.append((seen[key], col))
        else:
            seen[key] = col
    return dupes
