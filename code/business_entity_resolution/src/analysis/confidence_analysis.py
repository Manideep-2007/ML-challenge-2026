"""
Do errors happen when the top candidates are tightly clustered? Compares
correct (F0.5 = 1) and imperfect entities on score shape, and bootstraps the
macro F0.5 to show how stable the headline number is.
"""

import numpy as np
import pandas as pd

COLUMNS = ["top_probability", "second_probability", "margin", "above_0.9", "above_0.95", "above_0.99", "candidate_count"]


def correct_vs_incorrect(entities: pd.DataFrame) -> pd.DataFrame:
    groups = {"correct (F0.5=1)": entities["f05"] >= 1.0, "imperfect": entities["f05"] < 1.0}
    rows = []
    for name, mask in groups.items():
        part = entities[mask]
        row = {"group": name, "s1": int(mask.sum())}
        for col in COLUMNS:
            row[f"{col}_median"] = float(part[col].median())
            row[f"{col}_mean"] = round(float(part[col].mean()), 4)
        row["share_margin_below_0.01"] = round(float((part["margin"] < 0.01).mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap(per_entity_f: np.ndarray, samples: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(per_entity_f)
    stats = np.array([per_entity_f[rng.integers(0, n, n)].mean() for _ in range(samples)])
    return {"macro_f05": float(per_entity_f.mean()), "bootstrap_samples": samples,
            "p05": float(np.percentile(stats, 5)), "median": float(np.median(stats)),
            "p95": float(np.percentile(stats, 95)), "std": float(stats.std())}
