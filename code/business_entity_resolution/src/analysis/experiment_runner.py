"""
Controlled comparison of an experiment against the frozen baseline on the same
validation S1 entities: headline delta, how many entities improved / degraded,
FP / FN added and removed, and a paired bootstrap interval for the delta.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def compare(baseline: pd.DataFrame, experiment: pd.DataFrame, samples: int = 1000, seed: int = 0) -> dict:
    """baseline / experiment: entity tables (s1_id, f05, fp, fn) for the same S1 set."""
    b = baseline.set_index("s1_id")
    e = experiment.set_index("s1_id").reindex(b.index)
    delta = (e["f05"] - b["f05"]).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(delta)
    boot = np.array([delta[rng.integers(0, n, n)].mean() for _ in range(samples)])
    return {
        "s1": int(n),
        "baseline_macro_f05": round(float(b["f05"].mean()), 6),
        "experiment_macro_f05": round(float(e["f05"].mean()), 6),
        "delta_f05": round(float(delta.mean()), 6),
        "delta_p05": round(float(np.percentile(boot, 5)), 6),
        "delta_p95": round(float(np.percentile(boot, 95)), 6),
        "entities_improved": int((delta > 1e-9).sum()),
        "entities_degraded": int((delta < -1e-9).sum()),
        "entities_unchanged": int((np.abs(delta) <= 1e-9).sum()),
        "fp_change": int(e["fp"].sum() - b["fp"].sum()),
        "fn_change": int(e["fn"].sum() - b["fn"].sum()),
    }


def log_experiment(path: Path, row: dict):
    frame = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["experiment_id"] != row["experiment_id"]]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)
