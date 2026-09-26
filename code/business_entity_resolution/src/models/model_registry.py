"""Saves each model run and appends it to the comparison scoreboard."""

from pathlib import Path
import json

import joblib
import pandas as pd


def save_run(run_dir: Path, model, config: dict, metrics: dict, importance: list[dict], sweep: pd.DataFrame,
             calibrator=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    if calibrator is not None:
        joblib.dump(calibrator, run_dir / "calibrator.joblib")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(importance).to_csv(run_dir / "feature_importance.csv", index=False)
    sweep.to_csv(run_dir / "threshold_sweep.csv", index=False)
    booster = getattr(model, "booster_", None)
    if booster is not None:
        booster.save_model(str(run_dir / "model.txt"))
    else:
        model.save_model(str(run_dir / "model.json"))


def append_comparison(path: Path, row: dict):
    frame = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["model_id"] != row["model_id"]]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)
