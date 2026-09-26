"""
Probability-average ensemble of already evaluated models.

    python src/models/ensemble.py --id E001_lgbm_xgb --members M001_all_weighted,M008_xgboost_all_weighted

Member validation prediction files are row-aligned (all stream the same pair
file), so the ensemble probability is a plain mean. The ensemble directory
holds only its member list; inference loads each member.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))

from models.model_diagnostics import pair_metrics, probability_separation, threshold_sweep  # noqa: E402
from models.model_registry import append_comparison  # noqa: E402
from models.train import STAGE6, PREDICTIONS_DIR, validation_context  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--members", required=True)
    args = parser.parse_args()
    members = args.members.split(",")

    tables = [pq.read_table(PREDICTIONS_DIR / f"{m}_validation.parquet") for m in members]
    base = tables[0]
    for t in tables[1:]:
        if t.num_rows != base.num_rows or not t["candidate_entity_id"].equals(base["candidate_entity_id"]):
            raise ValueError("member prediction files are not row-aligned")
    prob = np.mean([t["probability"].to_numpy() for t in tables], axis=0).astype(np.float32)
    pq.write_table(pa.table({
        "source1_entity_id": base["source1_entity_id"], "candidate_entity_id": base["candidate_entity_id"],
        "probability": prob, "label": base["label"],
    }), PREDICTIONS_DIR / f"{args.id}_validation.parquet", compression="zstd")

    s1_index, true_counts, groups = validation_context(1.0)
    pred = {"code": s1_index.get_indexer(base["source1_entity_id"].to_pandas()).astype(np.int32),
            "label": base["label"].to_numpy(), "prob": prob}
    sweep = threshold_sweep(pred, true_counts, groups=groups)
    best = sweep.loc[sweep["macro_f05"].idxmax()]
    metrics = {
        "model_id": args.id, "members": members, "evaluation_scope": "all validation S1",
        **pair_metrics(pred),
        "best_threshold": float(best["threshold"]),
        "validation_macro_f05": float(best["macro_f05"]),
        "validation_precision": float(best["macro_precision"]),
        "validation_recall": float(best["macro_recall"]),
        "singleton_f05": float(best["singleton_f05"]),
        "probability_separation": probability_separation(pred),
    }
    run_dir = STAGE6 / args.id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"members": members}, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    sweep.to_csv(run_dir / "threshold_sweep.csv", index=False)
    append_comparison(STAGE6 / "model_final_validation.csv", {
        "model_id": args.id, "evaluated_s1": len(s1_index), "model": "mean(" + "+".join(members) + ")",
        "pair_auc": metrics["pair_auc"], "pair_logloss": metrics["pair_logloss"],
        "best_threshold": metrics["best_threshold"],
        "validation_macro_f05": round(metrics["validation_macro_f05"], 6),
        "validation_precision": round(metrics["validation_precision"], 6),
        "validation_recall": round(metrics["validation_recall"], 6),
        "singleton_f05": round(metrics["singleton_f05"], 6),
    })
    print(json.dumps({k: metrics[k] for k in ("validation_macro_f05", "best_threshold", "validation_precision",
                                              "validation_recall", "singleton_f05", "pair_auc")}, indent=2))


if __name__ == "__main__":
    main()
