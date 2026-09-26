"""
Stage 6 experiment runner.

Trains on a stratified negative sample of the train_sample pairs (internal
dev split by S1 for early stopping / calibration), scores the untouched
Stage 2 validation pairs, and compares models by entity-level macro F0.5
over a threshold sweep. Validation is never used for fitting.
"""

from pathlib import Path
import argparse
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))

from evaluation.evaluator import load_ground_truth  # noqa: E402
from models import dataset as ds  # noqa: E402
from models import lightgbm_model as lgbm  # noqa: E402
from models import xgboost_model as xgbm  # noqa: E402
from models.calibration import Calibrator  # noqa: E402
from models.model_diagnostics import pair_metrics, probability_separation, threshold_sweep  # noqa: E402
from models.model_registry import append_comparison, save_run  # noqa: E402
from models.predict import predict_pairs  # noqa: E402

FEATURES_DIR = ROOT / "artifacts" / "features"
PREDICTIONS_DIR = ROOT / "artifacts" / "predictions"
STAGE6 = ROOT / "experiments" / "stage6"
VALIDATION_GT = ROOT / "experiments" / "stage2" / "validation_ground_truth.tsv"
S1_NORMALIZED = ROOT / "artifacts" / "normalized" / "train_source1.parquet"
SEED = 42

TRAIN_PAIRS = FEATURES_DIR / "train_sample_pairs.parquet"
VALIDATION_PAIRS = FEATURES_DIR / "validation_pairs.parquet"

# Blocked candidates are hard by construction: 12.2M hard / 12.9M medium /
# 2.1M easy negatives vs 0.68M positives in the train sample, so hard
# negatives are sampled too, at the highest rate.
SAMPLING = {"hard_rate": 0.20, "medium_rate": 0.06, "easy_rate": 0.04}

EXPERIMENTS = {
    "M001_all_weighted": {"groups": None, "weighted": True, "model": "lightgbm"},
    "M002_all_hardneg_unweighted": {"groups": None, "weighted": False, "model": "lightgbm"},
    "M003_name_only": {"groups": ["name"], "weighted": True, "model": "lightgbm"},
    "M004_name_address": {"groups": ["name", "address", "cross_field"], "weighted": True, "model": "lightgbm"},
    "M005_name_address_numeric": {"groups": ["name", "address", "cross_field", "numeric"], "weighted": True, "model": "lightgbm"},
    "M006_all_but_blocking": {"groups": ["name", "address", "cross_field", "numeric", "country_missing"], "weighted": True, "model": "lightgbm"},
    "M007_all_weighted_isotonic": {"groups": None, "weighted": True, "model": "lightgbm", "calibration": "isotonic"},
    "M008_xgboost_all_weighted": {"groups": None, "weighted": True, "model": "xgboost"},
}


def load_training(sampling: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    t = time.time()
    counts = ds.bucket_counts(TRAIN_PAIRS)
    sample = ds.sample_training_pairs(TRAIN_PAIRS, sampling["hard_rate"], sampling["medium_rate"], sampling["easy_rate"], SEED)
    train, dev = ds.split_by_s1(sample, dev_fraction=0.1, seed=SEED)
    info = {
        "full_pair_counts": counts,
        "sampled_rows": len(sample),
        "sampled_buckets": sample["difficulty_bucket"].value_counts().to_dict(),
        "train_rows": len(train), "dev_rows": len(dev),
        "train_s1": int(train["source1_entity_id"].nunique()), "dev_s1": int(dev["source1_entity_id"].nunique()),
        "s1_overlap_train_dev": len(set(train["source1_entity_id"]) & set(dev["source1_entity_id"])),
        "sampling": sampling, "seconds": round(time.time() - t, 1),
    }
    print(f"Training data: {json.dumps(info, default=str)}")
    return train, dev, info


def validation_context(fraction: float = 1.0):
    """fraction < 1: a fixed random S1 subset (seeded) used identically for every model."""
    gt = load_ground_truth(VALIDATION_GT)
    ids = np.array(sorted(gt))
    if fraction < 1.0:
        rng = np.random.default_rng(SEED)
        ids = np.sort(rng.choice(ids, size=int(len(ids) * fraction), replace=False))
    s1_index = pd.Index(ids)
    true_counts = np.array([len(gt[k]) for k in s1_index], dtype=np.float64)
    country = pd.read_parquet(S1_NORMALIZED, columns=["entity_id", "country_key"]).set_index("entity_id")["country_key"]
    groups = country.reindex(s1_index).to_numpy()
    return s1_index, true_counts, groups


def fit_experiment(model_id: str, spec: dict, train, dev, data_info, all_features):
    """Fit phase: needs only the training pairs. Saves model (+ calibrator) to the run directory."""
    features = [f for f in ds.select_features(all_features, spec["groups"])
                if train[f].nunique(dropna=False) > 1]  # constant columns carry no information
    started = time.time()
    w_train = train["sample_weight"].to_numpy() if spec["weighted"] else None
    w_dev = dev["sample_weight"].to_numpy() if spec["weighted"] else None
    trainer = lgbm.train_lightgbm if spec["model"] == "lightgbm" else xgbm.train_xgboost
    model = trainer(train[features], train["label"], dev[features], dev["label"], w_train, w_dev)
    predict_raw = lgbm.predict if spec["model"] == "lightgbm" else xgbm.predict

    calibrator = None
    if spec.get("calibration"):
        calibrator = Calibrator(spec["calibration"]).fit(predict_raw(model, dev[features]), dev["label"].to_numpy(), w_dev)

    run_dir = STAGE6 / model_id
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator, "features": features, "spec": spec,
                 "train_seconds": round(time.time() - started, 1),
                 "train_positive": int((train["label"] == 1).sum()), "train_negative": int((train["label"] == 0).sum()),
                 "data_info": data_info}, run_dir / "fitted.joblib")
    print(f"{model_id}: fitted in {time.time() - started:.0f}s, best iteration "
          f"{getattr(model, 'best_iteration_', None) or getattr(model, 'best_iteration', None)}")


def run_experiment(model_id: str, context, scope: str = "full") -> dict:
    """
    Predict/evaluate phase on the untouched validation pairs.
    scope "subset": the fixed 25% validation S1 subset (model comparison);
    scope "full": all validation S1 (the selected model).
    """
    s1_index, true_counts, groups = context
    fitted = joblib.load(STAGE6 / model_id / "fitted.joblib")
    model, calibrator, features, spec = fitted["model"], fitted["calibrator"], fitted["features"], fitted["spec"]
    data_info = fitted["data_info"]
    started = time.time()
    predict_raw = lgbm.predict if spec["model"] == "lightgbm" else xgbm.predict

    def predict_fn(X):
        p = predict_raw(model, X)
        return calibrator.transform(p) if calibrator is not None else p

    train_seconds = fitted["train_seconds"]
    suffix = "validation" if scope == "full" else "validation_subset"
    pred = predict_pairs(VALIDATION_PAIRS, features, predict_fn, s1_index,
                         out_path=PREDICTIONS_DIR / f"{model_id}_{suffix}.parquet", subset=scope != "full")
    sweep = threshold_sweep(pred, true_counts, groups=groups)
    best = sweep.loc[sweep["macro_f05"].idxmax()]

    importance = lgbm.importance(model, features) if spec["model"] == "lightgbm" else [
        {"feature": f, "gain": float(g)} for f, g in zip(features, model.feature_importances_)]
    metrics = {
        "model_id": model_id,
        "evaluation_scope": "all validation S1" if scope == "full" else f"fixed {len(s1_index):,}-S1 validation subset",
        "evaluated_s1": int(len(s1_index)),
        **pair_metrics(pred),
        "best_threshold": float(best["threshold"]),
        "validation_macro_f05": float(best["macro_f05"]),
        "validation_precision": float(best["macro_precision"]),
        "validation_recall": float(best["macro_recall"]),
        "singleton_f05": float(best["singleton_f05"]),
        "matched_f05": float(best["matched_f05"]),
        "f05_by_country": {k.removeprefix("macro_f05_"): float(v) for k, v in best.items() if k.startswith("macro_f05_")},
        "f05_at_0.5": float(sweep.loc[(sweep["threshold"] - 0.5).abs().idxmin(), "macro_f05"]),
        "probability_separation": probability_separation(pred),
        "validation_pairs": int(len(pred["prob"])),
        "train_seconds": train_seconds,
        "total_seconds": round(time.time() - started, 1),
    }
    best_iteration = getattr(model, "best_iteration_", None) or getattr(model, "best_iteration", None)
    config = {"spec": spec, "features": features, "data": data_info, "best_iteration": best_iteration,
              "params": getattr(model, "get_params", lambda: {})()}
    save_run(STAGE6 / model_id, model, config, metrics, importance, sweep, calibrator)

    board = "model_comparison.csv" if scope != "full" else "model_final_validation.csv"
    append_comparison(STAGE6 / board, {
        "model_id": model_id, "evaluated_s1": int(len(s1_index)), "model": spec["model"], "features": len(features),
        "feature_groups": "all" if spec["groups"] is None else "+".join(spec["groups"]),
        "train_pairs": data_info["train_rows"],
        "positive_pairs": fitted["train_positive"], "negative_pairs": fitted["train_negative"],
        "sampling": f"all pos; neg hard {SAMPLING['hard_rate']}, medium {SAMPLING['medium_rate']}, easy {SAMPLING['easy_rate']}",
        "class_weight": "inverse sampling rate" if spec["weighted"] else "none (hard-negative emphasis)",
        "calibration": spec.get("calibration", "none"),
        "best_iteration": best_iteration,
        "learning_rate": lgbm.BASE_PARAMS["learning_rate"] if spec["model"] == "lightgbm" else xgbm.BASE_PARAMS["learning_rate"],
        "leaves_or_depth": lgbm.BASE_PARAMS["num_leaves"] if spec["model"] == "lightgbm" else xgbm.BASE_PARAMS["max_depth"],
        "pair_auc": metrics["pair_auc"], "pair_logloss": metrics["pair_logloss"],
        "best_threshold": metrics["best_threshold"],
        "validation_macro_f05": round(metrics["validation_macro_f05"], 6),
        "validation_precision": round(metrics["validation_precision"], 6),
        "validation_recall": round(metrics["validation_recall"], 6),
        "singleton_f05": round(metrics["singleton_f05"], 6),
        "f05_at_0.5": round(metrics["f05_at_0.5"], 6),
    })
    print(f"{model_id}: macro F0.5 {metrics['validation_macro_f05']:.5f} @ {metrics['best_threshold']}  "
          f"(P {metrics['validation_precision']:.4f}, R {metrics['validation_recall']:.4f}, "
          f"singleton {metrics['singleton_f05']:.4f})  AUC {metrics['pair_auc']}  {metrics['total_seconds']}s")
    return metrics


def write_report():
    comparison = pd.read_csv(STAGE6 / "model_comparison.csv").sort_values("validation_macro_f05", ascending=False)
    best = comparison.iloc[0]["model_id"]
    metrics = json.loads((STAGE6 / best / "metrics.json").read_text())
    config = json.loads((STAGE6 / best / "config.json").read_text())
    data = config["data"]
    importance = pd.read_csv(STAGE6 / best / "feature_importance.csv").head(20)
    cols = ["model_id", "features", "pair_auc", "pair_logloss", "best_threshold", "validation_macro_f05",
            "validation_precision", "validation_recall", "singleton_f05", "f05_at_0.5", "best_iteration"]
    lines = [
        "STAGE 6 PAIRWISE MATCHER REPORT", "=" * 78,
        "Training data (train_sample candidates, split by S1; validation never used for fitting):",
        f"  full candidate pairs: {json.dumps(data['full_pair_counts'])}",
        f"  sampled rows: {data['sampled_rows']:,}  buckets: {json.dumps(data['sampled_buckets'])}",
        f"  train / dev rows: {data['train_rows']:,} / {data['dev_rows']:,}   "
        f"train / dev S1: {data['train_s1']:,} / {data['dev_s1']:,}   S1 overlap: {data['s1_overlap_train_dev']}",
        f"  sampling: {json.dumps(data['sampling'])}",
        "", f"Model comparison (entity-level macro F0.5, best global threshold, identical fixed "
            f"{int(comparison['evaluated_s1'].iloc[0]):,}-S1 validation subset):",
        comparison[cols].to_string(index=False),
    ]
    final_path = STAGE6 / "model_final_validation.csv"
    if final_path.exists():
        final = pd.read_csv(final_path)
        lines += ["", "Selected model(s) on ALL validation S1:", final[cols].to_string(index=False)]
    lines += [
        "", f"Best on the comparison subset: {best}  ({metrics['evaluation_scope']})",
        f"  F0.5 by country: {metrics['f05_by_country']}",
        f"  probability separation: {json.dumps(metrics['probability_separation'])}",
        "", "Top 20 features by gain:",
        importance[["feature", "gain_pct"]].to_string(index=False) if "gain_pct" in importance else importance.to_string(index=False),
        "", "Stage 7 turns these probabilities into per-S1 decisions (threshold search, rules, exclusivity).",
    ]
    (STAGE6 / "stage6_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--phase", choices=["fit", "evaluate", "all"], default="all")
    parser.add_argument("--scope", choices=["subset", "full"], default="subset",
                        help="subset: fixed 25%% of validation S1 for model comparison; full: all validation S1")
    args = parser.parse_args()
    STAGE6.mkdir(parents=True, exist_ok=True)
    experiments = args.experiments.split(",")

    if args.phase in ("fit", "all"):
        train, dev, data_info = load_training(SAMPLING)
        all_features = ds.feature_columns(TRAIN_PAIRS)
        for model_id in experiments:
            fit_experiment(model_id, EXPERIMENTS[model_id], train, dev, data_info, all_features)
        del train, dev

    if args.phase in ("evaluate", "all"):
        context = validation_context(1.0 if args.scope == "full" else 0.25)
        for model_id in experiments:
            run_experiment(model_id, context, args.scope)
        write_report()


if __name__ == "__main__":
    main()
