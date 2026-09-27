"""
Stage 8 controlled experiments: ONE change against the frozen baseline.

    python src/analysis/run_stage8_experiments.py --experiment E001   # hard-example mining (model)
    python src/analysis/run_stage8_experiments.py --experiment E002   # name-only features v2 (features)
    python src/analysis/run_stage8_experiments.py --experiment E003   # empty-address char retrieval (blocking)
    python src/analysis/run_stage8_experiments.py --experiment E004   # street-level features v3 (features)

Every experiment is scored on the fixed 25% validation S1 subset used for the
Stage 6 model comparison, with the frozen Stage 7 decision rules, and compared
entity-by-entity with the frozen baseline on the same S1 (delta with paired
bootstrap interval, entities improved / degraded, FP / FN change). Training
data only ever comes from TRAIN S1. Results: experiments/stage8/improvement_experiments.csv
"""

from pathlib import Path
from datetime import date
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))

from analysis.experiment_runner import compare, log_experiment  # noqa: E402
from decision.decision_engine import DecisionParams, entity_outcomes, select  # noqa: E402
from decision.ranking import prepare  # noqa: E402
from evaluation.evaluator import load_ground_truth  # noqa: E402
from models import dataset as ds, hard_mining  # noqa: E402
from models import lightgbm_model as lgbm, xgboost_model as xgbm  # noqa: E402
from models.predict import predict_pairs  # noqa: E402
from models.train import SAMPLING, SEED, validation_context  # noqa: E402

STAGE8 = ROOT / "experiments" / "stage8"
FEATURES = ROOT / "artifacts" / "features"
PREDICTIONS = ROOT / "artifacts" / "predictions"
NORMALIZED = ROOT / "artifacts" / "normalized"
VALIDATION_GT = ROOT / "experiments" / "stage2" / "validation_ground_truth.tsv"
DECISION = ROOT / "experiments" / "stage7" / "decision_params.json"
BASELINE_PREDICTIONS = PREDICTIONS / "E001_lgbm_xgb_validation.parquet"
TRAIN_PAIRS = FEATURES / "train_sample_pairs.parquet"
VALIDATION_PAIRS = FEATURES / "validation_pairs.parquet"


# ============================================================
# Shared: decisions -> entity table on the comparison subset
# ============================================================

def decision_params() -> DecisionParams:
    raw = json.loads(DECISION.read_text())
    raw.pop("model")
    raw["country_t"] = {k: tuple(v) for k, v in raw["country_t"].items()}
    return DecisionParams(**raw)


def entity_table(pred: pd.DataFrame, s1_index: pd.Index, true_counts: np.ndarray, params: DecisionParams) -> pd.DataFrame:
    """pred: source1_entity_id, candidate_entity_id, probability, label (only subset S1)."""
    country = pd.read_parquet(NORMALIZED / "train_source1.parquet", columns=["entity_id", "country_key"]) \
        .set_index("entity_id")["country_key"].reindex(s1_index).to_numpy()
    code = s1_index.get_indexer(pred["source1_entity_id"]).astype(np.int64)
    cand, _ = pd.factorize(pred["candidate_entity_id"])
    scored = prepare(code, cand.astype(np.int64), pred["candidate_entity_id"].str.startswith("S2-").to_numpy(),
                     pred["probability"].to_numpy(np.float32), pred["label"].to_numpy(np.int8), country)
    o = entity_outcomes(scored, select(scored, params), true_counts)
    return pd.DataFrame({"s1_id": s1_index, "f05": o["f05"], "precision": o["precision"], "recall": o["recall"],
                         "fp": (o["n_pred"] - o["tp"]).astype(int), "fn": (true_counts - o["tp"]).astype(int)})


def baseline_table(s1_index, true_counts, params) -> pd.DataFrame:
    pred = pq.read_table(BASELINE_PREDICTIONS, filters=[("source1_entity_id", "in", list(s1_index))]).to_pandas()
    return entity_table(pred, s1_index, true_counts, params)


def record(experiment_id, change, hypothesis, component, experiment, baseline, extra_notes=""):
    result = compare(baseline, experiment)
    row = {
        "experiment_id": experiment_id, "date": date.today().isoformat(), "baseline_version": "baseline_v1",
        "change": change, "hypothesis": hypothesis, "affected_component": component,
        **{k: result[k] for k in ("s1", "baseline_macro_f05", "experiment_macro_f05", "delta_f05", "delta_p05",
                                   "delta_p95", "entities_improved", "entities_degraded", "fp_change", "fn_change")},
        "precision": round(float(experiment["precision"].mean()), 6),
        "recall": round(float(experiment["recall"].mean()), 6),
        "status": "KEEP" if result["delta_p05"] > 0 else ("REJECT" if result["delta_f05"] <= 0 else "INCONCLUSIVE"),
        "notes": extra_notes,
    }
    log_experiment(STAGE8 / "improvement_experiments.csv", row)
    print(json.dumps(row, indent=2))
    return row


# ============================================================
# Model training shared by E001 / E002
# ============================================================

def train_ensemble(sample: pd.DataFrame, features: list[str]):
    train, dev = ds.split_by_s1(sample, dev_fraction=0.1, seed=SEED)
    features = [f for f in features if train[f].nunique(dropna=False) > 1]
    w_train, w_dev = train["sample_weight"].to_numpy(), dev["sample_weight"].to_numpy()
    lgb_model = lgbm.train_lightgbm(train[features], train["label"], dev[features], dev["label"], w_train, w_dev)
    xgb_model = xgbm.train_xgboost(train[features], train["label"], dev[features], dev["label"], w_train, w_dev)

    def predict(X):
        return (lgbm.predict(lgb_model, X[features]) + xgbm.predict(xgb_model, X[features])) / 2

    return predict, features, {"lgbm_best_iteration": lgb_model.best_iteration_,
                               "xgb_best_iteration": xgb_model.best_iteration}


def score_subset(pairs_path: Path, predict, features, s1_index) -> pd.DataFrame:
    pred = predict_pairs(pairs_path, features, predict, s1_index, subset=True)
    label = pred["label"]
    frame = pq.read_table(pairs_path, columns=["source1_entity_id", "candidate_entity_id"],
                          filters=[("source1_entity_id", "in", list(s1_index))]).to_pandas()
    if len(frame) != len(label):
        raise ValueError("prediction rows do not align with the pair file subset")
    return frame.assign(probability=pred["prob"], label=label)


# ============================================================
# Experiments
# ============================================================

def e001(s1_index, true_counts, params, baseline):
    sample = ds.sample_training_pairs(TRAIN_PAIRS, SAMPLING["hard_rate"], SAMPLING["medium_rate"],
                                      SAMPLING["easy_rate"], SEED, keep_candidate_ids=True)
    features = ds.feature_columns(TRAIN_PAIRS)
    quick = hard_mining.fit_quick_folds(sample, features)
    hard = hard_mining.mine(TRAIN_PAIRS, features, quick)
    hard.to_parquet(STAGE8 / "errors" / "train_hard_examples.parquet")
    augmented, info = hard_mining.add_hard_examples(sample, hard)
    predict, used, fit_info = train_ensemble(augmented.drop(columns=["candidate_entity_id"]), features)
    experiment = entity_table(score_subset(VALIDATION_PAIRS, predict, used, s1_index), s1_index, true_counts, params)
    return record("E001", "hard-example mining from TRAIN S1 (out-of-fold, boost 3)",
                  "features/model FNs and confident FPs come from under-represented hard pairs",
                  "model", experiment, baseline, json.dumps({**info, **fit_info}))


def e002(s1_index, true_counts, params, baseline):
    from features.feature_builder import build_pair_features
    from features.run_stage5 import CHANNELS, QUERY_SETS, ground_truth_for
    train_v2 = FEATURES / "train_sample_pairs_v2.parquet"
    valid_v2 = FEATURES / "validation_subset_pairs_v2.parquet"
    candidates = ROOT / "artifacts" / "candidates"
    if not train_v2.exists():
        build_pair_features(candidates / "train_sample_candidates.parquet", [NORMALIZED / "train_source1.parquet"],
                            [NORMALIZED / "train_source2.parquet", NORMALIZED / "train_source3.parquet"],
                            QUERY_SETS["train_sample"]["translations"],
                            ground_truth_for("train_sample", candidates / "train_sample_candidates.parquet"),
                            train_v2, CHANNELS, feature_set="v2")
    if not valid_v2.exists():
        build_pair_features(candidates / "validation_candidates.parquet", [NORMALIZED / "train_source1.parquet"],
                            [NORMALIZED / "train_source2.parquet", NORMALIZED / "train_source3.parquet"],
                            QUERY_SETS["validation"]["translations"], load_ground_truth(VALIDATION_GT),
                            valid_v2, CHANNELS, feature_set="v2", s1_ids=set(s1_index))
    sample = ds.sample_training_pairs(train_v2, SAMPLING["hard_rate"], SAMPLING["medium_rate"],
                                      SAMPLING["easy_rate"], SEED)
    predict, used, fit_info = train_ensemble(sample, ds.feature_columns(train_v2))
    experiment = entity_table(score_subset(valid_v2, predict, used, s1_index), s1_index, true_counts, params)
    return record("E002", "feature set v2: name-only evidence for address-less candidates",
                  "26% of missed true pairs have an empty candidate address; name evidence needs typo/leet-robust signals",
                  "features", experiment, baseline, json.dumps(fit_info))


def e004(s1_index, true_counts, params, baseline):
    """v3 = v2 + street features; needs features/add_street_features.py run first."""
    train_v3 = FEATURES / "train_sample_pairs_v3.parquet"
    valid_v3 = FEATURES / "validation_subset_pairs_v3.parquet"
    if not (train_v3.exists() and valid_v3.exists()):
        subprocess.run([sys.executable, "-u", str(SRC / "features" / "add_street_features.py")], check=True)
    sample = ds.sample_training_pairs(train_v3, SAMPLING["hard_rate"], SAMPLING["medium_rate"],
                                      SAMPLING["easy_rate"], SEED)
    predict, used, fit_info = train_ensemble(sample, ds.feature_columns(train_v3))
    pred = score_subset(valid_v3, predict, used, s1_index)
    pred.to_parquet(STAGE8 / "E004_validation_subset_predictions.parquet")
    experiment = entity_table(pred, s1_index, true_counts, params)
    return record("E004", "feature set v3: v2 + street-level address evidence + dotted legal forms",
                  "siblings (same name / house number / city, different street + legal form) get high "
                  "whole-address overlap; France test collisions 4% of refs vs 0.09% on validation",
                  "features", experiment, baseline, json.dumps(fit_info))


def e003():
    """Retrieval-level measurement on the TRAIN tuning sample (validation untouched)."""
    channels = "content_name,fingerprint_name,fingerprint_address,hybrid_translated,char_name_no_address"
    subprocess.run([sys.executable, "-u", str(SRC / "blocking" / "run_stage4.py"), "--queries", "tuning",
                    "--tag", "E003_char_no_address", "--channels", channels], check=True)
    table = pd.read_csv(ROOT / "experiments" / "stage4" / "tuning" / "E003_char_no_address" / "candidate_recall.csv")
    with_channel = table[table["experiment"] == "UNION (all channels)"].iloc[0]
    without = table[table["experiment"] == "union without char_name_no_address"].iloc[0]
    row = {
        "experiment_id": "E003", "date": date.today().isoformat(), "baseline_version": "baseline_v1",
        "change": "blocking: char 3-4gram TF-IDF over names of address-less references (top-20)",
        "hypothesis": "42% of missed true pairs are never retrieved, concentrated on address-less records",
        "affected_component": "blocking",
        "candidate_recall": round(float(with_channel["pair_recall"]), 6),
        "baseline_candidate_recall": round(float(without["pair_recall"]), 6),
        "ceiling_gain": round(float(with_channel["f05_ceiling"] - without["f05_ceiling"]), 6),
        "extra_candidates_per_s1": round(float(with_channel["cand_mean"] - without["cand_mean"]), 2),
        "status": "PROMOTE_TO_FULL_TEST" if with_channel["f05_ceiling"] - without["f05_ceiling"] >= 0.001 else "REJECT",
        "notes": "measured on 50k train-split S1 (tuning sample); full effect needs features + model on the new pairs",
    }
    log_experiment(STAGE8 / "improvement_experiments.csv", row)
    print(json.dumps(row, indent=2))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=["E001", "E002", "E003", "E004"])
    args = parser.parse_args()
    (STAGE8 / "errors").mkdir(parents=True, exist_ok=True)
    if args.experiment == "E003":
        e003()
        return
    params = decision_params()
    s1_index, true_counts, _ = validation_context(0.25)
    baseline = baseline_table(s1_index, true_counts, params)
    {"E001": e001, "E002": e002, "E004": e004}[args.experiment](s1_index, true_counts, params, baseline)


if __name__ == "__main__":
    main()
