"""
Stage 8: deep error analysis of the frozen baseline on validation.

    python src/analysis/run_stage8.py

Freezes the baseline (experiments/stage8/baseline.json, never overwritten),
classifies every validation error, ranks root causes by macro-F0.5 actually
lost, and analyses subgroups, confidence and hard cases. Test data is not used.
"""

from pathlib import Path
from dataclasses import asdict
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))

from evaluation.evaluator import load_ground_truth  # noqa: E402
from decision.decision_engine import DecisionParams  # noqa: E402
from analysis import confidence_analysis, false_negative_analysis, false_positive_analysis  # noqa: E402
from analysis import hard_case_mining, root_cause, subgroup_analysis  # noqa: E402
from analysis.error_analyzer import build  # noqa: E402

STAGE8 = ROOT / "experiments" / "stage8"
NORMALIZED = ROOT / "artifacts" / "normalized"
VALIDATION_GT = ROOT / "experiments" / "stage2" / "validation_ground_truth.tsv"
DECISION = ROOT / "experiments" / "stage7" / "decision_params.json"


def decision_params() -> tuple[str, DecisionParams]:
    raw = json.loads(DECISION.read_text())
    model = raw.pop("model")
    raw["country_t"] = {k: tuple(v) for k, v in raw["country_t"].items()}
    return model, DecisionParams(**raw)


def freeze_baseline(model: str, params: DecisionParams, entities: pd.DataFrame, truth: pd.DataFrame) -> dict:
    path = STAGE8 / "baseline.json"
    if path.exists():
        return json.loads(path.read_text())
    baseline = {
        "name": "baseline_v1",
        "model": f"{model} (mean of LightGBM M001_all_weighted and XGBoost M008_xgboost_all_weighted)",
        "normalization": "stage3 multi-view, deterministic (token_features sorted + ref-first vocabulary)",
        "blocking": "stage4: content_name + fingerprint_name + fingerprint_address + hybrid_translated "
                    "(max_df 50000, top-100, learned translation), country-scoped",
        "features": "stage5: 110 columns (102 non-constant used)",
        "decision": asdict(params),
        "threshold": params.t_first,
        "macro_f05": round(float(entities["f05"].mean()), 6),
        "macro_precision": round(float(entities["precision"].mean()), 6),
        "macro_recall": round(float(entities["recall"].mean()), 6),
        "candidate_pair_recall": round(float(truth["retrieved"].mean()), 6),
        "validation_s1": int(len(entities)),
    }
    path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return baseline


def situation(components: pd.DataFrame) -> str:
    loss = components.set_index("component")["f05_points"]
    blocking = loss.get("blocking", 0.0)
    model = loss.get("features/model", 0.0) + loss.get("features/model + decision", 0.0)
    decision = loss.get("decision threshold", 0.0) + loss.get("decision rule", 0.0)
    largest = max(("A", blocking), ("B", model), ("C", decision), key=lambda x: x[1])[0]
    return {
        "A": "A - candidate recall is the largest loss: improve Stage 4",
        "B": "B - retrieved true matches are mis-scored / wrong candidates accepted: improve Stage 5/6",
        "C": "C - the model ranks well but decisions lose points: improve Stage 7",
    }[largest]


def main():
    for sub in ("errors", "subgroups"):
        (STAGE8 / sub).mkdir(parents=True, exist_ok=True)

    model, params = decision_params()
    gt = load_ground_truth(VALIDATION_GT)
    country = pd.read_parquet(NORMALIZED / "train_source1.parquet", columns=["entity_id", "country_key"]).set_index("entity_id")["country_key"]
    tables = build(ROOT / "artifacts" / "predictions" / f"{model}_validation.parquet",
                   ROOT / "artifacts" / "features" / "validation_pairs.parquet", gt, country, params)
    entities, pairs, truth = tables.entities, tables.pairs, tables.truth
    baseline = freeze_baseline(model, params, entities, truth)
    print(f"baseline macro F0.5 {entities['f05'].mean():.6f} (frozen: {baseline['macro_f05']})")

    # --- error classification -------------------------------------------
    fn = false_negative_analysis.classify(truth, params.t_rest)
    fp = false_positive_analysis.classify(pairs, entities)
    causes = root_cause.rank(fn, fp, entities)
    components = root_cause.by_component(causes)
    causes.to_csv(STAGE8 / "root_causes.csv", index=False)
    components.to_csv(STAGE8 / "root_cause_components.csv", index=False)

    dominant = pd.concat([
        fn.groupby("code")["category"].agg(lambda s: s.value_counts().index[0]),
        fp.groupby("code")["category"].agg(lambda s: s.value_counts().index[0]),
    ]).groupby(level=0).first()
    entities["error_type"] = np.select([entities["f05"] >= 1, entities["fp"] > 0, entities["fn"] > 0],
                                       ["correct", "has_false_positive", "has_false_negative"], "correct")
    entities["root_cause"] = dominant.reindex(np.arange(len(entities))).fillna("").to_numpy()

    # --- subgroups, confidence, hard cases -------------------------------
    s1_len = pd.read_parquet(NORMALIZED / "train_source1.parquet", columns=["entity_id", "name_basic"]) \
        .set_index("entity_id")["name_basic"].reindex(tables.s1_index).str.len().reset_index(drop=True)
    subgroups = subgroup_analysis.all_subgroups(entities, s1_len)
    for name, table in subgroups.items():
        table.to_csv(STAGE8 / "subgroups" / f"{name}.csv", index=False)
    matrix = subgroup_analysis.name_address_matrix(pairs)
    matrix.to_csv(STAGE8 / "subgroups" / "name_address_matrix.csv", index=False)
    confidence = confidence_analysis.correct_vs_incorrect(entities)
    confidence.to_csv(STAGE8 / "confidence_analysis.csv", index=False)
    bootstrap = confidence_analysis.bootstrap(entities["f05"].to_numpy())

    hard = hard_case_mining.attach_raw(hard_case_mining.hard_cases(fn, params.t_rest), NORMALIZED)
    hard.to_csv(STAGE8 / "errors" / "hard_cases.csv", index=False)
    hard_case_mining.case_report(hard, STAGE8 / "errors" / "hard_case_report.txt")
    negatives = hard_case_mining.hard_negatives(fp)
    hard_case_mining.attach_raw(negatives.head(2000), NORMALIZED).to_csv(STAGE8 / "errors" / "hard_negatives.csv", index=False)
    positives = hard_case_mining.hard_positives(truth)
    hard_case_mining.attach_raw(positives.head(2000), NORMALIZED).to_csv(STAGE8 / "errors" / "hard_positives.csv", index=False)
    hard_case_mining.attach_raw(fp.sample(min(3000, len(fp)), random_state=0), NORMALIZED).to_csv(
        STAGE8 / "errors" / "false_positives.csv", index=False)
    hard_case_mining.attach_raw(fn.sample(min(3000, len(fn)), random_state=0), NORMALIZED).to_csv(
        STAGE8 / "errors" / "false_negatives.csv", index=False)
    ambiguous = entities[(entities["margin"] < 0.01) & (entities["f05"] < 1)]
    ambiguous.to_csv(STAGE8 / "errors" / "ambiguous_cases.csv", index=False)
    empty = hard_case_mining.empty_entities(entities, pairs)
    empty.head(2000).to_csv(STAGE8 / "errors" / "empty_entities.csv", index=False)
    multi = hard_case_mining.multi_match(entities)
    entities.to_csv(STAGE8 / "entity_error_table.csv", index=False)

    # --- report ---------------------------------------------------------
    verdict = situation(components)
    lines = [
        "=" * 70, "STAGE 8 ERROR ANALYSIS REPORT", "=" * 70, "",
        "BASELINE (frozen in baseline.json)", "-" * 70,
        *[f"{k}: {v}" for k, v in baseline.items()],
        f"bootstrap 90% interval of macro F0.5: {bootstrap['p05']:.6f} - {bootstrap['p95']:.6f} (std {bootstrap['std']:.6f})",
        "", "ERROR SUMMARY", "-" * 70,
        f"False positives: {len(fp):,} pairs in {fp['code'].nunique():,} S1",
        f"False negatives: {len(fn):,} pairs in {fn['code'].nunique():,} S1",
        f"Imperfect S1: {(entities['f05'] < 1).sum():,} of {len(entities):,}   "
        f"total F0.5 points lost {float((1 - entities['f05']).sum()):,.1f}",
        "", "LOSS BY PIPELINE COMPONENT (macro F0.5 recoverable if fixed completely)", "-" * 70,
        components.round(6).to_string(index=False),
        "", "TOP ROOT CAUSES", "-" * 70,
        causes.head(20).round(6).to_string(index=False),
        "", "FALSE POSITIVES by structure / evidence", "-" * 70,
        pd.crosstab(fp["structure"], fp["evidence"], margins=True).to_string(),
        f"high-confidence (p >= 0.99) false positives: {int(fp['high_confidence'].sum()):,}",
        "", "FALSE NEGATIVES by category", "-" * 70,
        fn["category"].value_counts().to_string(),
        "", "RETRIEVAL", "-" * 70,
        f"candidate pair recall: {truth['retrieved'].mean():.6f}   missed true links: {(~truth['retrieved']).sum():,}",
        "", "SUBGROUPS", "-" * 70,
        *[f"[{name}]\n{table.to_string(index=False)}\n" for name, table in subgroups.items()],
        "[multi-match S1]", json.dumps(multi, indent=2),
        "", "NAME x ADDRESS similarity (pair level)", "-" * 70, matrix.to_string(index=False),
        "", "CONFIDENCE: correct vs imperfect S1", "-" * 70, confidence.to_string(index=False),
        "", "HARD CASES", "-" * 70,
        hard["hard_type"].value_counts().to_string(),
        f"hard negatives (p >= 0.9 false positives): {len(negatives):,}",
        f"hard positives (retrieved true pairs with p < 0.5): {len(positives):,}",
        f"empty-truth S1 with a false match: {int(empty['predicted_match'].sum()):,} of {len(empty):,}",
        "", "SITUATION", "-" * 70, verdict,
        f"Highest-impact root cause: {causes.iloc[0]['root_cause']} ({causes.iloc[0]['component']}), "
        f"up to +{causes.iloc[0]['macro_f05_gain_if_fixed']:.4f} macro F0.5 if fixed completely",
    ]
    (STAGE8 / "stage8_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
