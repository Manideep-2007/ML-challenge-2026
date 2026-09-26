"""
Stage 7: precision-first decision engine, tuned and analysed on validation.

    python src/decision/run_stage7.py --model M001_all_weighted

Inputs: Stage 6 validation predictions (artifacts/predictions/<model>_validation.parquet,
row-aligned with artifacts/features/validation_pairs.parquet). Test data is never used.
"""

from pathlib import Path
from dataclasses import asdict
import argparse
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))

from evaluation.evaluator import load_ground_truth  # noqa: E402
from decision import ambiguity, singleton  # noqa: E402
from decision.confidence import decision_types, s1_confidence  # noqa: E402
from decision.decision_engine import DecisionParams, entity_outcomes, score, select  # noqa: E402
from decision.ranking import prepare  # noqa: E402
from decision.threshold_search import COARSE, plateau, sweep  # noqa: E402
from decision.validation_search import EXPERIMENTS, cross_fit  # noqa: E402

PREDICTIONS = ROOT / "artifacts" / "predictions"
VALIDATION_PAIRS = ROOT / "artifacts" / "features" / "validation_pairs.parquet"
NORMALIZED = ROOT / "artifacts" / "normalized"
STAGE7 = ROOT / "experiments" / "stage7"
VALIDATION_GT = ROOT / "experiments" / "stage2" / "validation_ground_truth.tsv"
TAXONOMY_FEATURES = ["name_token_set_ratio", "address_token_set_ratio", "number_conflict",
                     "address_missing_cand", "name_cand_non_latin", "legal_conflict"]


# ============================================================
# LOAD
# ============================================================

def load(model_id: str):
    gt = load_ground_truth(VALIDATION_GT)
    s1_index = pd.Index(sorted(gt))
    true_counts = np.array([len(gt[k]) for k in s1_index], dtype=np.float64)
    country = pd.read_parquet(NORMALIZED / "train_source1.parquet", columns=["entity_id", "country_key"]) \
        .set_index("entity_id")["country_key"].reindex(s1_index).to_numpy()

    table = pq.read_table(PREDICTIONS / f"{model_id}_validation.parquet").to_pandas()
    code = s1_index.get_indexer(table["source1_entity_id"]).astype(np.int64)
    cand_code, cand_ids = pd.factorize(table["candidate_entity_id"])
    is_s2 = table["candidate_entity_id"].str.startswith("S2-").to_numpy()
    label = table["label"].to_numpy(np.int8)
    s = prepare(code, cand_code.astype(np.int64), is_s2, table["probability"].to_numpy(np.float32), label, country)
    full = {"label": label, "code": code, "is_s2": is_s2, "prob": table["probability"].to_numpy(np.float32)}
    del table
    return s, full, gt, s1_index, true_counts, country, np.asarray(cand_ids)


# ============================================================
# ANALYSES
# ============================================================

def per_source(s, keep, gt) -> pd.DataFrame:
    true_by_source = {"S2": sum(m.startswith("S2-") for v in gt.values() for m in v),
                      "S3": sum(m.startswith("S3-") for v in gt.values() for m in v)}
    rows = []
    for name, flag in (("S2", True), ("S3", False)):
        sel = keep & (s.is_s2 == flag)
        tp = int((sel & (s.label == 1)).sum())
        rows.append({"source": name, "predicted": int(sel.sum()), "tp": tp, "fp": int(sel.sum()) - tp,
                     "true_pairs": true_by_source[name],
                     "pair_precision": round(tp / max(int(sel.sum()), 1), 6),
                     "pair_recall": round(tp / max(true_by_source[name], 1), 6)})
    return pd.DataFrame(rows)


def taxonomy(s, keep, full, true_counts, retrieved_true: int, total_true: int):
    """False-positive / false-negative categories from pair evidence (diagnostic only)."""
    selected_rows = np.sort(s.row[keep])
    fp_rows = np.sort(s.row[keep & (s.label == 0)])
    tp_s1 = np.bincount(s.code[keep & (s.label == 1)], minlength=len(true_counts))
    positives = np.flatnonzero(full["label"] == 1)
    fn_rows = positives[~np.isin(positives, selected_rows)]

    features = pq.read_table(VALIDATION_PAIRS, columns=TAXONOMY_FEATURES)

    def frame(rows):
        return features.take(rows).to_pandas().assign(row=rows, s1=full["code"][rows], prob=full["prob"][rows])

    fp = frame(fp_rows)
    name_hi, addr_hi = fp["name_token_set_ratio"].fillna(0) >= 0.9, fp["address_token_set_ratio"].fillna(0) >= 0.9
    fp["category"] = np.select(
        [true_counts[fp["s1"]] == 0, tp_s1[fp["s1"]] > 0, name_hi & addr_hi, name_hi, addr_hi, fp["prob"] < 0.95],
        ["FP_ON_EMPTY_TRUTH", "FP_MULTI_MATCH_OVERPREDICTION", "FP_NEAR_DUPLICATE", "FP_NAME_COLLISION",
         "FP_ADDRESS_COLLISION", "FP_LOW_CONFIDENCE"], "FP_OTHER")

    fn = frame(fn_rows)
    name_lo, addr_lo = fn["name_token_set_ratio"].fillna(0) < 0.6, fn["address_token_set_ratio"].fillna(0) < 0.6
    fn["category"] = np.select(
        [fn["address_missing_cand"] == 1, fn["name_cand_non_latin"] == 1, name_lo & addr_lo, name_lo, addr_lo],
        ["FN_MISSING_FIELD", "FN_TRANSLITERATION", "FN_BOTH_FIELDS_NOISY", "FN_NAME_NOISE", "FN_ADDRESS_NOISE"],
        "FN_BELOW_THRESHOLD")

    counts = pd.concat([
        fp["category"].value_counts().rename_axis("category").reset_index(name="pairs").assign(kind="false_positive"),
        fn["category"].value_counts().rename_axis("category").reset_index(name="pairs").assign(kind="false_negative"),
        pd.DataFrame([{"category": "FN_BLOCKING", "pairs": total_true - retrieved_true, "kind": "false_negative"}]),
    ], ignore_index=True)
    counts["share_of_kind"] = (counts["pairs"] / counts.groupby("kind")["pairs"].transform("sum")).round(4)
    return counts, fp, fn


def examples(errors: pd.DataFrame, cand_ids, s1_index, full_cand_codes, per_category: int = 15) -> pd.DataFrame:
    picks = errors.groupby("category", group_keys=False).apply(lambda g: g.sample(min(per_category, len(g)), random_state=0))
    picks = picks.assign(source1_entity_id=s1_index[picks["s1"]].to_numpy(),
                         candidate_entity_id=cand_ids[full_cand_codes[picks["row"]]])
    raw = []
    for split_file, id_col in [("train_source1", "source1_entity_id"), ("train_source2", "candidate_entity_id"),
                               ("train_source3", "candidate_entity_id")]:
        t = pq.read_table(NORMALIZED / f"{split_file}.parquet", columns=["entity_id", "name_raw", "address_raw"]).to_pandas()
        raw.append(t[t["entity_id"].isin(set(picks[id_col]))].set_index("entity_id"))
    s1_raw, cand_raw = raw[0], pd.concat(raw[1:])
    return picks.assign(
        s1_name=s1_raw.reindex(picks["source1_entity_id"])["name_raw"].values,
        cand_name=cand_raw.reindex(picks["candidate_entity_id"])["name_raw"].values,
        s1_address=s1_raw.reindex(picks["source1_entity_id"])["address_raw"].values,
        cand_address=cand_raw.reindex(picks["candidate_entity_id"])["address_raw"].values,
    ).drop(columns=["row", "s1"])


def entity_metrics(s, keep, gt, s1_index, cand_ids, true_counts, conf, types, country) -> pd.DataFrame:
    o = entity_outcomes(s, keep, true_counts)
    predicted = pd.Series(cand_ids[s.cand[keep]]).groupby(s.code[keep]).agg(",".join)
    return pd.DataFrame({
        "s1_id": s1_index,
        "country": country,
        "truth_ids": [",".join(gt[k]) for k in s1_index],
        "predicted_ids": predicted.reindex(np.arange(len(s1_index))).fillna("").to_numpy(),
        "num_truth": true_counts.astype(int),
        "num_predicted": o["n_pred"].astype(int),
        "tp": o["tp"].astype(int),
        "fp": (o["n_pred"] - o["tp"]).astype(int),
        "fn": (true_counts - o["tp"]).astype(int),
        "precision": o["precision"].round(6), "recall": o["recall"].round(6), "f05": o["f05"].round(6),
        "top_probability": conf["top_probability"], "second_probability": conf["second_probability"],
        "top_second_margin": conf["top_second_margin"],
        "decision_type": types,
    })


def stability(s, params: DecisionParams, true_counts) -> pd.DataFrame:
    """Macro F0.5 when each threshold of the chosen point is nudged (plateau check)."""
    rows = []
    for field in ("t_first", "t_rest"):
        for delta in (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02):
            value = min(max(getattr(params, field) + delta, 0.0), 0.9999)
            p = DecisionParams(**{**asdict(params), field: value})
            rows.append({"parameter": field, "delta": delta, "value": round(value, 4),
                         "macro_f05": round(score(s, select(s, p), true_counts)["macro_f05"], 6)})
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    STAGE7.mkdir(parents=True, exist_ok=True)

    s, full, gt, s1_index, true_counts, country, cand_ids = load(args.model)
    countries = sorted(np.unique(country))
    total_true = int(true_counts.sum())
    retrieved_true = int((full["label"] == 1).sum())
    ceiling_recall = np.divide(np.bincount(full["code"][full["label"] == 1], minlength=len(true_counts)),
                               true_counts, out=np.ones(len(true_counts)), where=true_counts > 0)
    ceiling = np.where(true_counts > 0, 1.25 * ceiling_recall / (0.25 + ceiling_recall), 1.0)
    ceiling[(true_counts > 0) & (ceiling_recall == 0)] = 0.0
    print(f"{args.model}: {len(full['prob']):,} pairs, {len(s.prob):,} with p >= 0.02, {len(true_counts):,} S1; "
          f"candidate F0.5 ceiling {ceiling.mean():.6f}")

    # --- D1 global sweep + stability ------------------------------------
    table = sweep(s, true_counts)
    table.round(6).to_csv(STAGE7 / "threshold_sweep.csv", index=False)
    global_plateau = plateau(table)

    profiles = []
    for t in list(COARSE) + [0.9, 0.95, 0.97, 0.98, 0.99]:
        keep = select(s, DecisionParams(t, t))
        f = entity_outcomes(s, keep, true_counts)["f05"]
        row = {"threshold": t, "all": f.mean()}
        row.update({f"country_{c}": f[country == c].mean() for c in countries})
        row.update({f"truth_{k}": f[np.minimum(true_counts, 4) == k].mean() for k in range(5)})
        profiles.append(row)
    pd.DataFrame(profiles).round(6).to_csv(STAGE7 / "threshold_profiles.csv", index=False)

    # --- D1..D5 with cross-fitting ---------------------------------------
    experiments = []
    results = {}
    for name, families in EXPERIMENTS.items():
        print(f"  {name}:")
        r = cross_fit(s, true_counts, countries, families, log=print)
        results[name] = r
        experiments.append({"experiment": name, "families": "+".join(families),
                            "out_of_fold_macro_f05": round(r["out_of_fold_macro_f05"], 6),
                            "fold_0": r["out_of_fold_by_fold"][0], "fold_1": r["out_of_fold_by_fold"][1],
                            "final_in_sample_macro_f05": round(r["final_in_sample_macro_f05"], 6),
                            "final_params": json.dumps(asdict(r["final_params"]))})
        print(f"    out-of-fold {r['out_of_fold_macro_f05']:.6f}   in-sample {r['final_in_sample_macro_f05']:.6f}")
    pd.DataFrame(experiments).to_csv(STAGE7 / "decision_experiments.csv", index=False)

    best_name = max(results, key=lambda k: results[k]["out_of_fold_macro_f05"])
    params = results[best_name]["final_params"]
    keep = select(s, params)
    stable = stability(s, params, true_counts)
    stable.to_csv(STAGE7 / "operating_point_stability.csv", index=False)

    # --- per-entity analysis ---------------------------------------------
    conf = s1_confidence(s, len(true_counts))
    types = decision_types(s, keep, len(true_counts))
    outcomes = entity_outcomes(s, keep, true_counts)
    entity_metrics(s, keep, gt, s1_index, cand_ids, true_counts, conf, types, country).to_csv(
        STAGE7 / "entity_metrics.csv", index=False)

    ambiguity_table = pd.concat([
        ambiguity.by_margin(conf["top_second_margin"], outcomes, true_counts).assign(view="top_second_margin"),
        ambiguity.by_group(types, outcomes, true_counts, "decision_type").assign(view="decision_type"),
    ], ignore_index=True)
    ambiguity_table.round(6).to_csv(STAGE7 / "ambiguity_analysis.csv", index=False)

    singleton_table = pd.concat([
        singleton.best_score_profile(conf["top_probability"], true_counts).assign(view="best_score_quantiles"),
        singleton.empty_truth_false_merge_rate(conf["top_probability"], true_counts,
                                               [0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]).assign(view="no_match_threshold"),
    ], ignore_index=True)
    singleton_table.to_csv(STAGE7 / "singleton_analysis.csv", index=False)

    groups = pd.concat([
        ambiguity.by_group(country, outcomes, true_counts, "group").assign(view="country"),
        ambiguity.by_group(np.where(true_counts >= 4, "4+", true_counts.astype(int).astype(str)), outcomes,
                           true_counts, "group").assign(view="true_match_count"),
    ], ignore_index=True)
    groups.round(6).to_csv(STAGE7 / "entity_group_metrics.csv", index=False)
    sources = per_source(s, keep, gt)
    sources.to_csv(STAGE7 / "source_metrics.csv", index=False)

    tax, fp, fn = taxonomy(s, keep, full, true_counts, retrieved_true, total_true)
    tax.to_csv(STAGE7 / "error_taxonomy.csv", index=False)
    table_codes = pd.factorize(pq.read_table(PREDICTIONS / f"{args.model}_validation.parquet",
                                             columns=["candidate_entity_id"])["candidate_entity_id"].to_pandas())[0]
    examples(fp, cand_ids, s1_index, table_codes).to_csv(STAGE7 / "false_positive_examples.csv", index=False)
    examples(fn, cand_ids, s1_index, table_codes).to_csv(STAGE7 / "false_negative_examples.csv", index=False)

    # --- operating point + report ------------------------------------------
    final = score(s, keep, true_counts)
    point = {
        "model": args.model, "experiment": best_name, "params": asdict(params),
        "out_of_fold_macro_f05": results[best_name]["out_of_fold_macro_f05"],
        "in_sample_macro_f05": final["macro_f05"],
        "macro_precision": final["macro_precision"], "macro_recall": final["macro_recall"],
        "empty_truth_f05": final["empty_truth_f05"], "matched_f05": final["matched_f05"],
        "candidate_f05_ceiling": float(ceiling.mean()),
        "global_threshold_plateau": global_plateau,
        "stability_min_macro_f05_within_0.01": float(stable[stable["delta"].abs() <= 0.01]["macro_f05"].min()),
        "note": "Exclusivity on validation only arbitrates among validation S1; on test every competing S1 is present.",
    }
    (STAGE7 / "best_operating_point.json").write_text(json.dumps(point, indent=2, default=str), encoding="utf-8")
    (STAGE7 / "decision_params.json").write_text(json.dumps({"model": args.model, **asdict(params)}, indent=2), encoding="utf-8")

    lines = [
        "STAGE 7 DECISION ENGINE REPORT", "=" * 78,
        f"Model: {args.model}   validation S1: {len(true_counts):,}   pairs: {len(full['prob']):,}",
        f"Candidate F0.5 ceiling (perfect decisions on these candidates): {ceiling.mean():.6f}",
        "", "D1 global threshold sweep (entity-level macro F0.5):",
        f"  best {global_plateau['best_macro_f05']:.6f} at {global_plateau['best_threshold']}; within 0.0005 of best for "
        f"thresholds {global_plateau['plateau_low']}-{global_plateau['plateau_high']} ({global_plateau['plateau_points']} points)",
        "", "Decision experiments (2-fold cross-fitted on validation S1; out-of-fold is the honest estimate):",
        pd.DataFrame(experiments)[["experiment", "out_of_fold_macro_f05", "fold_0", "fold_1", "final_in_sample_macro_f05"]].to_string(index=False),
        "", f"Chosen: {best_name}  params {json.dumps(asdict(params))}",
        f"  macro F0.5 {final['macro_f05']:.6f}  precision {final['macro_precision']:.6f}  recall {final['macro_recall']:.6f}",
        f"  empty-truth S1 F0.5 {final['empty_truth_f05']:.6f}   matched S1 F0.5 {final['matched_f05']:.6f}",
        "", "Stability (nudging each threshold):", stable.to_string(index=False),
        "", "By group:", groups.round(4).to_string(index=False),
        "", "By source (pair level):", sources.to_string(index=False),
        "", "Ambiguity (by top-second margin and decision type):", ambiguity_table.round(4).to_string(index=False),
        "", "Empty-truth protection:", singleton_table.to_string(index=False),
        "", "Error taxonomy:", tax.to_string(index=False),
        "", point["note"],
    ]
    (STAGE7 / "stage7_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
