from pathlib import Path
import argparse
import json
import sys
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))

from evaluation.evaluator import load_ground_truth  # noqa: E402
from features.feature_builder import build_pair_features  # noqa: E402
from features import feature_validation as fv  # noqa: E402

NORMALIZED = ROOT / "artifacts" / "normalized"
CANDIDATES = ROOT / "artifacts" / "candidates"
FEATURES = ROOT / "artifacts" / "features"
STAGE4 = ROOT / "experiments" / "stage4"
STAGE5 = ROOT / "experiments" / "stage5"
GT_PATH = ROOT / "challenge" / "dataset" / "train" / "train_ground_truth.tsv"
VALIDATION_GT = ROOT / "experiments" / "stage2" / "validation_ground_truth.tsv"

CHANNELS = ["content_name", "fingerprint_name", "fingerprint_address", "hybrid_translated"]
ANALYSIS_SAMPLE = 2_000_000
SEED = 42

QUERY_SETS = {
    "validation": {"translations": STAGE4, "gt": "validation"},
    "train_sample": {"translations": STAGE4 / "train_sample", "gt": "train"},
}


def ground_truth_for(query_set: str, candidates_path: Path) -> dict:
    if QUERY_SETS[query_set]["gt"] == "validation":
        return load_ground_truth(VALIDATION_GT)
    ids = set(pq.read_table(candidates_path, columns=["source1_entity_id"])["source1_entity_id"].unique().to_pylist())
    full = load_ground_truth(GT_PATH)
    return {k: v for k, v in full.items() if k in ids}


def label_check(pairs_path: Path, ground_truth: dict) -> dict:
    """Positives must be exactly the true pairs that Stage 4 retrieved (integer pair keys)."""
    table = pq.read_table(pairs_path, columns=["source1_entity_id", "candidate_entity_id", "label"])
    s1_index = pd.Index(sorted(ground_truth))
    s1_code = s1_index.get_indexer(table["source1_entity_id"].to_pandas()).astype(np.int64)
    cand_code, cand_ids = pd.factorize(table["candidate_entity_id"].to_pandas())
    label = table["label"].to_numpy()
    del table
    n_cand = len(cand_ids)
    keys = s1_code * n_cand + cand_code.astype(np.int64)

    true_s1 = [k for k, v in ground_truth.items() for _ in v]
    true_cand = [m for v in ground_truth.values() for m in v]
    t_cand = pd.Index(cand_ids).get_indexer(true_cand).astype(np.int64)
    t_keys = s1_index.get_indexer(true_s1).astype(np.int64) * n_cand + t_cand
    retrievable = np.unique(t_keys[t_cand >= 0])

    positive_keys = np.unique(keys[label == 1])
    retrieved_true = np.intersect1d(np.unique(keys), retrievable)
    return {
        "labelled_positives": int((label == 1).sum()),
        "retrieved_true_pairs": int(len(retrieved_true)),
        "positives_match_ground_truth": bool(np.array_equal(positive_keys, retrieved_true)),
        "negatives": int((label == 0).sum()),
        "s1_entities": int(len(np.unique(s1_code))),
        "s1_entities_with_a_positive": int(len(np.unique(s1_code[label == 1]))),
        "duplicate_pairs": int(len(keys) - len(np.unique(keys))),
        "unknown_s1_ids": int((s1_code < 0).sum()),
    }


def sample_pairs(path: Path, n: int, seed: int) -> pd.DataFrame:
    table = pq.read_table(path)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(table.num_rows, size=min(n, table.num_rows), replace=False))
    return table.take(pick).to_pandas()


def feature_examples(sample: pd.DataFrame, per_label: int = 25) -> pd.DataFrame:
    """Positive and hard-negative examples with raw text for manual inspection."""
    picks = pd.concat([
        sample[sample["label"] == 1].sample(per_label, random_state=SEED).assign(kind="positive"),
        sample[(sample["label"] == 0) & (sample["name_token_set_ratio"] >= 0.9)].sample(per_label, random_state=SEED).assign(kind="hard_negative_same_name"),
        sample[(sample["label"] == 0) & (sample["address_token_set_ratio"] >= 0.9)].sample(per_label, random_state=SEED).assign(kind="hard_negative_same_address"),
        sample[(sample["label"] == 1) & (sample["name_token_set_ratio"] < 0.5)].sample(per_label, random_state=SEED).assign(kind="positive_low_name_similarity"),
    ])
    raw = []
    for path, id_col in [(NORMALIZED / "train_source1.parquet", "source1_entity_id")] + [
            (NORMALIZED / f"train_source{i}.parquet", "candidate_entity_id") for i in (2, 3)]:
        table = pq.read_table(path, columns=["entity_id", "name_raw", "address_raw"]).to_pandas()
        raw.append((id_col, table[table["entity_id"].isin(set(picks[id_col]))]))
    s1 = raw[0][1].set_index("entity_id")
    cand = pd.concat([raw[1][1], raw[2][1]]).set_index("entity_id")
    keep = ["kind", "label", "source1_entity_id", "candidate_entity_id", "name_token_set_ratio",
            "name_content_idf_jaccard", "address_token_set_ratio", "address_idf_jaccard",
            "number_conflict", "legal_conflict", "hybrid_translated_rank", "combined_idf_evidence_rank_in_s1"]
    out = picks[keep].copy()
    out.insert(4, "s1_name", s1.reindex(out["source1_entity_id"])["name_raw"].values)
    out.insert(5, "cand_name", cand.reindex(out["candidate_entity_id"])["name_raw"].values)
    out.insert(6, "s1_address", s1.reindex(out["source1_entity_id"])["address_raw"].values)
    out.insert(7, "cand_address", cand.reindex(out["candidate_entity_id"])["address_raw"].values)
    return out


def analyse(pairs_path: Path, build_info: dict, labels: dict) -> dict:
    sample = sample_pairs(pairs_path, ANALYSIS_SAMPLE, SEED)
    stats = fv.feature_statistics(sample)
    stats.to_csv(STAGE5 / "feature_statistics.csv", index=False)
    fv.label_profile(sample, 1).to_csv(STAGE5 / "positive_feature_profile.csv", index=False)
    fv.label_profile(sample, 0).to_csv(STAGE5 / "negative_feature_profile.csv", index=False)
    separability = fv.separability(sample)
    separability.to_csv(STAGE5 / "feature_separability.csv", index=False)
    correlated = fv.correlated_pairs(sample)
    correlated.to_csv(STAGE5 / "feature_correlations.csv", index=False)
    duplicates = fv.exact_duplicates(sample)
    feature_examples(sample).to_csv(STAGE5 / "feature_examples.csv", index=False)

    return {
        "build": build_info,
        "labels": labels,
        "analysis_sample_pairs": len(sample),
        "analysis_sample_positive_rate": round(float(sample["label"].mean()), 5),
        "features": len(fv.feature_columns(sample)),
        "constant_features": stats.loc[stats["constant"], "feature"].tolist(),
        "features_with_inf": stats.loc[stats["inf_count"] > 0, "feature"].tolist(),
        "exact_duplicate_features": duplicates,
        "highly_correlated_pairs_ge_0.95": int(len(correlated)),
        "top_separating_features": separability.head(15)[["feature", "auc"]].to_dict("records"),
        "weakest_features": separability.tail(10)[["feature", "auc"]].to_dict("records"),
    }


def write_report(profile: dict):
    lines = ["STAGE 5 PAIR FEATURE REPORT", "=" * 78]
    for name, info in profile["sets"].items():
        b, l = info["build"], info["labels"]
        lines += [f"{name}: {b['pairs']:,} pairs for {b['s1_entities']:,} S1 in {b['seconds']}s  "
                  f"({l['labelled_positives']:,} positive, {l['negatives']:,} negative)",
                  f"  positives == retrieved true pairs: {l['positives_match_ground_truth']}   "
                  f"duplicate pairs: {l['duplicate_pairs']}   "
                  f"S1 with >=1 positive: {l['s1_entities_with_a_positive']:,}"]
    a = profile.get("analysis")
    if a:
        lines += ["", f"Quality gate on {a['analysis_sample_pairs']:,} train pairs "
                      f"(positive rate {a['analysis_sample_positive_rate']}):",
                  f"  features: {a['features']}",
                  f"  constant: {a['constant_features']}",
                  f"  with inf: {a['features_with_inf']}",
                  f"  exact duplicates: {a['exact_duplicate_features']}",
                  f"  feature pairs with |pearson| >= 0.95: {a['highly_correlated_pairs_ge_0.95']}",
                  "", "Most separating single features (AUC):"]
        lines += [f"  {r['auc']:.4f}  {r['feature']}" for r in a["top_separating_features"]]
        lines += ["", "Least separating:"] + [f"  {r['auc']:.4f}  {r['feature']}" for r in a["weakest_features"]]
    (STAGE5 / "stage5_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", default="train_sample,validation")
    parser.add_argument("--skip-build", action="store_true", help="reuse existing pair files")
    args = parser.parse_args()
    STAGE5.mkdir(parents=True, exist_ok=True)

    profile_path = STAGE5 / "feature_profile.json"
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else {"sets": {}}

    for query_set in args.sets.split(","):
        candidates_path = CANDIDATES / f"{query_set}_candidates.parquet"
        pairs_path = FEATURES / f"{query_set}_pairs.parquet"
        ground_truth = ground_truth_for(query_set, candidates_path)
        print(f"\n{query_set}:")
        if args.skip_build and pairs_path.exists():
            build_info = profile.get("sets", {}).get(query_set, {}).get("build") or {
                "pairs": pq.ParquetFile(pairs_path).metadata.num_rows,
                "positives": int(pq.read_table(pairs_path, columns=["label"])["label"].to_numpy().sum()),
                "seconds": None,
                "s1_entities": len(ground_truth),
                "output": str(pairs_path),
            }
        else:
            build_info = build_pair_features(
                candidates_path,
                [NORMALIZED / "train_source1.parquet"],
                [NORMALIZED / "train_source2.parquet", NORMALIZED / "train_source3.parquet"],
                QUERY_SETS[query_set]["translations"],
                ground_truth, pairs_path, CHANNELS,
            )
        labels = label_check(pairs_path, ground_truth)
        print(f"  labels: {labels}")
        profile["sets"][query_set] = {"build": build_info, "labels": labels}

        if query_set == "train_sample":
            profile["analysis"] = analyse(pairs_path, build_info, labels)
            log_path = STAGE5 / "feature_experiments.csv"
            row = pd.DataFrame([{
                "experiment_id": f"FEAT-{len(pd.read_csv(log_path)) + 1 if log_path.exists() else 1:03d}",
                "date": date.today().isoformat(),
                "feature_group": "all (name, address, numeric, country, cross-field, relative, blocking)",
                "num_features": profile["analysis"]["features"],
                "train_pairs": build_info["pairs"],
                "positive_pairs": labels["labelled_positives"],
                "negative_pairs": labels["negatives"],
                "notes": "Stage 5 v1; train_sample candidates, analysis on 2M-pair sample",
            }])
            row.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, default=str)
    write_report(profile)


if __name__ == "__main__":
    main()
