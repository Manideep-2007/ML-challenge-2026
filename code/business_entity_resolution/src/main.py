"""
End-to-end inference entry point.

    cd code/business_entity_resolution
    python -m src.main --split test         # final submission files in output/
    python -m src.main --split validation   # reproduce + score the validation split

Uses the model and decision rules selected in Stages 6-7:
experiments/stage7/decision_params.json names the model directory.
"""

from pathlib import Path
import argparse
import gc
import json
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
sys.path.insert(0, str(SRC))

from blocking.indexes import open_records  # noqa: E402
from blocking.run_stage4 import CONFIG as BLOCKING_CONFIG, learn_translators  # noqa: E402
from evaluation.evaluator import evaluate_predictions, load_ground_truth  # noqa: E402
from decision.decision_engine import DecisionParams  # noqa: E402
from inference.pipeline import Pipeline, decide, part_files  # noqa: E402
from features.feature_builder import reference_document_frequencies  # noqa: E402

NORMALIZED = ROOT / "artifacts" / "normalized"
FINAL = ROOT / "artifacts" / "final"
OUTPUT = ROOT / "output"
# Shipped with the code folder (src/export_artifacts.py); experiment folders are the
# development fallback.
MODEL_ARTIFACTS = SRC.parent / "model_artifacts"
STAGE6 = MODEL_ARTIFACTS if (MODEL_ARTIFACTS / "decision_params.json").exists() else ROOT / "experiments" / "stage6"
STAGE7 = MODEL_ARTIFACTS if (MODEL_ARTIFACTS / "decision_params.json").exists() else ROOT / "experiments" / "stage7"
VALIDATION_GT = ROOT / "experiments" / "stage2" / "validation_ground_truth.tsv"


def ensure_normalized(split: str):
    missing = [s for s in ("source1", "source2", "source3") if not (NORMALIZED / f"{split}_{s}.parquet").exists()]
    if missing:
        from normalization.run_stage3 import normalize_all
        print(f"Normalizing {split} files (Stage 3)...")
        normalize_all(force=False)


def write_tsv(path: Path, rows: dict[str, list[str]], column: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{column}\n")
        for s1_id, ids in rows.items():
            f.write(f"{s1_id}\t{','.join(dict.fromkeys(ids))}\n")


def s1_aligned_frames(files, row_groups_per_read: int = 8):
    """(source1_entity_id, candidate_entity_id) frames that never split one S1's rows
    (a legacy single file is read a few row groups at a time)."""
    import pyarrow.parquet as pq
    carry = None
    for path in files:
        pf = pq.ParquetFile(path)
        for g in range(0, pf.num_row_groups, row_groups_per_read):
            groups = list(range(g, min(g + row_groups_per_read, pf.num_row_groups)))
            frame = pf.read_row_groups(groups, columns=["source1_entity_id", "candidate_entity_id"]).to_pandas()
            if carry is not None:
                frame = pd.concat([carry, frame], ignore_index=True)
            last = frame["source1_entity_id"].iat[-1]
            tail = frame["source1_entity_id"].to_numpy() == last
            carry, frame = frame[tail], frame[~tail]
            if len(frame):
                yield frame
    if carry is not None and len(carry):
        yield carry


def write_candidate_tsv(path: Path, out_dirs, s1_ids: np.ndarray):
    """Stream candidate_pairs.tsv chunk by chunk (every S1 row, empty list when no candidates)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for d in out_dirs:
            for frame in s1_aligned_frames(part_files(d, "candidates")):
                frame = frame.drop_duplicates()
                grouped = frame.groupby("source1_entity_id", sort=False)["candidate_entity_id"].agg(",".join)
                f.writelines(f"{k}\t{v}\n" for k, v in grouped.items() if k not in seen)
                seen.update(grouped.index)
        f.writelines(f"{k}\t\n" for k in s1_ids if k not in seen)


def check_candidate_tsv(path: Path, s1_ids: set, matching_path: Path) -> dict:
    """Validator rules for candidate_pairs.tsv, streamed: header, one row per S1, no duplicate IDs,
    only S2-/S3- IDs, and every final match contained in its S1's candidate list."""
    matches = {}
    with open(matching_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            k, v = line.rstrip("\n").split("\t")
            if v:
                matches[k] = set(v.split(","))
    rows, issues, seen = 0, [], set()
    with open(path, encoding="utf-8") as f:
        if next(f).rstrip("\n") != "source1_entity_id\tcandidate_entity_ids":
            issues.append("bad header")
        for line in f:
            k, v = line.rstrip("\n").split("\t")
            rows += 1
            ids = v.split(",") if v else []
            if k in seen or k not in s1_ids:
                issues.append(f"row {k}: duplicate or unknown S1")
            seen.add(k)
            if len(ids) != len(set(ids)) or any(not i.startswith(("S2-", "S3-")) for i in ids):
                issues.append(f"row {k}: duplicate or non S2/S3 candidate")
            if k in matches and not matches[k] <= set(ids):
                issues.append(f"row {k}: match outside candidates")
            if len(issues) > 20:
                break
    if seen != s1_ids:
        issues.append(f"missing S1 rows: {len(s1_ids - seen)}")
    return {"rows": rows, "issues": issues[:20], "ok": not issues}


def reproduction_check(work: Path, s1_ids: set, model_id: str) -> dict:
    """Pipeline output vs the staged Stage 4 candidates and Stage 6 validation predictions."""
    staged_c = pd.read_parquet(ROOT / "artifacts" / "candidates" / "validation_candidates.parquet",
                               columns=["source1_entity_id", "candidate_entity_id"])
    staged_c = staged_c[staged_c["source1_entity_id"].isin(s1_ids)]
    piped_c = pd.concat([pd.read_parquet(f, columns=["source1_entity_id", "candidate_entity_id"])
                         for f in part_files(work, "candidates")], ignore_index=True)
    staged_keys = set(zip(staged_c["source1_entity_id"], staged_c["candidate_entity_id"]))
    piped_keys = set(zip(piped_c["source1_entity_id"], piped_c["candidate_entity_id"]))

    staged_p = pd.read_parquet(ROOT / "artifacts" / "predictions" / f"{model_id}_validation.parquet",
                               columns=["source1_entity_id", "candidate_entity_id", "probability"])
    staged_p = staged_p[staged_p["source1_entity_id"].isin(s1_ids)]
    piped_p = pd.concat([pd.read_parquet(f) for f in part_files(work, "predictions")], ignore_index=True)
    joined = staged_p.merge(piped_p, on=["source1_entity_id", "candidate_entity_id"], suffixes=("_staged", "_pipeline"))
    diff = (joined["probability_staged"] - joined["probability_pipeline"]).abs()
    return {
        "staged_candidate_pairs": len(staged_keys), "pipeline_candidate_pairs": len(piped_keys),
        "candidate_sets_identical": staged_keys == piped_keys,
        "pairs_only_in_staged": len(staged_keys - piped_keys), "pairs_only_in_pipeline": len(piped_keys - staged_keys),
        "compared_probabilities": int(len(joined)),
        "max_abs_probability_diff": float(diff.max()) if len(diff) else None,
        "pairs_with_diff_gt_1e-4": int((diff > 1e-4).sum()),
    }


def keep_awake():
    """Windows: keep the system awake while this process runs (reverts when it exits)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    keep_awake()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--chunk-size", type=int, default=20_000,
                        help="S1 per chunk (~2.8M candidate pairs at 20k; larger chunks need more RAM)")
    parser.add_argument("--sample", type=int, default=0,
                        help="validation only: run on N random validation S1 and compare with the staged artifacts")
    parser.add_argument("--country", default=None,
                        help="test only: process just this country and exit (run each country in its own "
                             "process to bound memory; a final run without --country writes the outputs)")
    parser.add_argument("--all-countries-in-subprocesses", action="store_true",
                        help="test only: process every country in a fresh subprocess, then write the outputs")
    args = parser.parse_args()
    started = time.time()

    decision = json.loads((STAGE7 / "decision_params.json").read_text())
    model_dir = STAGE6 / decision.pop("model")
    params = DecisionParams(**{**decision, "country_t": {k: tuple(v) for k, v in decision["country_t"].items()}})

    if args.split == "test":
        ensure_normalized("test")
        work = FINAL / "test"
        shipped = MODEL_ARTIFACTS / "translations"
        translations = shipped if (shipped / "token_translation_address.csv").exists() else FINAL / "translations"
        if not (translations / "token_translation_address.csv").exists():
            print("Learning token translations from all training labels...")
            translations.mkdir(parents=True, exist_ok=True)
            train_reference = open_records([NORMALIZED / "train_source2.parquet", NORMALIZED / "train_source3.parquet"])
            learn_translators(BLOCKING_CONFIG, set(), train_reference, translations, include_validation=True)
            del train_reference
            gc.collect()
        s1_paths = [NORMALIZED / "test_source1.parquet"]
        ref_paths = [NORMALIZED / "test_source2.parquet", NORMALIZED / "test_source3.parquet"]
        s1_ids = None
    else:
        work = FINAL / "validation"
        translations = ROOT / "experiments" / "stage4"   # learned without validation labels
        s1_paths = [NORMALIZED / "train_source1.parquet"]
        ref_paths = [NORMALIZED / "train_source2.parquet", NORMALIZED / "train_source3.parquet"]
        s1_ids = set(load_ground_truth(VALIDATION_GT))
        if args.sample:
            rng = np.random.default_rng(0)
            s1_ids = set(rng.choice(sorted(s1_ids), size=args.sample, replace=False))
            work = FINAL / "validation_sample"

    # Test runs one country at a time (exact: no true link crosses countries, every channel is
    # country-scoped) so 10M references fit in memory; IDF still comes from the whole universe.
    countries, global_df = [None], None
    if args.split == "test":
        countries = sorted(pd.read_parquet(s1_paths[0], columns=["country_key"])["country_key"].unique())
        work.mkdir(parents=True, exist_ok=True)
        df_path = work / "reference_document_frequencies.joblib"
        if not df_path.exists():
            print("Document frequencies over the full test reference universe...")
            joblib.dump(reference_document_frequencies(ref_paths, translations), df_path)
            gc.collect()
        if args.all_countries_in_subprocesses:
            for country_name in countries:
                if not (work / country_name / "done.json").exists():
                    subprocess.run([sys.executable, "-u", "-m", "src.main", "--split", "test", "--country", country_name,
                                    "--chunk-size", str(args.chunk_size)], cwd=SRC.parent, check=True)
        if args.country:
            countries = [args.country]
        if any(not (work / c / "done.json").exists() for c in countries):
            global_df = joblib.load(df_path)

    out_dirs, s1_parts, country_parts, runs = [], [], [], []
    for country_name in countries:
        out = work if country_name is None else work / country_name
        done = out / "done.json"
        if done.exists():
            print(f"{country_name}: already processed, reusing {out}")
            info = json.loads(done.read_text())
            s1_frame = pd.read_parquet(out / "s1_ids.parquet")
        else:
            print(f"Country: {country_name or 'all'}")
            pipeline = Pipeline(s1_paths, ref_paths, translations, model_dir, BLOCKING_CONFIG, s1_ids=s1_ids,
                                country=country_name, global_df=global_df)
            info = pipeline.run(out, chunk_size=args.chunk_size)
            s1_frame = pipeline.s1[["entity_id", "country_key"]].copy()
            del pipeline
            gc.collect()
            s1_frame.to_parquet(out / "s1_ids.parquet")
            done.write_text(json.dumps(info), encoding="utf-8")
        runs.append({"country": country_name, **info})
        out_dirs.append(out)
        s1_parts.append(s1_frame["entity_id"].to_numpy())
        country_parts.append(s1_frame["country_key"].to_numpy())

    if args.split == "test" and args.country:
        print(f"{args.country}: done ({runs[-1]})")
        return

    s1_ids_ordered = np.concatenate(s1_parts)
    country = pd.Series(np.concatenate(country_parts), index=s1_ids_ordered)
    run = {"s1": int(sum(r["s1"] for r in runs)), "pairs": int(sum(r["pairs"] for r in runs)),
           "seconds": round(sum(r["seconds"] for r in runs), 1), "per_country": runs}

    matches = decide(out_dirs, s1_ids_ordered, params, country)

    summary = {"split": args.split, "model": model_dir.name, "decision": decision, **run,
               "predicted_matches": int(sum(len(v) for v in matches.values())),
               "s1_with_matches": int(sum(bool(v) for v in matches.values()))}

    if args.split == "test":
        write_tsv(OUTPUT / "matching_results.tsv", matches, "matched_entity_ids")
        write_candidate_tsv(OUTPUT / "candidate_pairs.tsv", out_dirs, s1_ids_ordered)
        # The official validator holds every candidate in Python sets (~200M test pairs does
        # not fit in 16 GB), so the leaderboard file is validated with --check-ids and the
        # candidate file is checked by check_candidate_tsv (same rules, streamed).
        result = subprocess.run(
            [sys.executable, str(ROOT / "challenge" / "utils" / "validate_submission.py"),
             "--matching", str(OUTPUT / "matching_results.tsv"),
             "--candidate", str(OUTPUT / "__none__.tsv"),
             "--test-dir", str(ROOT / "challenge" / "dataset" / "test"), "--check-ids"],
            capture_output=True, text=True)
        summary["candidate_file_check"] = check_candidate_tsv(OUTPUT / "candidate_pairs.tsv", set(s1_ids_ordered),
                                                              OUTPUT / "matching_results.tsv")
        print(result.stdout, result.stderr)
        summary["validator_exit_code"] = result.returncode
    else:
        write_tsv(work / "matching_results.tsv", matches, "matched_entity_ids")
        gt_all = load_ground_truth(VALIDATION_GT)
        gt = {k: gt_all[k] for k in s1_ids_ordered}
        result = evaluate_predictions(matches, gt)
        summary["validation_macro_f05"] = round(result["macro_f_beta"], 6)
        summary["validation_singleton_f05"] = round(result["by_truth_type"]["singleton"]["macro_f_beta"], 6)
        summary["prediction_issues"] = result["issues"]
        if args.sample:
            summary["reproduction_check"] = reproduction_check(work, set(s1_ids_ordered), model_dir.name)

    summary["total_seconds"] = round(time.time() - started, 1)
    (work / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
