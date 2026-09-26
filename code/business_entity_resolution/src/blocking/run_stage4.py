from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))

from evaluation.evaluator import load_ground_truth  # noqa: E402
from blocking.candidate_evaluation import (  # noqa: E402
    build_truth, evaluate_keys, found_mask, miss_reasons, size_distribution,
)
from blocking.candidate_generator import (  # noqa: E402
    generate_candidates, peak_memory_mb, union_at_k, union_ranks,
)
from blocking.char_blocks import char_channel  # noqa: E402
from blocking.exact_blocks import exact_view_channel  # noqa: E402
from blocking.hybrid_blocks import hybrid_channel  # noqa: E402
from blocking.indexes import open_records, source_of  # noqa: E402
from blocking.numeric_blocks import number_channel  # noqa: E402
from blocking.token_alignment import (  # noqa: E402
    document_frequency, learn_translation, sample_positions, translator,
)
from blocking.token_blocks import deleet, token_channel  # noqa: E402

NORMALIZED = ROOT / "artifacts" / "normalized"
CANDIDATES_DIR = ROOT / "artifacts" / "candidates"
STAGE4_DIR = ROOT / "experiments" / "stage4"
GT_PATH = ROOT / "challenge" / "dataset" / "train" / "train_ground_truth.tsv"
VALIDATION_GT = ROOT / "experiments" / "stage2" / "validation_ground_truth.tsv"

SEED = 42
HARD_MISS_SAMPLE = 1000
LARGEST_SETS = 50

# Blocking configuration. Chosen on the train-split tuning sample (see
# experiments/stage4/tuning/tuning_log.md), then reported on validation.
CONFIG = {
    "exact_max_block": 1000,
    "number_max_block": 500,
    "name_token_max_df": 5000,
    "address_token_max_df": 5000,
    "hybrid_max_df": 50000,
    "token_top_k": 100,
    "char_ngram_range": (3, 4),
    "char_max_df": 20000,
    "char_top_k": 50,
    "translation_pairs": 1_500_000,
    "translation_min_count": 5,
    "translation_min_probability": 0.5,
}


def build_channels(config: dict, translators: dict | None = None) -> dict:
    """Order matters for the cumulative experiments (A, B, C, ...)."""
    channels = {
        "exact_name": exact_view_channel("name_basic", config["exact_max_block"]),
        "compact_name": exact_view_channel("name_compact", config["exact_max_block"]),
        "content_name": exact_view_channel("name_content_compact", config["exact_max_block"]),
        "fingerprint_name": exact_view_channel("name_fingerprint", config["exact_max_block"]),
        "rare_name_token": token_channel(["name_content"], config["name_token_max_df"], config["token_top_k"]),
        "exact_address": exact_view_channel("address_basic", config["exact_max_block"]),
        "compact_address": exact_view_channel("address_compact", config["exact_max_block"]),
        "fingerprint_address": exact_view_channel("address_fingerprint", config["exact_max_block"]),
        "address_number": number_channel("address_numbers", config["number_max_block"]),
        "rare_address_token": token_channel(["address_basic"], config["address_token_max_df"], config["token_top_k"]),
        "hybrid_name_address": hybrid_channel("name_content", "address_basic", config["hybrid_max_df"], config["token_top_k"]),
        "char_name": char_channel("name_basic", config["char_ngram_range"], config["char_max_df"], config["char_top_k"]),
        # Stage 8 / E003: names of address-less references only (small pool, typo-robust)
        "char_name_no_address": char_channel("name_basic", (3, 4), config["char_max_df"], 20,
                                             only_empty_address=True, deleet_reference=True),
    }
    if translators is not None:
        channels["hybrid_translated"] = hybrid_channel(
            "name_content", "address_basic", config["hybrid_max_df"], config["token_top_k"],
            transforms={"name_content": deleet},
            ref_transforms={"name_content": translators["name"], "address_basic": translators["address"]},
        )
    return channels


def learn_translators(config: dict, eval_ids: set[str], reference, out_dir: Path,
                      include_validation: bool = False) -> dict:
    """
    Learn S2/S3 -> S1 token translations from training pairs, excluding eval_ids.
    `reference` must be the TRAIN S2+S3 records. include_validation=True (final
    test inference only) also learns from the validation split's labels.
    """
    full = load_ground_truth(GT_PATH)
    validation = set() if include_validation else set(load_ground_truth(VALIDATION_GT))
    learn_gt = {k: v for k, v in full.items() if k not in validation and k not in eval_ids and v}
    learners = open_records([NORMALIZED / "train_source1.parquet"], ids=set(learn_gt))
    truth = build_truth(learn_gt, learners.ids, reference.ids)
    pick = sample_positions(len(truth.keys), config["translation_pairs"], SEED)
    ref_sample = sample_positions(len(reference), 2_000_000, SEED)

    translators = {}
    for field, column in [("name", "name_content"), ("address", "address_basic")]:
        s1_all = learners.columns([column])[column]
        s1_pairs = s1_all.iloc[truth.query_idx[pick]].tolist()
        target_pairs = reference.rows([column], truth.ref_idx[pick])[column].tolist()
        if field == "name":
            target_pairs = [deleet(t) for t in target_pairs]
        target_df_texts = reference.rows([column], ref_sample)[column]
        dictionary = learn_translation(
            s1_pairs, target_pairs,
            document_frequency(s1_all), len(s1_all),
            document_frequency(target_df_texts), len(target_df_texts),
            min_count=config["translation_min_count"],
            min_probability=config["translation_min_probability"],
        )
        pd.DataFrame(
            [(t, s, p, n) for t, (s, p, n) in dictionary.items()],
            columns=["s2s3_token", "s1_token", "probability", "support"],
        ).sort_values("support", ascending=False).to_csv(out_dir / f"token_translation_{field}.csv", index=False)
        translators[field] = translator(dictionary)
        print(f"  learned {field} translations: {len(dictionary):,}")
    return translators


FINAL_CHANNELS = ["content_name", "fingerprint_name", "fingerprint_address", "hybrid_translated"]

CUMULATIVE = [
    ("A: exact name", ["exact_name"]),
    ("B: + compact/content name", ["exact_name", "compact_name", "content_name"]),
    ("C: + rare name token", ["exact_name", "compact_name", "content_name", "rare_name_token"]),
    ("D: + exact/compact address", ["exact_name", "compact_name", "content_name", "rare_name_token",
                                    "exact_address", "compact_address"]),
    ("E: + numeric + address token", ["exact_name", "compact_name", "content_name", "rare_name_token",
                                      "exact_address", "compact_address", "address_number", "rare_address_token"]),
    ("F: + hybrid name/address", ["exact_name", "compact_name", "content_name", "rare_name_token",
                                  "exact_address", "compact_address", "address_number", "rare_address_token",
                                  "hybrid_name_address"]),
    ("G: + char n-gram", None),  # all channels
    ("FINAL: content + fingerprints + translated hybrid", FINAL_CHANNELS),
]


# ============================================================
# QUERY SETS
# ============================================================

def query_set(name: str, size: int) -> dict[str, list[str]]:
    """
    validation: the Stage 2 validation split.
    tuning: train-split sample for choosing blocking parameters (seed 42).
    train_sample: train-split sample whose candidates train the Stage 6 model (seed 7).
    """
    if name == "validation":
        return load_ground_truth(VALIDATION_GT)
    full = load_ground_truth(GT_PATH)
    validation = set(load_ground_truth(VALIDATION_GT))
    train_ids = sorted(k for k in full if k not in validation)
    rng = np.random.default_rng(SEED if name == "tuning" else 7)
    sample = rng.choice(train_ids, size=min(size, len(train_ids)), replace=False)
    return {k: full[k] for k in sample}


# ============================================================
# INTEGRITY
# ============================================================

def integrity(query_ids, sources, q_idx, keys, n_channels) -> dict:
    return {
        "all_source1_ids_start_with_S1": bool(all(i.startswith("S1-") for i in query_ids[np.unique(q_idx)])),
        # Also rules out S1 -> S1 self-matches.
        "all_candidates_S2_or_S3": bool(np.isin(sources, ["S2", "S3"]).all()),
        "no_duplicate_pairs": bool(len(np.unique(keys)) == len(keys)),
        "every_pair_has_a_channel": bool((n_channels >= 1).all()),
    }


# ============================================================
# REPORT
# ============================================================

def write_report(out_dir: Path, profile: dict, channel_table: pd.DataFrame, recall_rows: list[dict]):
    u = profile["union"]
    lines = [
        "STAGE 4 CANDIDATE GENERATION REPORT",
        "=" * 78,
        f"Query set: {profile['query_set']}   queries: {profile['queries']:,}   "
        f"reference S2+S3: {profile['reference_records']:,}   true pairs: {profile['true_pairs']:,}",
        f"Channels: {', '.join(profile['channels'])}",
        f"Config: {profile['config']}",
        f"Runtime: {profile['total_seconds']}s   peak memory: {profile['process_peak_mb']} MB",
        "",
        "UNION",
        f"  pair recall            {u['pair_recall']:.4f}",
        f"  S1 with all found      {u['entity_all_found']:.4f}",
        f"  S1 with any found      {u['entity_any_found']:.4f}",
        f"  F0.5 ceiling           {u['f05_ceiling']:.4f}   (perfect matcher on these candidates)",
        f"  candidates per S1      min {u['cand_min']}  median {u['cand_median']:.0f}  mean {u['cand_mean']:.1f}  "
        f"p90 {u['cand_p90']:.0f}  p95 {u['cand_p95']:.0f}  p99 {u['cand_p99']:.0f}  max {u['cand_max']}",
        f"  S1 without candidates  {u['queries_without_candidates']:,}",
    ]
    for key in sorted(k for k in u if k.startswith("f05_ceiling_")):
        country = key.removeprefix("f05_ceiling_")
        lines.append(f"  {country:<8} pair recall {u[f'pair_recall_{country}']:.4f}  ceiling {u[key]:.4f}  "
                     f"mean candidates {u[f'cand_mean_{country}']:.1f}")

    lines += ["", "PER CHANNEL (standalone)",
              f"  {'channel':<22}{'recall':>8}{'ceiling':>9}{'mean':>8}{'p95':>7}{'max':>7}{'seconds':>9}"]
    for row in channel_table.itertuples():
        lines.append(f"  {row.channel:<22}{row.pair_recall:>8.4f}{row.f05_ceiling:>9.4f}"
                     f"{row.cand_mean:>8.1f}{row.cand_p95:>7.0f}{row.cand_max:>7}{row.seconds:>9}")

    lines += ["", "CUMULATIVE / LEAVE-ONE-OUT",
              f"  {'experiment':<40}{'recall':>8}{'ceiling':>9}{'mean':>8}{'p95':>7}"]
    for row in recall_rows:
        lines.append(f"  {row['experiment']:<40}{row['pair_recall']:>8.4f}{row['f05_ceiling']:>9.4f}"
                     f"{row['cand_mean']:>8.1f}{row['cand_p95']:>7.0f}")

    union_k = out_dir / "union_at_k.csv"
    if union_k.exists():
        lines += ["", "UNION WITH RANKED CHANNELS CUT AT TOP-K",
                  f"  {'K':>5}{'recall':>9}{'ceiling':>9}{'mean':>8}{'p95':>7}"]
        for row in pd.read_csv(union_k).itertuples():
            lines.append(f"  {row.k:>5}{row.pair_recall:>9.4f}{row.f05_ceiling:>9.4f}{row.cand_mean:>8.1f}{row.cand_p95:>7.0f}")

    m = profile["misses"]
    lines += ["", "MISSED TRUE PAIRS", f"  total {m['missed_true_pairs']:,}"]
    if "by_reason_pct" in m:
        lines += [f"  % {k:<26} {v}" for k, v in m["by_reason_pct"].items()]
        lines.append(f"  target name script: {m['target_name_script_pct']}")
        lines.append(f"  by country: {m['by_country']}")

    lines += ["", "INTEGRITY"] + [f"  {k}: {v}" for k, v in profile["integrity"].items()]
    (out_dir / "stage4_report.txt").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", choices=["tuning", "validation", "train_sample"], default="tuning")
    parser.add_argument("--size", type=int, default=50_000, help="sample size for tuning / train_sample")
    parser.add_argument("--channels", default="final", help="comma list, 'final' or 'all'")
    parser.add_argument("--set", action="append", default=[], help="override config, e.g. token_top_k=100")
    parser.add_argument("--write-candidates", action="store_true")
    parser.add_argument("--tag", default="latest", help="tuning output subfolder")
    args = parser.parse_args()

    config = dict(CONFIG)
    for item in args.set:
        key, value = item.split("=", 1)
        if isinstance(CONFIG[key], tuple):
            config[key] = tuple(int(v) for v in value.split(","))
        else:
            config[key] = type(CONFIG[key])(value)

    out_dir = {
        "validation": STAGE4_DIR,
        "train_sample": STAGE4_DIR / "train_sample",
        "tuning": STAGE4_DIR / "tuning" / args.tag,
    }[args.queries]
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    ground_truth = query_set(args.queries, args.size)
    queries = open_records([NORMALIZED / "train_source1.parquet"], ids=set(ground_truth))
    reference = open_records([NORMALIZED / "train_source2.parquet", NORMALIZED / "train_source3.parquet"])
    truth = build_truth(ground_truth, queries.ids, reference.ids)
    n_q, n_r = len(queries), len(reference)
    print(f"Queries ({args.queries}): {n_q:,}   reference S2+S3: {n_r:,}   true pairs: {len(truth.keys):,}")
    print(f"Config: {config}")

    wanted = {"all": None, "final": FINAL_CHANNELS}.get(args.channels, args.channels.split(","))
    translators = None
    if wanted is None or "hybrid_translated" in wanted:
        translators = learn_translators(config, set(ground_truth), reference, out_dir)
    all_channels = build_channels(config, translators)
    channels = all_channels if wanted is None else {k: all_channels[k] for k in wanted}

    union, channel_metrics, ranked_store = generate_candidates(
        channels, queries, reference, truth, keep_ranks=True)

    if ranked_store:
        pd.DataFrame(union_at_k(union, ranked_store, truth, n_q, queries.country)).to_csv(
            out_dir / "union_at_k.csv", index=False)

    # ---- per-channel, cumulative, leave-one-out -----------------------
    channel_table = pd.DataFrame([
        {k: v for k, v in m.items() if k not in ("channel_stats", "recall_at_k")} for m in channel_metrics.values()
    ])
    channel_table.to_csv(out_dir / "channel_recall.csv", index=False)
    k_rows = [{"channel": name, **row} for name, m in channel_metrics.items() for row in m.get("recall_at_k", [])]
    if k_rows:
        pd.DataFrame(k_rows).to_csv(out_dir / "recall_at_k.csv", index=False)

    rows = []
    for label, names in CUMULATIVE:
        names = [n for n in (names or list(channels)) if n in channels]
        if names:
            rows.append({"experiment": label, "channels": "|".join(names),
                         **evaluate_keys(union.subset(names), truth, n_q, n_r, queries.country)})
    for name in channels:
        others = [n for n in channels if n != name]
        if others:
            rows.append({"experiment": f"union without {name}", "channels": "|".join(others),
                         **evaluate_keys(union.subset(others), truth, n_q, n_r, queries.country)})
    union_metrics = evaluate_keys(union.keys, truth, n_q, n_r, queries.country)
    rows.append({"experiment": "UNION (all channels)", "channels": "|".join(channels), **union_metrics})
    pd.DataFrame(rows).to_csv(out_dir / "candidate_recall.csv", index=False)

    size_distribution(union.keys, n_q, n_r).to_csv(out_dir / "candidate_size_distribution.csv", index=False)

    # ---- candidate statistics per country + source --------------------
    q_idx, r_idx = union.query_ref()
    labels, n_channels = union.channel_labels()
    counts = np.bincount(q_idx, minlength=n_q)
    stats_rows = []
    for country in ["ALL"] + sorted(np.unique(queries.country)):
        mask_q = np.ones(n_q, bool) if country == "ALL" else queries.country == country
        c = counts[mask_q]
        stats_rows.append({"country": country, "queries": int(mask_q.sum()), "pairs": int(c.sum()),
                           "min": int(c.min()), "median": float(np.median(c)), "mean": round(float(c.mean()), 2),
                           "p90": float(np.percentile(c, 90)), "p95": float(np.percentile(c, 95)),
                           "p99": float(np.percentile(c, 99)), "max": int(c.max())})
    pd.DataFrame(stats_rows).to_csv(out_dir / "candidate_statistics.csv", index=False)

    # ---- largest candidate sets (explosion investigation) --------------
    largest = np.argsort(counts)[::-1][:LARGEST_SETS]
    q_view = queries.rows(["name_raw", "address_raw", "name_basic", "address_basic"], largest)
    dominant = []
    for q in largest:
        lo, hi = np.searchsorted(q_idx, q, "left"), np.searchsorted(q_idx, q, "right")
        dominant.append(pd.Series(labels[lo:hi]).str.split("|").explode().value_counts().head(3).to_dict())
    pd.DataFrame({
        "source1_entity_id": queries.ids[largest], "country": queries.country[largest],
        "candidates": counts[largest], "true_matches": truth.true_counts[largest],
        "name_raw": q_view["name_raw"], "address_raw": q_view["address_raw"],
        "pairs_by_channel": [json.dumps(d) for d in dominant],
    }).to_csv(out_dir / "largest_candidate_sets.csv", index=False)

    # ---- hard misses ----------------------------------------------------
    found = found_mask(truth, union.keys)
    missed = np.flatnonzero(~found)
    miss_summary = {"missed_true_pairs": int(len(missed))}
    if len(missed):
        mq, mr = truth.query_idx[missed], truth.ref_idx[missed]
        diag_cols = ["name_content", "address_basic", "address_numbers"]
        reasons = miss_reasons(
            np.arange(len(mq)), np.arange(len(mr)),
            queries.rows(diag_cols, mq),
            reference.rows(diag_cols + ["name_script"], mr),
        )
        reasons["country"] = queries.country[mq]
        miss_summary["by_reason_pct"] = {
            col: round(100 * float(reasons[col].mean()), 2)
            for col in ["target_address_missing", "shares_name_token", "shares_address_token", "shares_address_number"]
        }
        miss_summary["target_name_script_pct"] = (100 * reasons["target_name_script"].value_counts(normalize=True)).round(2).to_dict()
        miss_summary["by_country"] = reasons["country"].value_counts().to_dict()

        rng = np.random.default_rng(SEED)
        pick = np.sort(rng.choice(len(missed), size=min(HARD_MISS_SAMPLE, len(missed)), replace=False))
        raw_cols = ["name_raw", "address_raw", "name_basic", "address_basic"]
        s1 = queries.rows(raw_cols, mq[pick]).add_prefix("s1_")
        tgt = reference.rows(raw_cols, mr[pick]).add_prefix("true_")
        hard = pd.concat([
            pd.DataFrame({"source1_entity_id": queries.ids[mq[pick]], "true_entity_id": reference.ids[mr[pick]],
                          "country": queries.country[mq[pick]]}),
            s1, tgt, reasons.iloc[pick].reset_index(drop=True).drop(columns=["country"]),
        ], axis=1)
        hard.to_csv(out_dir / "hard_misses.csv", index=False)

    # ---- integrity + optional candidate file ---------------------------
    sources = source_of(reference.ids[r_idx])
    checks = integrity(queries.ids, sources, q_idx, union.keys, n_channels)
    if args.write_candidates:
        CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
        path = CANDIDATES_DIR / f"{args.queries}_candidates.parquet"
        columns = {
            "source1_entity_id": queries.ids[q_idx].astype(str),
            "candidate_entity_id": reference.ids[r_idx].astype(str),
            "candidate_source": sources.astype(str),
            "blocking_channels": labels.astype(str),
            "num_blocking_channels": n_channels,
        }
        for name, ranked in ranked_store.items():
            columns[f"{name}_rank"] = union_ranks(union, ranked)
        columns["candidates_for_source1"] = counts[q_idx].astype(np.int32)
        pq.write_table(pa.table(columns), path, compression="zstd")
        checks["candidate_file"] = str(path.relative_to(ROOT))

    profile = {
        "query_set": args.queries,
        "queries": n_q, "reference_records": n_r, "true_pairs": int(len(truth.keys)),
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in config.items()},
        "channels": list(channels),
        "channel_metrics": channel_metrics,
        "union": union_metrics,
        "misses": miss_summary,
        "integrity": checks,
        "total_seconds": round(time.time() - started, 1),
        "process_peak_mb": peak_memory_mb(),
    }
    with open(out_dir / "blocking_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False, default=str)
    write_report(out_dir, profile, channel_table, rows)

    print("\nUNION:", json.dumps(union_metrics))
    print("Misses:", json.dumps(miss_summary, ensure_ascii=False))
    print("Integrity:", checks)
    print(f"Total {profile['total_seconds']}s, peak memory {profile['process_peak_mb']} MB -> {out_dir}")


if __name__ == "__main__":
    main()
