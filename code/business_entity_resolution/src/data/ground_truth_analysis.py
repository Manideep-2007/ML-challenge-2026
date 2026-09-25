from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = ROOT / "challenge" / "dataset" / "train"

GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"
S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"

OUTPUT_DIR = ROOT / "experiments" / "stage1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LISTED = 20


def load_tsv(path, columns=None):
    return pd.read_csv(path, sep="\t", engine="pyarrow", usecols=columns)


def explode_pairs(gt):
    # One row per (source1_entity_id, matched_id) positive link.
    ids = gt["matched_entity_ids"].fillna("").astype(str).str.strip()
    pairs = pd.DataFrame({
        "source1_entity_id": gt["source1_entity_id"],
        "matched_id": ids.str.split(","),
    }).explode("matched_id")
    pairs["matched_id"] = pairs["matched_id"].astype(str).str.strip()
    return pairs[pairs["matched_id"].ne("")].reset_index(drop=True)


def value_counts_dict(series):
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def main():

    print("=" * 80)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 80)

    gt = load_tsv(GT_PATH)
    s1 = load_tsv(S1_PATH, ["entity_id", "country"])

    pairs = explode_pairs(gt)
    pairs["target_source"] = pairs["matched_id"].str[:3]

    # --------------------------------------------------------
    # Per-S1 match counts
    # --------------------------------------------------------

    per_s1 = (
        pairs.groupby(["source1_entity_id", "target_source"])
        .size()
        .unstack(fill_value=0)
    )

    counts = pd.DataFrame({"source1_entity_id": gt["source1_entity_id"]})
    counts = counts.merge(per_s1, left_on="source1_entity_id", right_index=True, how="left")
    for column in ["S2-", "S3-"]:
        if column not in counts:
            counts[column] = 0
    counts[["S2-", "S3-"]] = counts[["S2-", "S3-"]].fillna(0).astype(int)
    counts["match_count"] = counts["S2-"] + counts["S3-"]

    total_s1 = len(gt)
    singleton_count = int((counts["match_count"] == 0).sum())
    matched_count = total_s1 - singleton_count
    total_positive_links = int(counts["match_count"].sum())
    max_matches = int(counts["match_count"].max())

    max_entities = counts.loc[
        counts["match_count"] == max_matches, "source1_entity_id"
    ].tolist()

    # --------------------------------------------------------
    # Source 2 / Source 3 split
    # --------------------------------------------------------

    s2_links = int((pairs["target_source"] == "S2-").sum())
    s3_links = int((pairs["target_source"] == "S3-").sum())
    invalid_ids = pairs.loc[
        ~pairs["target_source"].isin(["S2-", "S3-"]), "matched_id"
    ].tolist()

    non_singletons = counts[counts["match_count"] > 0]
    s2_and_s3 = int(((non_singletons["S2-"] > 0) & (non_singletons["S3-"] > 0)).sum())
    only_s2 = int(((non_singletons["S2-"] > 0) & (non_singletons["S3-"] == 0)).sum())
    only_s3 = int(((non_singletons["S2-"] == 0) & (non_singletons["S3-"] > 0)).sum())

    # --------------------------------------------------------
    # Integrity: GT vs Source 1
    # --------------------------------------------------------

    gt_s1_ids = set(gt["source1_entity_id"])
    actual_s1_ids = set(s1["entity_id"])

    gt_missing_from_s1 = sorted(gt_s1_ids - actual_s1_ids)
    s1_missing_from_gt = sorted(actual_s1_ids - gt_s1_ids)
    duplicate_gt_rows = int(gt["source1_entity_id"].duplicated().sum())
    duplicate_ids_within_list = int(
        pairs.duplicated(["source1_entity_id", "matched_id"]).sum()
    )

    del gt_s1_ids, actual_s1_ids

    # --------------------------------------------------------
    # Integrity: matched IDs vs Source 2 / Source 3 files,
    # coverage of S2/S3, and country consistency of links
    # --------------------------------------------------------

    s1_country = s1.set_index("entity_id")["country"]
    pairs["s1_country"] = pairs["source1_entity_id"].map(s1_country)
    counts["country"] = counts["source1_entity_id"].map(s1_country)
    del s1, s1_country

    coverage = {}
    target_country_frames = []

    for label, path in [("S2-", S2_PATH), ("S3-", S3_PATH)]:
        src = load_tsv(path, ["entity_id", "country"])
        linked = pairs.loc[pairs["target_source"] == label, "matched_id"]
        linked_unique = linked.drop_duplicates()
        exists = linked_unique.isin(src["entity_id"])

        coverage[label] = {
            "records_in_file": int(len(src)),
            "distinct_ids_referenced_by_gt": int(len(linked_unique)),
            "referenced_ids_missing_from_file": int((~exists).sum()),
            "records_never_matched": int(len(src) - exists.sum()),
            "pct_records_never_matched": round(
                100 * (len(src) - exists.sum()) / len(src), 2
            ),
            # >1 means one S2/S3 record was linked to several S1 entities.
            "ids_linked_to_multiple_source1": int(linked.duplicated().sum()),
        }

        target_country_frames.append(
            src.rename(columns={"entity_id": "matched_id", "country": "target_country"})
        )
        del src

    target_countries = pd.concat(target_country_frames, ignore_index=True)
    del target_country_frames
    pairs = pairs.merge(target_countries, on="matched_id", how="left")
    del target_countries

    same_country = pairs["s1_country"] == pairs["target_country"]
    cross_country = pairs.loc[~same_country & pairs["target_country"].notna()]

    country_consistency = {
        "links_same_country": int(same_country.sum()),
        "links_cross_country": int(len(cross_country)),
        "cross_country_examples": cross_country.head(MAX_LISTED)[
            ["source1_entity_id", "s1_country", "matched_id", "target_country"]
        ].to_dict("records"),
    }

    del pairs

    by_country = {}
    for country, group in counts.groupby("country"):
        by_country[str(country)] = {
            "source1_entities": int(len(group)),
            "singletons": int((group["match_count"] == 0).sum()),
            "singleton_pct": round(100 * (group["match_count"] == 0).mean(), 2),
            "average_matches": round(float(group["match_count"].mean()), 3),
        }

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "total_source1_entities": total_s1,
        "entities_with_matches": matched_count,
        "singleton_entities": singleton_count,
        "singleton_pct": round(100 * singleton_count / total_s1, 2),
        "total_positive_links": total_positive_links,
        "average_matches_per_source1": total_positive_links / total_s1,
        "average_matches_per_matched_source1": total_positive_links / matched_count,
        "maximum_matches_for_one_source1": max_matches,
        "entities_with_maximum_matches_count": len(max_entities),
        "entities_with_maximum_matches": max_entities[:MAX_LISTED],

        "source2_links": s2_links,
        "source3_links": s3_links,
        "matched_source1_with_both_s2_and_s3": s2_and_s3,
        "matched_source1_with_only_s2": only_s2,
        "matched_source1_with_only_s3": only_s3,

        "invalid_match_id_prefixes": invalid_ids[:MAX_LISTED],
        "invalid_match_id_prefixes_count": len(invalid_ids),
        "ground_truth_entities_missing_from_source1": gt_missing_from_s1[:MAX_LISTED],
        "ground_truth_entities_missing_from_source1_count": len(gt_missing_from_s1),
        "source1_entities_missing_from_ground_truth": s1_missing_from_gt[:MAX_LISTED],
        "source1_entities_missing_from_ground_truth_count": len(s1_missing_from_gt),
        "duplicate_source1_rows_in_ground_truth": duplicate_gt_rows,
        "duplicate_ids_within_a_match_list": duplicate_ids_within_list,

        "match_count_distribution": value_counts_dict(counts["match_count"]),
        "s2_match_count_distribution": value_counts_dict(counts["S2-"]),
        "s3_match_count_distribution": value_counts_dict(counts["S3-"]),

        "target_source_coverage": coverage,
        "country_consistency": country_consistency,
        "by_source1_country": by_country,
    }

    print(json.dumps(report, indent=2))

    output_file = OUTPUT_DIR / "ground_truth_profile.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved:")
    print(output_file)


if __name__ == "__main__":
    main()
