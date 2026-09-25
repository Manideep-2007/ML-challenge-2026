from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[4]

STAGE1_DIR = ROOT / "experiments" / "stage1"

SOURCE_DATASETS = [
    "train_source1", "train_source2", "train_source3",
    "test_source1", "test_source2", "test_source3",
]


def load_json(filename):
    with open(STAGE1_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    data_profile = load_json("data_profile.json")
    gt = load_json("ground_truth_profile.json")

    out = []
    add = out.append

    add("BUSINESS ENTITY RESOLUTION")
    add("STAGE 1 DATA AUDIT REPORT")
    add("=" * 70)

    # --------------------------------------------------------
    add("")
    add("DATASET SIZES")
    add("-" * 70)
    for name, profile in data_profile.items():
        add(f"{name:<20} {profile['rows']:>12,} rows")

    # --------------------------------------------------------
    add("")
    add("GROUND TRUTH")
    add("-" * 70)
    add(f"Source 1 entities:              {gt['total_source1_entities']:,}")
    add(f"Entities with matches:          {gt['entities_with_matches']:,}")
    add(f"Singleton entities:             {gt['singleton_entities']:,} ({gt['singleton_pct']}%)")
    add(f"Total positive links:           {gt['total_positive_links']:,}")
    add(f"Average matches/S1 (all):       {gt['average_matches_per_source1']:.4f}")
    add(f"Average matches/S1 (matched):   {gt['average_matches_per_matched_source1']:.4f}")
    add(f"Maximum matches/S1:             {gt['maximum_matches_for_one_source1']} "
        f"({gt['entities_with_maximum_matches_count']} entities)")

    # --------------------------------------------------------
    add("")
    add("SOURCE DISTRIBUTION")
    add("-" * 70)
    add(f"S2 links:                       {gt['source2_links']:,}")
    add(f"S3 links:                       {gt['source3_links']:,}")
    add(f"Matched S1 with S2 and S3:      {gt['matched_source1_with_both_s2_and_s3']:,}")
    add(f"Matched S1 with only S2:        {gt['matched_source1_with_only_s2']:,}")
    add(f"Matched S1 with only S3:        {gt['matched_source1_with_only_s3']:,}")
    for label, cov in gt["target_source_coverage"].items():
        add(f"{label.rstrip('-')} records never matched:     "
            f"{cov['records_never_matched']:,} of {cov['records_in_file']:,} "
            f"({cov['pct_records_never_matched']}%)")
        add(f"{label.rstrip('-')} records linked to >1 S1:   {cov['ids_linked_to_multiple_source1']:,}")

    # --------------------------------------------------------
    add("")
    add("VALIDATION CHECKS")
    add("-" * 70)
    add(f"Invalid match prefixes:         {gt['invalid_match_id_prefixes_count']}")
    add(f"GT IDs missing from S1:         {gt['ground_truth_entities_missing_from_source1_count']}")
    add(f"S1 IDs missing from GT:         {gt['source1_entities_missing_from_ground_truth_count']}")
    add(f"Duplicate S1 rows in GT:        {gt['duplicate_source1_rows_in_ground_truth']}")
    add(f"Duplicate IDs within a list:    {gt['duplicate_ids_within_a_match_list']}")
    for label, cov in gt["target_source_coverage"].items():
        add(f"GT {label.rstrip('-')} IDs missing from file:  {cov['referenced_ids_missing_from_file']}")
    cc = gt["country_consistency"]
    add(f"Same-country links:             {cc['links_same_country']:,}")
    add(f"Cross-country links:            {cc['links_cross_country']:,}")

    # --------------------------------------------------------
    add("")
    add("MATCH COUNT DISTRIBUTION")
    add("-" * 70)
    total = gt["total_source1_entities"]
    for count, frequency in gt["match_count_distribution"].items():
        add(f"{count:>2} matches: {frequency:>10,} S1 entities ({100 * frequency / total:5.2f}%)")

    add("")
    add("BY SOURCE 1 COUNTRY (train)")
    add("-" * 70)
    for country, stats in gt["by_source1_country"].items():
        add(f"{country:<8} {stats['source1_entities']:>10,} entities, "
            f"{stats['singleton_pct']}% singletons, "
            f"{stats['average_matches']} avg matches")

    # --------------------------------------------------------
    add("")
    add("MISSING VALUES (NaN / blank)")
    add("-" * 70)
    add(f"{'dataset':<16}{'name':>10}{'address':>12}{'country':>10}")
    for name in SOURCE_DATASETS:
        p = data_profile[name]
        m, e = p["missing_values"], p["empty_strings"]
        add(f"{name:<16}"
            f"{m['business_name'] + e['business_name']:>10,}"
            f"{m['business_address'] + e['business_address']:>12,}"
            f"{m['country'] + e['country']:>10,}")

    # --------------------------------------------------------
    add("")
    add("DUPLICATES (rows repeating an earlier value; missing excluded)")
    add("-" * 70)
    add(f"{'dataset':<16}{'names':>12}{'addresses':>12}{'name+addr':>12}")
    for name in SOURCE_DATASETS:
        d = data_profile[name]["duplicates"]
        add(f"{name:<16}"
            f"{d['duplicate_business_names']:>12,}"
            f"{d['duplicate_business_addresses']:>12,}"
            f"{d['duplicate_name_address_pairs']:>12,}")

    # --------------------------------------------------------
    add("")
    add("COUNTRIES")
    add("-" * 70)
    train_countries, test_countries = set(), set()
    for name in SOURCE_DATASETS:
        dist = data_profile[name]["country_distribution"]
        (train_countries if name.startswith("train") else test_countries).update(dist)
        add(f"{name:<16} " + ", ".join(f"{c}={n:,}" for c, n in dist.items()))
    add(f"Countries in train:  {sorted(train_countries)}")
    add(f"Countries in test:   {sorted(test_countries)}")
    add(f"Test-only countries: {sorted(test_countries - train_countries)}")

    # --------------------------------------------------------
    add("")
    add("TEXT LENGTHS (characters: median / mean / p95; words: median)")
    add("-" * 70)
    for name in SOURCE_DATASETS:
        ts = data_profile[name]["text_statistics"]
        n, a = ts["business_name"], ts["business_address"]
        add(f"{name:<16} name {n['median_length']:>4.0f} / {n['mean_length']:>5.1f} / "
            f"{n['p95_length']:>4.0f}, {n['median_words']:.0f} words   "
            f"address {a['median_length']:>4.0f} / {a['mean_length']:>5.1f} / "
            f"{a['p95_length']:>4.0f}, {a['median_words']:.0f} words")

    add("")
    add("See observed_patterns.md for qualitative noise patterns and the decision table.")

    text = "\n".join(out)
    output_path = STAGE1_DIR / "stage1_report.txt"
    output_path.write_text(text, encoding="utf-8")

    print(text)
    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
