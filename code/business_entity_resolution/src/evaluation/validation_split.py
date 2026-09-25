from pathlib import Path
import hashlib
import json

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = ROOT / "challenge" / "dataset" / "train"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"
S1_PATH = TRAIN_DIR / "train_source1.tsv"

OUT_DIR = ROOT / "experiments" / "stage2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_FRACTION = 0.20
RANDOM_STATE = 42
MATCH_BIN_CAP = 5


def parse_match_ids(value):
    value = str(value).strip()
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def distribution(series):
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def split_profile(df):
    lists = df["match_list"]
    return {
        "entities": int(len(df)),
        "singletons": int((df["match_count"] == 0).sum()),
        "singleton_pct": round(100 * float((df["match_count"] == 0).mean()), 3),
        "single_match": int((df["match_count"] == 1).sum()),
        "multi_match": int((df["match_count"] > 1).sum()),
        "positive_links": int(df["match_count"].sum()),
        "s2_links": int(lists.map(lambda ids: sum(i.startswith("S2-") for i in ids)).sum()),
        "s3_links": int(lists.map(lambda ids: sum(i.startswith("S3-") for i in ids)).sum()),
        "average_matches": round(float(df["match_count"].mean()), 4),
        "maximum_matches": int(df["match_count"].max()),
        "country_distribution": distribution(df["country"]),
        "match_count_distribution": distribution(df["match_count"]),
    }


def main():

    gt = pd.read_csv(GT_PATH, sep="\t", dtype=str, keep_default_na=False)
    s1 = pd.read_csv(S1_PATH, sep="\t", engine="pyarrow", usecols=["entity_id", "country"])

    gt["country"] = gt["source1_entity_id"].map(s1.set_index("entity_id")["country"])
    del s1

    if gt["country"].isna().any():
        raise ValueError("Ground-truth entities missing from train_source1.tsv")

    gt["match_list"] = gt["matched_entity_ids"].map(parse_match_ids)
    gt["match_count"] = gt["match_list"].map(len)

    # Strata: country x match count (capped), so validation mirrors train
    # both overall and within each country.
    gt["stratum"] = (
        gt["country"] + "|" + gt["match_count"].clip(upper=MATCH_BIN_CAP).astype(str)
    )

    train_ids, valid_ids = train_test_split(
        gt["source1_entity_id"].tolist(),
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
        stratify=gt["stratum"],
    )

    valid_set = set(valid_ids)
    gt["split"] = gt["source1_entity_id"].map(
        lambda x: "validation" if x in valid_set else "train"
    )

    train_df = gt[gt["split"] == "train"]
    valid_df = gt[gt["split"] == "validation"]

    if set(train_df["source1_entity_id"]) & valid_set:
        raise AssertionError("Train and validation overlap")

    validation_path = OUT_DIR / "validation_ground_truth.tsv"
    valid_df[["source1_entity_id", "matched_entity_ids"]].to_csv(
        validation_path, sep="\t", index=False
    )

    fingerprint = hashlib.sha256(
        "\n".join(sorted(valid_ids)).encode("utf-8")
    ).hexdigest()

    profile = {
        "total_entities": int(len(gt)),
        "train_entities": int(len(train_df)),
        "validation_entities": int(len(valid_df)),
        "validation_fraction": round(len(valid_df) / len(gt), 6),
        "random_state": RANDOM_STATE,
        "split_strategy": (
            "Source 1 entity split stratified by country x ground-truth "
            f"match count (capped at {MATCH_BIN_CAP}). Source 2 and Source 3 "
            "are NOT split: both remain the full shared reference universe."
        ),
        "train_split_definition": (
            "train = all train_ground_truth.tsv rows whose source1_entity_id "
            "is not in validation_ground_truth.tsv"
        ),
        "validation_ids_sha256": fingerprint,
        "train": split_profile(train_df),
        "validation": split_profile(valid_df),
    }

    profile_path = OUT_DIR / "validation_profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print("\nValidation split created.")
    print(f"Total:      {len(gt):,}")
    print(f"Train:      {len(train_df):,}")
    print(f"Validation: {len(valid_df):,}")
    print(f"Validation IDs sha256: {fingerprint}")
    print(f"\nSaved:\n{validation_path}\n{profile_path}")


if __name__ == "__main__":
    main()
