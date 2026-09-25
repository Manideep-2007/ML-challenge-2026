from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = ROOT / "challenge" / "dataset" / "train"
TEST_DIR = ROOT / "challenge" / "dataset" / "test"

OUTPUT_DIR = ROOT / "experiments" / "stage1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILES
# ============================================================

FILES = {
    "train_source1": TRAIN_DIR / "train_source1.tsv",
    "train_source2": TRAIN_DIR / "train_source2.tsv",
    "train_source3": TRAIN_DIR / "train_source3.tsv",
    "train_ground_truth": TRAIN_DIR / "train_ground_truth.tsv",

    "test_source1": TEST_DIR / "test_source1.tsv",
    "test_source2": TEST_DIR / "test_source2.tsv",
    "test_source3": TEST_DIR / "test_source3.tsv",
}

TEXT_COLUMNS = ["business_name", "business_address", "country"]


# ============================================================
# LOAD
# ============================================================

def load_tsv(path):
    # One file at a time: the full dataset (~2.5 GB) does not fit in free RAM.
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, sep="\t", engine="pyarrow")


def stripped(series):
    return series.fillna("").astype(str).str.strip()


# ============================================================
# BASIC PROFILE
# ============================================================

def profile_dataframe(df, name):

    return {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "missing_values": {
            column: int(df[column].isna().sum())
            for column in df.columns
        },
        # Whitespace-only values that survived parsing as non-null.
        "empty_strings": {
            column: int((df[column].notna() & stripped(df[column]).eq("")).sum())
            for column in df.columns
        },
    }


# ============================================================
# TEXT STATISTICS
# ============================================================

def text_statistics(df):

    result = {}

    for column in TEXT_COLUMNS:

        present = df[column].dropna().astype(str).str.strip()
        lengths = present.str.len()
        words = present.str.split().str.len()

        result[column] = {
            "non_missing": int(len(present)),
            "min_length": int(lengths.min()),
            "p05_length": float(lengths.quantile(0.05)),
            "median_length": float(lengths.median()),
            "mean_length": round(float(lengths.mean()), 2),
            "p95_length": float(lengths.quantile(0.95)),
            "max_length": int(lengths.max()),
            "median_words": float(words.median()),
            "mean_words": round(float(words.mean()), 2),
        }

    return result


# ============================================================
# COUNTRY DISTRIBUTION
# ============================================================

def country_distribution(df):

    counts = df["country"].fillna("<MISSING>").astype(str).value_counts()

    return {str(country): int(count) for country, count in counts.items()}


# ============================================================
# DUPLICATES
# ============================================================

def duplicate_statistics(df):
    # "duplicate rows" = rows whose value already appeared earlier (count - unique).
    # Missing values are excluded so ~170k NaN addresses don't count as duplicates.

    name = stripped(df["business_name"])
    address = stripped(df["business_address"])

    has_name = name.ne("")
    has_address = address.ne("")
    has_both = has_name & has_address

    pair = name[has_both] + "|||" + address[has_both]

    return {
        "duplicate_entity_ids": int(df["entity_id"].duplicated().sum()),
        "duplicate_business_names": int(name[has_name].duplicated().sum()),
        "duplicate_business_names_case_insensitive": int(
            name[has_name].str.lower().duplicated().sum()
        ),
        "duplicate_business_addresses": int(address[has_address].duplicated().sum()),
        "duplicate_name_address_pairs": int(pair.duplicated().sum()),
    }


# ============================================================
# UNIQUE COUNTS
# ============================================================

def unique_statistics(df):

    return {
        f"{column}_unique": int(df[column].nunique(dropna=True))
        for column in ["entity_id", "business_name", "business_address", "country"]
    }


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 80)
    print("STAGE 1 — COMPLETE DATA AUDIT")
    print("=" * 80)

    report = {}
    duplicate_rows = []

    for name, path in FILES.items():

        print("\n" + "-" * 80)
        print(f"{name}  ({path.name})")
        print("-" * 80)

        df = load_tsv(path)

        print("Rows:", f"{len(df):,}")
        print("Columns:", len(df.columns))

        if name == "train_ground_truth":
            report[name] = {
                "dataset": name,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "column_names": list(df.columns),
                "note": "See ground_truth_profile.json for full analysis.",
            }
            del df
            continue

        dataset_report = profile_dataframe(df, name)
        dataset_report["text_statistics"] = text_statistics(df)
        dataset_report["country_distribution"] = country_distribution(df)
        dataset_report["duplicates"] = duplicate_statistics(df)
        dataset_report["unique_statistics"] = unique_statistics(df)

        report[name] = dataset_report

        duplicate_rows.append({
            "dataset": name,
            "rows": dataset_report["rows"],
            **dataset_report["duplicates"],
            **dataset_report["unique_statistics"],
        })

        print(json.dumps(dataset_report, indent=2, ensure_ascii=False))

        del df

    output_file = OUTPUT_DIR / "data_profile.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    duplicate_file = OUTPUT_DIR / "duplicate_report.csv"
    pd.DataFrame(duplicate_rows).to_csv(duplicate_file, index=False)

    print("\n" + "=" * 80)
    print("DATA AUDIT COMPLETE")
    print("=" * 80)
    print(f"\nSaved:\n{output_file}\n{duplicate_file}")


if __name__ == "__main__":
    main()
