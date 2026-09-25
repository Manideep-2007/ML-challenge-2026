from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = ROOT / "challenge" / "dataset" / "train"
TEST_DIR = ROOT / "challenge" / "dataset" / "test"

OUTPUT_DIR = ROOT / "experiments" / "stage1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MISSING_LIKE = {
    "",
    "-",
    "--",
    ".",
    "na",
    "n/a",
    "n.a.",
    "none",
    "null",
    "nan",
    "nil",
    "unknown",
    "not available",
}

def inspect(path):

    df = pd.read_csv(path, sep="\t", engine="pyarrow")

    records = []

    for column in ["business_name", "business_address", "country"]:

        raw = df[column]
        values = raw.fillna("").astype(str).str.strip()
        lowered = values.str.lower()

        is_nan = raw.isna()
        is_blank = raw.notna() & values.eq("")
        is_placeholder = values.ne("") & lowered.isin(MISSING_LIKE)
        missing_like = is_nan | is_blank | is_placeholder

        # Placeholder tokens embedded as a comma-separated component, e.g. "12 M, NULL, BIRMINGHAM, AL".
        # Empty components (trailing/double commas) are punctuation noise, not placeholders.
        components = lowered[~missing_like].str.split(r"\s*,\s*", regex=True)
        embedded = components.apply(
            lambda parts: any(p and p in MISSING_LIKE for p in parts)
        )

        records.append({
            "file": path.name,
            "column": column,
            "rows": len(df),
            "nan": int(is_nan.sum()),
            "blank": int(is_blank.sum()),
            "placeholder_value": int(is_placeholder.sum()),
            "missing_like": int(missing_like.sum()),
            "percentage": round(float(missing_like.mean() * 100), 4),
            "embedded_placeholder_component": int(embedded.sum()),
        })

    del df
    return records


def main():

    all_records = []

    for directory, prefix in [(TRAIN_DIR, "train"), (TEST_DIR, "test")]:
        for source in ["source1", "source2", "source3"]:
            all_records.extend(inspect(directory / f"{prefix}_{source}.tsv"))

    report = pd.DataFrame(all_records)

    print(report.to_string(index=False))

    report.to_csv(OUTPUT_DIR / "missing_report.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'missing_report.csv'}")


if __name__ == "__main__":
    main()
