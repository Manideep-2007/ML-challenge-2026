from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = ROOT / "challenge" / "dataset" / "train"
TEST_DIR = ROOT / "challenge" / "dataset" / "test"

OUTPUT_DIR = ROOT / "experiments" / "stage1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Share of addresses matching each documented noise pattern.
PATTERNS = {
    "five_digit_code": r"\b\d{5}(?:-\d{4})?\b",
    "six_digit_pin": r"\b\d{3}\s?\d{3}\b(?!\s*/)",
    "street_abbrev": r"(?i)\b(?:rd|st|ave|blvd|dr|ln|ct|hwy|pkwy|marg)\b\.?",
    "street_full": r"(?i)\b(?:road|street|avenue|boulevard|drive|lane|court|highway|parkway)\b",
    "landmark": r"(?i)\b(?:near|nr|opp|opposite|behind|beside|next to|adjacent)\b",
    "house_number_label": r"(?i)\b(?:h\.?\s?no|house no|plot no|flat no|door no|shop no|d\.?\s?no)\b",
    "unit_or_floor": r"(?i)\b(?:unit|suite|ste|apt|floor|flr|bldg)\b",
    "null_component": r"(?i)(?:^|,)\s*(?:null|n/a)\s*(?:,|$)",
    "hash_sign": r"#",
    "ends_with_state_code": r",\s*[A-Z]{2}\s*$",
    "french_street_word": r"(?i)\b(?:rue|chemin|allée|allee|impasse|quai|cedex|bd)\b",
    "non_ascii": r"[^\x00-\x7F]",
}


def analyze(path):

    df = pd.read_csv(
        path, sep="\t", engine="pyarrow",
        usecols=["business_address", "country"],
    )
    addresses = df["business_address"].dropna().astype(str).str.strip()
    country = df.loc[addresses.index, "country"]

    features = pd.DataFrame({
        "country": country,
        "length": addresses.str.len(),
        "word_count": addresses.str.split().str.len(),
        "comma_components": addresses.str.count(",") + 1,
        "digit_count": addresses.str.count(r"\d"),
        "special_char_count": addresses.str.count(r"[^A-Za-z0-9\s]"),
        "all_upper": addresses.str.contains(r"[A-Za-z]") & addresses.eq(addresses.str.upper()),
    })
    for label, pattern in PATTERNS.items():
        features[label] = addresses.str.contains(pattern, regex=True)

    rows = []
    for group_country, group in [("ALL", features), *features.groupby("country")]:
        row = {"file": path.name, "country": group_country, "rows": len(group)}
        for column in ["length", "word_count", "comma_components", "digit_count", "special_char_count"]:
            row[f"{column}_mean"] = round(float(group[column].mean()), 2)
            row[f"{column}_p50"] = float(group[column].median())
            row[f"{column}_p95"] = float(group[column].quantile(0.95))
        for column in ["all_upper", *PATTERNS]:
            row[f"pct_{column}"] = round(100 * float(group[column].mean()), 2)
        rows.append(row)

    del df, features
    return rows


def main():

    rows = []
    for directory, prefix in [(TRAIN_DIR, "train"), (TEST_DIR, "test")]:
        for source in ["source1", "source2", "source3"]:
            rows.extend(analyze(directory / f"{prefix}_{source}.tsv"))

    result = pd.DataFrame(rows)

    print("\nADDRESS STATISTICS")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(result.set_index(["file", "country"]).T.to_string())

    result.to_csv(OUTPUT_DIR / "address_statistics.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'address_statistics.csv'}")


if __name__ == "__main__":
    main()
