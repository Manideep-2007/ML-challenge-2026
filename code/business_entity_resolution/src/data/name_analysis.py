from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = ROOT / "challenge" / "dataset" / "train"
TEST_DIR = ROOT / "challenge" / "dataset" / "test"

OUTPUT_DIR = ROOT / "experiments" / "stage1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Share of names matching each documented noise pattern (case-insensitive unless noted).
PATTERNS = {
    "legal_suffix": r"(?i)\b(?:inc|llc|ltd|limited|pvt|private|corp|corporation|co|company|llp|plc|sa|sas|sarl|eurl|gmbh)\b",
    "ampersand": r"&",
    "word_and": r"(?i)\band\b",
    "dba_aka": r"(?i)\b(?:dba|d/b/a|aka|a/k/a|t/a)\b",
    "domain": r"(?i)\.(?:com|net|org|in|co\.in|fr)\b",
    "parentheses": r"[()]",
    "trailing_comma": r",\s*$",
    "has_digit": r"\d",
    "non_ascii": r"[^\x00-\x7F]",
    "dotted_initials": r"\b(?:[A-Za-z]\.){2,}",
}


def analyze(path):

    df = pd.read_csv(
        path, sep="\t", engine="pyarrow",
        usecols=["business_name", "country"],
    )
    names = df["business_name"].dropna().astype(str).str.strip()
    country = df.loc[names.index, "country"]

    features = pd.DataFrame({
        "country": country,
        "length": names.str.len(),
        "word_count": names.str.split().str.len(),
        "digit_count": names.str.count(r"\d"),
        "special_char_count": names.str.count(r"[^A-Za-z0-9\s]"),
        "all_upper": names.str.contains(r"[A-Za-z]") & names.eq(names.str.upper()),
        "all_lower": names.str.contains(r"[A-Za-z]") & names.eq(names.str.lower()),
    })
    for label, pattern in PATTERNS.items():
        features[label] = names.str.contains(pattern, regex=True)

    rows = []
    for group_country, group in [("ALL", features), *features.groupby("country")]:
        row = {"file": path.name, "country": group_country, "rows": len(group)}
        for column in ["length", "word_count", "digit_count", "special_char_count"]:
            row[f"{column}_mean"] = round(float(group[column].mean()), 2)
            row[f"{column}_p50"] = float(group[column].median())
            row[f"{column}_p95"] = float(group[column].quantile(0.95))
        for column in ["all_upper", "all_lower", *PATTERNS]:
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

    print("\nNAME STATISTICS")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(result.set_index(["file", "country"]).T.to_string())

    result.to_csv(OUTPUT_DIR / "name_statistics.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'name_statistics.csv'}")


if __name__ == "__main__":
    main()
