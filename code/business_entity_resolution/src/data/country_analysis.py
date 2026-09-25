from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = ROOT / "challenge" / "dataset" / "train"
TEST_DIR = ROOT / "challenge" / "dataset" / "test"

OUTPUT_DIR = ROOT / "experiments" / "stage1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_distribution(path):

    raw = pd.read_csv(path, sep="\t", engine="pyarrow", usecols=["country"])["country"]

    counts = (
        raw.fillna("<MISSING>")
        .astype(str)
        .value_counts()
        .rename_axis("country_raw")
        .reset_index(name="count")
    )

    # Same label differing only by case/whitespace would collapse here.
    counts["country_normalized"] = counts["country_raw"].str.strip().str.lower()
    counts["split"] = "train" if path.parent == TRAIN_DIR else "test"
    counts["file"] = path.name
    counts["pct_of_file"] = (100 * counts["count"] / counts["count"].sum()).round(2)

    return counts


def main():

    frames = []

    for directory in [TRAIN_DIR, TEST_DIR]:
        for path in sorted(directory.glob("*.tsv")):
            if "ground_truth" in path.name:
                continue
            frames.append(get_distribution(path))

    result = pd.concat(frames, ignore_index=True)[
        ["split", "file", "country_raw", "country_normalized", "count", "pct_of_file"]
    ]

    print(result.to_string(index=False))

    variants = result.groupby("country_normalized")["country_raw"].nunique()
    train_countries = set(result.loc[result["split"] == "train", "country_normalized"])
    test_countries = set(result.loc[result["split"] == "test", "country_normalized"])

    print("\nRaw spellings per normalized country:")
    print(variants.to_string())
    print("\nCountries in train:", sorted(train_countries))
    print("Countries in test: ", sorted(test_countries))
    print("Test-only countries:", sorted(test_countries - train_countries))
    print("Missing country values:", int(result.loc[result["country_raw"] == "<MISSING>", "count"].sum()))

    result.to_csv(OUTPUT_DIR / "country_report.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'country_report.csv'}")


if __name__ == "__main__":
    main()
