from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = ROOT / "challenge" / "dataset" / "train"

OUTPUT_DIR = ROOT / "experiments" / "stage1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 15


def repeated_values(series, column):
    # Missing values are dropped so NaN addresses don't show up as one giant "duplicate".
    values = series.dropna().astype(str).str.strip()
    counts = values[values.ne("")].value_counts()
    repeated = counts[counts > 1].rename_axis(column).reset_index(name="count")
    return repeated


def analyze_file(path):

    df = pd.read_csv(
        path, sep="\t", engine="pyarrow",
        usecols=["business_name", "business_address"],
    )

    print("\n" + "=" * 80)
    print(path.name)
    print("=" * 80)

    for column, label in [
        ("business_name", "names"),
        ("business_address", "addresses"),
    ]:
        repeated = repeated_values(df[column], column)

        print(f"\nDistinct {label} appearing more than once: {len(repeated):,}")
        print(f"Rows covered by repeated {label}: {int(repeated['count'].sum()):,}")
        print(f"Top {TOP_N}:")
        print(repeated.head(TOP_N).to_string(index=False))

        repeated.to_csv(
            OUTPUT_DIR / f"{path.stem}_duplicate_{label}.csv",
            index=False,
        )


def main():

    for filename in [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
    ]:
        analyze_file(TRAIN_DIR / filename)


if __name__ == "__main__":
    main()
