from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = ROOT / "challenge" / "dataset" / "train"
TEST_DIR = ROOT / "challenge" / "dataset" / "test"

CHUNK_SIZE = 250_000


def inspect(path):

    print("\n" + "=" * 60)
    print(path.name)
    print("=" * 60)

    na_counts = None
    empty_counts = None
    total_rows = 0

    for chunk in pd.read_csv(path, sep="\t", chunksize=CHUNK_SIZE):

        total_rows += len(chunk)

        chunk_na = chunk.isna().sum()
        chunk_empty = pd.Series(
            {
                col: chunk[col].astype(str).str.strip().eq("").sum()
                for col in chunk.columns
            }
        )

        if na_counts is None:
            na_counts = chunk_na
            empty_counts = chunk_empty
        else:
            na_counts = na_counts.add(chunk_na, fill_value=0)
            empty_counts = empty_counts.add(chunk_empty, fill_value=0)

    print(f"Rows: {total_rows:,}")
    print("\nMissing (NaN) values:")
    print(na_counts.astype(int))

    print("\nEmpty strings:")
    print(empty_counts.astype(int))


def main():

    for path in [
        TRAIN_DIR / "train_source1.tsv",
        TRAIN_DIR / "train_source2.tsv",
        TRAIN_DIR / "train_source3.tsv",
        TEST_DIR / "test_source1.tsv",
        TEST_DIR / "test_source2.tsv",
        TEST_DIR / "test_source3.tsv",
    ]:
        inspect(path)


if __name__ == "__main__":
    main()
