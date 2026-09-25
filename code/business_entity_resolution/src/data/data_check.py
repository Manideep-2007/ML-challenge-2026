from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = ROOT / "challenge" / "dataset" / "train"
TEST_DIR = ROOT / "challenge" / "dataset" / "test"

CHUNK_SIZE = 250_000


def count_rows_and_get_columns(path):
    columns = None
    total_rows = 0

    for chunk in pd.read_csv(path, sep="\t", chunksize=CHUNK_SIZE):
        if columns is None:
            columns = list(chunk.columns)
        total_rows += len(chunk)

    return total_rows, columns


def main():

    print("=" * 70)
    print("STAGE 0 - DATA CHECK")
    print("=" * 70)

    files = {
        "train_source1": TRAIN_DIR / "train_source1.tsv",
        "train_source2": TRAIN_DIR / "train_source2.tsv",
        "train_source3": TRAIN_DIR / "train_source3.tsv",
        "ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
        "test_source1": TEST_DIR / "test_source1.tsv",
        "test_source2": TEST_DIR / "test_source2.tsv",
        "test_source3": TEST_DIR / "test_source3.tsv",
    }

    for name, path in files.items():

        print(f"\nChecking: {name}")
        print(f"Path: {path}")

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:,.1f} MB")

        rows, columns = count_rows_and_get_columns(path)

        print(f"Rows: {rows:,}")
        print(f"Columns: {columns}")

    print("\n" + "=" * 70)
    print("DATA CHECK PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
