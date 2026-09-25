from pathlib import Path
from collections import Counter
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = ROOT / "challenge" / "dataset" / "train"
TEST_DIR = ROOT / "challenge" / "dataset" / "test"

CHUNK_SIZE = 250_000


def inspect(path):

    counts = Counter()

    for chunk in pd.read_csv(
        path, sep="\t", usecols=["country"], chunksize=CHUNK_SIZE
    ):
        counts.update(chunk["country"].fillna("<NaN>").tolist())

    print(f"\n{path.name}")
    for value, count in counts.most_common():
        print(f"  {value}: {count:,}")


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
