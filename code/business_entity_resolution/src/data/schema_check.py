from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = ROOT / "challenge" / "dataset" / "train"
TEST_DIR = ROOT / "challenge" / "dataset" / "test"


EXPECTED_ENTITY_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

EXPECTED_GT_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]


def check_columns(path, expected):

    # nrows=0 reads only the header - these files are hundreds of MB,
    # a full load is unnecessary just to check column names.
    df = pd.read_csv(path, sep="\t", nrows=0)

    actual = list(df.columns)

    print(f"\n{path.name}")
    print("Expected:", expected)
    print("Actual:  ", actual)

    if actual != expected:
        raise ValueError(
            f"Schema mismatch in {path}"
        )


def main():

    for source in ["source1", "source2", "source3"]:

        check_columns(
            TRAIN_DIR / f"train_{source}.tsv",
            EXPECTED_ENTITY_COLUMNS,
        )

        check_columns(
            TEST_DIR / f"test_{source}.tsv",
            EXPECTED_ENTITY_COLUMNS,
        )

    check_columns(
        TRAIN_DIR / "train_ground_truth.tsv",
        EXPECTED_GT_COLUMNS,
    )

    print("\nSCHEMA CHECK PASSED")


if __name__ == "__main__":
    main()
