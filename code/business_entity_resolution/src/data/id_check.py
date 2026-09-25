from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = ROOT / "challenge" / "dataset" / "train"
TEST_DIR = ROOT / "challenge" / "dataset" / "test"

CHUNK_SIZE = 250_000


def check_source(path, prefix):

    seen_ids = set()
    total_rows = 0
    invalid_examples = []
    invalid_count = 0
    duplicate_count = 0

    for chunk in pd.read_csv(
        path, sep="\t", usecols=["entity_id"], chunksize=CHUNK_SIZE
    ):
        ids = chunk["entity_id"].astype(str)
        total_rows += len(ids)

        invalid_mask = ~ids.str.startswith(prefix)
        if invalid_mask.any():
            invalid_count += int(invalid_mask.sum())
            if len(invalid_examples) < 5:
                invalid_examples.extend(
                    ids[invalid_mask].head(5 - len(invalid_examples)).tolist()
                )

        for entity_id in ids:
            if entity_id in seen_ids:
                duplicate_count += 1
            else:
                seen_ids.add(entity_id)

    print(f"\n{path.name}")
    print(f"Rows: {total_rows:,}")
    print(f"Unique IDs: {len(seen_ids):,}")
    print(f"Invalid prefix IDs: {invalid_count:,}")
    print(f"Duplicate IDs: {duplicate_count:,}")

    if invalid_count > 0:
        print("Example invalid IDs:", invalid_examples)
        raise ValueError("Invalid source IDs detected")

    if duplicate_count > 0:
        raise ValueError("Duplicate entity IDs detected")


def main():

    check_source(TRAIN_DIR / "train_source1.tsv", "S1-")
    check_source(TRAIN_DIR / "train_source2.tsv", "S2-")
    check_source(TRAIN_DIR / "train_source3.tsv", "S3-")

    check_source(TEST_DIR / "test_source1.tsv", "S1-")
    check_source(TEST_DIR / "test_source2.tsv", "S2-")
    check_source(TEST_DIR / "test_source3.tsv", "S3-")

    print("\nENTITY ID CHECK PASSED")


if __name__ == "__main__":
    main()
