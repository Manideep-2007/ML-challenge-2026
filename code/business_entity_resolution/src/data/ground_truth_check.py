from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

GT_PATH = (
    ROOT
    / "challenge"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

CHUNK_SIZE = 250_000


def main():

    print("=" * 70)
    print("GROUND TRUTH CHECK")
    print("=" * 70)

    sample = pd.read_csv(GT_PATH, sep="\t", nrows=10)

    print("\nColumns:")
    print(sample.columns.tolist())

    print("\nSample:")
    print(sample.to_string())

    total_rows = 0
    empty_count = 0
    match_count_total = 0
    max_matches = 0

    for chunk in pd.read_csv(GT_PATH, sep="\t", chunksize=CHUNK_SIZE):

        total_rows += len(chunk)

        matched = chunk["matched_entity_ids"].fillna("").astype(str).str.strip()
        is_empty = matched.eq("")
        empty_count += int(is_empty.sum())

        match_lists = matched[~is_empty].str.split(",")
        match_count_total += int(match_lists.str.len().sum())
        if len(match_lists) > 0:
            max_matches = max(max_matches, int(match_lists.str.len().max()))

    print(f"\nRows: {total_rows:,}")
    print(f"Entities with no matches (singletons): {empty_count:,}")
    print(f"Entities with matches: {total_rows - empty_count:,}")
    print(f"Total matched ID references: {match_count_total:,}")
    print(f"Max matches for a single entity: {max_matches:,}")
    if total_rows - empty_count > 0:
        print(
            f"Average matches per non-singleton entity: "
            f"{match_count_total / (total_rows - empty_count):.2f}"
        )

    print("\nGROUND TRUTH CHECK COMPLETE")


if __name__ == "__main__":
    main()
