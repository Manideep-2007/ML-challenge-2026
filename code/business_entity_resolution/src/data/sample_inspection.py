from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = ROOT / "challenge" / "dataset" / "train"
TEST_DIR = ROOT / "challenge" / "dataset" / "test"

OUTPUT_DIR = ROOT / "experiments" / "stage1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = ["entity_id", "business_name", "business_address", "country"]
HEAD_ROWS = 20
RANDOM_PER_COUNTRY = 10
SEED = 42


def section(title, df):
    lines = ["", "=" * 100, title, "=" * 100]
    with pd.option_context("display.max_colwidth", 90, "display.width", 250):
        lines.append(df[COLUMNS].to_string(index=False))
    return lines


def main():

    lines = []

    for directory, prefix in [(TRAIN_DIR, "train"), (TEST_DIR, "test")]:
        for source in ["source1", "source2", "source3"]:
            path = directory / f"{prefix}_{source}.tsv"
            df = pd.read_csv(path, sep="\t", engine="pyarrow")

            if prefix == "train":
                lines += section(f"{path.name} — first {HEAD_ROWS} rows", df.head(HEAD_ROWS))

            for country, group in df.groupby("country"):
                sample = group.sample(min(RANDOM_PER_COUNTRY, len(group)), random_state=SEED)
                lines += section(
                    f"{path.name} — {RANDOM_PER_COUNTRY} random rows, country={country}",
                    sample,
                )
            del df

    text = "\n".join(lines)
    print(text)

    output_file = OUTPUT_DIR / "sample_records.txt"
    output_file.write_text(text, encoding="utf-8")
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
