from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = ROOT / "challenge" / "dataset" / "train"

OUTPUT_DIR = ROOT / "experiments" / "stage1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHED_PER_COUNTRY = 15
SINGLETONS_PER_COUNTRY = 5
SEED = 7


def load_tsv(path):
    return pd.read_csv(path, sep="\t", engine="pyarrow")


def main():

    gt = load_tsv(TRAIN_DIR / "train_ground_truth.tsv")
    s1 = load_tsv(TRAIN_DIR / "train_source1.tsv")

    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("").astype(str).str.strip()
    gt = gt.merge(
        s1[["entity_id", "country"]],
        left_on="source1_entity_id", right_on="entity_id", how="left",
    )

    picks = []
    for _, group in gt.groupby("country"):
        matched = group[group["matched_entity_ids"].ne("")]
        singletons = group[group["matched_entity_ids"].eq("")]
        picks.append(matched.sample(MATCHED_PER_COUNTRY, random_state=SEED))
        picks.append(singletons.sample(SINGLETONS_PER_COUNTRY, random_state=SEED))
    picks = pd.concat(picks, ignore_index=True)

    wanted = set(
        i for ids in picks["matched_entity_ids"] if ids for i in ids.split(",")
    )

    s1_records = s1.set_index("entity_id")
    del s1

    target_records = []
    for filename in ["train_source2.tsv", "train_source3.tsv"]:
        df = load_tsv(TRAIN_DIR / filename)
        target_records.append(df[df["entity_id"].isin(wanted)])
        del df
    targets = pd.concat(target_records).set_index("entity_id")

    def fmt(entity_id, record):
        return f"  {entity_id:<14} | {record['business_name']!s:<45} | {record['business_address']!s}"

    lines = []
    for _, row in picks.iterrows():
        s1_id = row["source1_entity_id"]
        kind = "MATCHED" if row["matched_entity_ids"] else "SINGLETON"
        lines.append("")
        lines.append(f"--- {kind} ({row['country']}) " + "-" * 70)
        lines.append(fmt(s1_id, s1_records.loc[s1_id]))
        for match_id in sorted(filter(None, row["matched_entity_ids"].split(","))):
            lines.append(fmt(match_id, targets.loc[match_id]))

    text = "\n".join(lines)
    print(text)

    output_file = OUTPUT_DIR / "matched_examples.txt"
    output_file.write_text(text, encoding="utf-8")
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
