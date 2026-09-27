"""
Feature set v3 = v2 + street features (Stage 8 / E004), added to existing v2
pair files without rebuilding the other 117 columns:

    python src/features/add_street_features.py --sets train_sample,validation_subset

Reads artifacts/features/<set>_pairs_v2.parquet, writes <set>_pairs_v3.parquet
(same rows, same order, street columns appended).
"""

from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from features.street_features import add_street_views, street_features  # noqa: E402

ROOT = SRC.parents[2]
FEATURES = ROOT / "artifacts" / "features"
NORMALIZED = ROOT / "artifacts" / "normalized"
STREET_COLUMNS = ["street_core", "street_number", "legal_dotted"]


def record_views(paths: list[Path]) -> pd.DataFrame:
    frame = pd.concat([pq.read_table(p, columns=["entity_id", "country_key", "address_nfkc", "name_nfkc"]).to_pandas()
                       for p in paths], ignore_index=True)
    add_street_views(frame)
    return frame[["entity_id"] + STREET_COLUMNS].set_index("entity_id")


def extend(pairs_in: Path, pairs_out: Path, s1: pd.DataFrame, ref: pd.DataFrame):
    pf = pq.ParquetFile(pairs_in)
    tmp = pairs_out.with_suffix(".tmp")
    writer = None
    start = time.time()
    for g in range(pf.num_row_groups):
        table = pf.read_row_group(g)
        ids = table.select(["source1_entity_id", "candidate_entity_id"]).to_pandas()
        s = s1.reindex(ids["source1_entity_id"])
        c = ref.reindex(ids["candidate_entity_id"])
        if s["street_core"].isna().any() or c["street_core"].isna().any():
            raise ValueError("pairs reference records without street views")
        f = street_features({k: s[k].to_numpy() for k in STREET_COLUMNS}, {k: c[k].to_numpy() for k in STREET_COLUMNS})
        for name, values in f.items():
            table = table.append_column(name, pa.array(values, type=pa.float32()))
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
        writer.write_table(table)
        print(f"  {pairs_out.name}: row group {g + 1}/{pf.num_row_groups} ({time.time() - start:.0f}s)", flush=True)
    writer.close()
    tmp.replace(pairs_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", default="train_sample,validation_subset")
    args = parser.parse_args()
    print("Street views: train S1 / S2 / S3")
    s1 = record_views([NORMALIZED / "train_source1.parquet"])
    ref = record_views([NORMALIZED / "train_source2.parquet", NORMALIZED / "train_source3.parquet"])
    for name in args.sets.split(","):
        extend(FEATURES / f"{name}_pairs_v2.parquet", FEATURES / f"{name}_pairs_v3.parquet", s1, ref)


if __name__ == "__main__":
    main()
