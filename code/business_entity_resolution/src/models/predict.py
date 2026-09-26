"""
Streams a pair file through a model. Returns compact arrays (S1 code, label,
probability) for evaluation and optionally writes per-pair predictions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def predict_pairs(path: Path, features: list[str], predict_fn, s1_index: pd.Index,
                  out_path: Path | None = None, batch_rows: int = 2_000_000, subset: bool = False) -> dict:
    """
    s1_index: evaluated S1 ids (codes are positions in this index).
    subset=True: pairs of S1 outside s1_index are skipped (model comparison on an
    S1 subset); otherwise they are an error.
    predict_fn: X (DataFrame) -> probability array.
    """
    columns = ["source1_entity_id", "candidate_entity_id", "label"] + features
    codes, labels, probs = [], [], []
    writer = None
    try:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=columns):
            frame = batch.to_pandas()
            code = s1_index.get_indexer(frame["source1_entity_id"])
            if (code < 0).any():
                if not subset:
                    raise ValueError("pairs contain S1 ids outside the evaluated set")
                frame, code = frame[code >= 0], code[code >= 0]
                if not len(frame):
                    continue
            prob = predict_fn(frame[features])
            codes.append(code.astype(np.int32))
            labels.append(frame["label"].to_numpy(np.int8))
            probs.append(prob)
            if out_path is not None:
                table = pa.table({
                    "source1_entity_id": frame["source1_entity_id"].to_numpy(),
                    "candidate_entity_id": frame["candidate_entity_id"].to_numpy(),
                    "probability": prob,
                    "label": frame["label"].to_numpy(np.int8),
                })
                if writer is None:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return {"code": np.concatenate(codes), "label": np.concatenate(labels), "prob": np.concatenate(probs)}
