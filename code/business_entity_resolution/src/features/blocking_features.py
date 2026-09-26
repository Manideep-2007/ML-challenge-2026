"""
Blocking provenance: which retrieval routes found the candidate, its rank in
the ranked channel, and how crowded the S1 entity's candidate set is.
"""

import numpy as np
import pandas as pd


def blocking_features(candidates: pd.DataFrame, channel_names: list[str]) -> dict:
    labels = candidates["blocking_channels"].to_numpy(dtype=object)
    f = {f"blocked_{name}": np.array([name in l.split("|") for l in labels], dtype=np.float32)
         for name in channel_names}
    f["blocking_channel_count"] = candidates["num_blocking_channels"].to_numpy(dtype=np.float32)
    f["candidates_for_source1"] = candidates["candidates_for_source1"].to_numpy(dtype=np.float32)
    for column in [c for c in candidates.columns if c.endswith("_rank")]:
        rank = candidates[column].to_numpy(dtype=np.float32)
        f[column] = np.where(rank < 0, np.nan, rank).astype(np.float32)
    return f
