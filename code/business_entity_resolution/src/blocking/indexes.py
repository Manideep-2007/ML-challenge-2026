"""
Reference universe (S2 + S3) and query (S1) records, addressed by integer
position (ref_idx / query_idx) rather than ID strings.

Only entity_id and country are held in memory; each channel loads just the
view columns it needs from the Stage 3 parquet files and frees them after,
so ~10M reference records fit in the available RAM.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass
class Records:
    paths: list[Path]
    row_filters: list[np.ndarray | None]   # per-file boolean mask (None = all rows)
    ids: np.ndarray
    country: np.ndarray

    def __len__(self):
        return len(self.ids)

    def columns(self, names: list[str]) -> pd.DataFrame:
        """View columns for all records, aligned with self.ids."""
        frames = []
        for path, row_filter in zip(self.paths, self.row_filters):
            frame = pq.read_table(path, columns=names).to_pandas()
            frames.append(frame if row_filter is None else frame[row_filter])
        return pd.concat(frames, ignore_index=True)

    def rows(self, names: list[str], positions: np.ndarray) -> pd.DataFrame:
        """View columns for selected record positions, in the given order."""
        positions = np.asarray(positions, dtype=np.int64)
        parts, offset = [], 0
        for path, row_filter in zip(self.paths, self.row_filters):
            n_rows = pq.ParquetFile(path).metadata.num_rows
            file_rows = np.arange(n_rows) if row_filter is None else np.flatnonzero(row_filter)
            in_file = (positions >= offset) & (positions < offset + len(file_rows))
            if in_file.any():
                table = pq.read_table(path, columns=names)
                taken = table.take(file_rows[positions[in_file] - offset]).to_pandas()
                taken.index = np.flatnonzero(in_file)
                parts.append(taken)
                del table
            offset += len(file_rows)
        return pd.concat(parts).sort_index().reset_index(drop=True)


def open_records(paths: list[Path], ids: set[str] | None = None) -> Records:
    filters, id_parts, country_parts = [], [], []
    for path in paths:
        frame = pq.read_table(path, columns=["entity_id", "country_key"]).to_pandas()
        row_filter = None if ids is None else frame["entity_id"].isin(ids).to_numpy()
        if row_filter is not None:
            frame = frame[row_filter]
        filters.append(row_filter)
        id_parts.append(frame["entity_id"].to_numpy(dtype=object))
        country_parts.append(frame["country_key"].to_numpy(dtype=object))
    return Records(paths, filters, np.concatenate(id_parts), np.concatenate(country_parts))


def source_of(ids: np.ndarray) -> np.ndarray:
    """ "S2-123" -> "S2" """
    return np.array([i[:2] for i in ids], dtype=object)


def country_partitions(queries: Records, reference: Records):
    """Yield (country, query positions, reference positions) per shared country."""
    q_groups = pd.Series(queries.country).groupby(queries.country).indices
    r_groups = pd.Series(reference.country).groupby(reference.country).indices
    for country in sorted(q_groups):
        if country in r_groups:
            yield country, np.asarray(q_groups[country]), np.asarray(r_groups[country])


def sorted_block_index(keys: np.ndarray):
    """
    keys: int codes (-1 = no key). Returns (order, sorted_keys) so the block
    for key k is order[searchsorted(sorted_keys, k, 'left'):...'right'].
    """
    order = np.argsort(keys, kind="stable")
    return order, keys[order]


def expand_ranges(starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Concatenate arange(start, start+size) for each pair, vectorized."""
    total = int(sizes.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(sizes)[:-1])), sizes)
    return np.arange(total, dtype=np.int64) + offsets
