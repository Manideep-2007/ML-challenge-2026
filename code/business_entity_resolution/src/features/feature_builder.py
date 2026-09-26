"""
Builds the pair feature dataset from a Stage 4 candidate file.

Every record is tokenized once per view (TokenIndex); batches of pairs only
select rows. Candidates are streamed in batches that always contain every
candidate of an S1 entity (the file is sorted by S1), so within-S1 relative
features are exact. Candidate-side translated views reuse the token
translation Stage 4 learned for the same query set.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from blocking.token_alignment import translator
from blocking.token_blocks import deleet

from .address_features import address_features
from .blocking_features import blocking_features
from .country_features import country_features
from .cross_field_features import cross_field_features, relative_features
from .name_features import initials, legal_tokens, name_features
from .name_only_features import name_only_features
from .numeric_features import first_token, long_numbers, numeric_features
from .token_features import TokenIndex

VIEWS = [
    "country_key", "name_basic", "name_compact", "name_content", "name_content_compact",
    "name_fingerprint", "name_numbers", "name_script", "name_script_mixed", "name_missing",
    "address_basic", "address_compact", "address_fingerprint", "address_numbers", "address_missing",
]
ID_COLUMNS = ["source1_entity_id", "candidate_entity_id", "candidate_source"]

# view -> (S1 column, candidate column, weight tokens by IDF)
TOKEN_VIEWS = {
    "name_tok": ("name_basic", "name_basic", False),
    "name_content": ("name_content", "name_content_x", True),
    "legal": ("legal", "legal", False),
    "address": ("address_basic", "address_basic_x", True),
    "number": ("address_numbers", "address_numbers", False),
    "long_number": ("long_numbers", "long_numbers", False),
    "name_number": ("name_numbers", "name_numbers", False),
}


def load_views(paths: list[Path], ids: set[str] | None = None, country: str | None = None) -> pd.DataFrame:
    frames = []
    for path in paths:
        filters = [("country_key", "=", country)] if country is not None else None
        frame = pq.read_table(path, columns=["entity_id"] + VIEWS, filters=filters).to_pandas()
        if ids is not None:
            frame = frame[frame["entity_id"].isin(ids)]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_translator(path: Path):
    table = pd.read_csv(path, keep_default_na=False)
    return translator({r.s2s3_token: (r.s1_token, r.probability, r.support) for r in table.itertuples()})


def add_record_views(frame: pd.DataFrame, translations_dir: Path | None) -> None:
    """Per-record derived views, computed once. translations_dir=None -> S1 side (no translation)."""
    if translations_dir is not None:
        name_tr = load_translator(translations_dir / "token_translation_name.csv")
        address_tr = load_translator(translations_dir / "token_translation_address.csv")
        frame["name_content_x"] = pd.Series([name_tr(deleet(t)) for t in frame["name_content"]], dtype="str")
        frame["name_basic_x"] = pd.Series([deleet(t) for t in frame["name_basic"]], dtype="str")
        frame["address_basic_x"] = pd.Series([address_tr(t) for t in frame["address_basic"]], dtype="str")
        legal_source, initials_source = frame["name_basic_x"], frame["name_content_x"]
    else:
        legal_source, initials_source = frame["name_basic"], frame["name_content"]
    frame["legal"] = pd.Series(legal_tokens(legal_source, frame["country_key"]), dtype="str")
    frame["initials"] = pd.Series([initials(t) for t in initials_source], dtype="str")
    frame["first_number"] = pd.Series([first_token(t) for t in frame["address_numbers"]], dtype="str")
    frame["long_numbers"] = pd.Series([long_numbers(t) for t in frame["address_numbers"]], dtype="str")


def build_indexes(s1: pd.DataFrame, reference: pd.DataFrame, global_df: dict | None = None) -> dict[str, TokenIndex]:
    """global_df: view -> (token -> df, n_docs) over the full reference universe (per-country runs)."""
    return {
        view: TokenIndex(s1[s1_col], reference[ref_col], with_idf,
                         global_df.get(view) if (global_df and with_idf) else None)
        for view, (s1_col, ref_col, with_idf) in TOKEN_VIEWS.items()
    }


def reference_document_frequencies(reference_paths: list[Path], translations_dir: Path) -> dict:
    """Document frequencies of the IDF views over the whole reference universe, country by country."""
    from .token_features import document_frequency_counts
    counts = {view: {} for view, (_, _, with_idf) in TOKEN_VIEWS.items() if with_idf}
    n_docs = 0
    countries = pd.concat([pq.read_table(p, columns=["country_key"]).to_pandas() for p in reference_paths])["country_key"].unique()
    for country in countries:
        part = load_views(reference_paths, country=country)
        add_record_views(part, translations_dir)
        n_docs += len(part)
        for view in counts:
            document_frequency_counts(part[TOKEN_VIEWS[view][1]], counts[view])
        del part
    return {view: (c, n_docs) for view, c in counts.items()}


# Columns the feature functions read directly (token views go through TokenIndex).
SIDE_COLUMNS = [
    "country_key", "name_basic", "name_compact", "name_content", "name_content_compact", "name_fingerprint",
    "name_script", "name_script_mixed", "name_missing", "address_basic", "address_compact",
    "address_fingerprint", "address_missing", "initials", "first_number", "name_content_x", "address_basic_x",
]
# Feature set v2 (name_only) reads nothing beyond these columns.


def side(frame: pd.DataFrame, positions: np.ndarray) -> dict:
    columns = [c for c in SIDE_COLUMNS if c in frame.columns]
    rows = frame[columns].take(positions)
    return {c: rows[c].to_numpy() for c in columns}


FEATURE_SETS = {
    "v1": [],                 # Stage 5 baseline (110 columns)
    "v2": ["name_only"],      # + Stage 8 / E002 name evidence for address-less candidates
}


def batch_features(batch: pd.DataFrame, s1: pd.DataFrame, reference: pd.DataFrame,
                   s1_pos: pd.Index, ref_pos: pd.Index, indexes: dict, channel_names: list[str],
                   feature_set: str = "v1") -> pd.DataFrame:
    q = s1_pos.get_indexer(batch["source1_entity_id"])
    r = ref_pos.get_indexer(batch["candidate_entity_id"])
    if (q < 0).any() or (r < 0).any():
        raise ValueError("candidate pairs reference unknown records")
    s, c = side(s1, q), side(reference, r)
    t = {view: index.pairs(q, r) for view, index in indexes.items()}

    f = {}
    f.update(name_features(s, c, t, indexes))
    if "name_only" in FEATURE_SETS[feature_set]:
        f.update(name_only_features(s, c, f))
    f.update(address_features(s, c, t, indexes))
    f.update(numeric_features(s, c, t))
    f.update(country_features(s, c, batch["candidate_source"].to_numpy()))
    f.update(blocking_features(batch, channel_names))
    f.update(cross_field_features(f))
    f.update(relative_features(batch["source1_entity_id"].to_numpy(), f))

    out = batch[ID_COLUMNS].reset_index(drop=True)
    return pd.concat([out, pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})], axis=1)


def iter_s1_batches(candidates_path: Path, batch_rows: int):
    """Yield candidate batches that never split one S1 entity across batches."""
    carry = None
    for record_batch in pq.ParquetFile(candidates_path).iter_batches(batch_size=batch_rows):
        frame = record_batch.to_pandas()
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        last = frame["source1_entity_id"].iloc[-1]
        tail = frame["source1_entity_id"].to_numpy() == last
        carry = frame[tail]
        if (~tail).any():
            yield frame[~tail].reset_index(drop=True)
    if carry is not None and len(carry):
        yield carry.reset_index(drop=True)


def build_pair_features(
    candidates_path: Path,
    s1_paths: list[Path],
    reference_paths: list[Path],
    translations_dir: Path,
    ground_truth: dict[str, list[str]] | None,
    out_path: Path,
    channel_names: list[str],
    batch_rows: int = 2_000_000,
    log=print,
    feature_set: str = "v1",
    s1_ids: set[str] | None = None,
) -> dict:
    """s1_ids: restrict to these S1 (e.g. a validation subset for a Stage 8 experiment)."""
    start = time.time()
    query_ids = set(pq.read_table(candidates_path, columns=["source1_entity_id"])["source1_entity_id"].unique().to_pylist())
    if s1_ids is not None:
        query_ids &= s1_ids
    s1 = load_views(s1_paths, query_ids)
    reference = load_views(reference_paths)
    log(f"  loaded {len(s1):,} S1 and {len(reference):,} reference records ({time.time() - start:.0f}s)")

    add_record_views(s1, None)
    add_record_views(reference, translations_dir)
    log(f"  record views ready ({time.time() - start:.0f}s)")
    indexes = build_indexes(s1, reference)
    log(f"  token indexes ready ({time.time() - start:.0f}s)")
    s1_pos = pd.Index(s1["entity_id"])
    ref_pos = pd.Index(reference["entity_id"])

    truth = None
    if ground_truth is not None:
        truth = {f"{k}|{m}" for k, v in ground_truth.items() for m in v}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer, rows, positives = None, 0, 0
    try:
        for batch in iter_s1_batches(candidates_path, batch_rows):
            if s1_ids is not None:
                batch = batch[batch["source1_entity_id"].isin(s1_ids)].reset_index(drop=True)
                if not len(batch):
                    continue
            features = batch_features(batch, s1, reference, s1_pos, ref_pos, indexes, channel_names, feature_set)
            if truth is not None:
                keys = features["source1_entity_id"] + "|" + features["candidate_entity_id"]
                features["label"] = keys.isin(truth).astype(np.int8)
                positives += int(features["label"].sum())
            table = pa.Table.from_pandas(features, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(features)
            log(f"  {rows:,} pairs  ({positives:,} positive)  {time.time() - start:.0f}s")
    finally:
        if writer is not None:
            writer.close()

    return {"pairs": rows, "positives": positives, "seconds": round(time.time() - start, 1),
            "s1_entities": len(s1), "output": str(out_path)}
