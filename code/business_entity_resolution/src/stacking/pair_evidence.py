"""
Stage 9 / E006 — pair-local evidence for the collective matcher.

Recomputes, for the plausible pairs only (stage-1 p >= FLOOR), the stage-1
pairwise features of feature set v2 (name / address / number / legal / blocking
provenance + E002 name-only evidence) plus the E004 street features. Within-S1
relative features are left out: they need every candidate of the S1, and the
collective matcher already has its own within-S1 structure.

IDF comes from document frequencies over the whole reference universe
(global_df), so only the records in the pairs are loaded -- the numbers equal
the Stage 5 / inference features.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from features.feature_builder import (add_record_views, batch_features, build_indexes, load_views)
from features.street_features import add_street_views, street_features

CHANNELS = ["content_name", "fingerprint_name", "fingerprint_address", "hybrid_translated"]
PROVENANCE = ["blocking_channels", "num_blocking_channels", "hybrid_translated_rank", "candidates_for_source1"]
RELATIVE_MARKERS = ("_rank_in_s1", "_margin_to_best", "_lead_over_second")
DROP = {"s1_candidates_with_exact_content_name"}
PREFIX = "x_"


def _street_views(paths, ids) -> pd.DataFrame:
    frames = [pq.read_table(p, columns=["entity_id", "country_key", "address_nfkc", "name_nfkc"]).to_pandas() for p in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["entity_id"].isin(ids)].reset_index(drop=True)
    add_street_views(frame)
    return frame.set_index("entity_id")[["street_core", "street_number", "legal_dotted"]]


def pair_evidence(pairs: pd.DataFrame, s1_paths, ref_paths, translations_dir, global_df,
                  country: str | None = None, batch_rows: int = 1_000_000, log=print) -> pd.DataFrame:
    """
    pairs: source1_entity_id, candidate_entity_id, candidate_source + PROVENANCE columns.
    Returns one row per pair (same order) with PREFIX-ed feature columns.
    """
    s1_ids, ref_ids = set(pairs["source1_entity_id"]), set(pairs["candidate_entity_id"])
    s1 = load_views(s1_paths, s1_ids, country)
    reference = load_views(ref_paths, country=country)
    reference = reference[reference["entity_id"].isin(ref_ids)].reset_index(drop=True)
    add_record_views(s1, None)
    add_record_views(reference, translations_dir)
    indexes = build_indexes(s1, reference, global_df)
    s1_pos, ref_pos = pd.Index(s1["entity_id"]), pd.Index(reference["entity_id"])
    street_s1 = _street_views(s1_paths, s1_ids)
    street_ref = _street_views(ref_paths, ref_ids)
    log(f"    evidence views: {len(s1):,} S1, {len(reference):,} reference")

    parts = []
    for begin in range(0, len(pairs), batch_rows):
        batch = pairs.iloc[begin:begin + batch_rows].reset_index(drop=True)
        f = batch_features(batch, s1, reference, s1_pos, ref_pos, indexes, CHANNELS, feature_set="v2")
        keep = [c for c in f.columns if c not in ("source1_entity_id", "candidate_entity_id", "candidate_source")
                and not any(m in c for m in RELATIVE_MARKERS) and c not in DROP]
        f = f[keep]
        cols = ["street_core", "street_number", "legal_dotted"]
        a = street_s1.reindex(batch["source1_entity_id"])
        b = street_ref.reindex(batch["candidate_entity_id"])
        for k, v in street_features({c: a[c].to_numpy() for c in cols}, {c: b[c].to_numpy() for c in cols}).items():
            f[k] = v
        parts.append(f.add_prefix(PREFIX))
    return pd.concat(parts, ignore_index=True).astype(np.float32)
