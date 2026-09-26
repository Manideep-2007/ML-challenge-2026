"""
Training data for the pairwise matcher.

The candidate pair files are too large to load whole (tens of millions of
pairs x ~110 features), so training data is a stratified negative sample:
all positives and all hard negatives are kept, medium and easy negatives are
sampled, and each kept row carries an inverse-sampling-rate weight so a
weighted model sees the true candidate distribution.

Splits are always by Source 1 entity, never by pair row.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ID_COLUMNS = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
LABEL = "label"

# Difficulty of a negative = strongest single similarity signal it has.
DIFFICULTY_COLUMNS = ["name_token_set_ratio", "address_token_set_ratio", "name_content_idf_jaccard", "address_idf_jaccard"]
HARD_THRESHOLD = 0.8
MEDIUM_THRESHOLD = 0.5

FEATURE_GROUPS = {
    "name": ("name_", "legal_"),
    "address": ("address_",),
    "numeric": ("number_", "long_number_"),
    "country_missing": ("country_", "candidate_source_", "both_address_missing"),
    "cross_field": ("combined_idf_evidence", "s1_candidates_with_exact_content_name"),
    "blocking": ("blocked_", "blocking_channel_count", "candidates_for_source1", "hybrid_translated_rank"),
}


def feature_columns(path: Path) -> list[str]:
    names = pq.ParquetFile(path).schema_arrow.names
    return [c for c in names if c not in ID_COLUMNS and c != LABEL]


def group_of(feature: str) -> str:
    """Relative features (rank / margin / lead) belong to the group of their base score."""
    base = feature.split("_rank_in_s1")[0].split("_margin_to_best")[0].split("_lead_over_second")[0]
    if base in ("name_address_min", "name_address_product", "name_address_mean"):
        return "cross_field"
    for group, prefixes in FEATURE_GROUPS.items():
        if base.startswith(prefixes):
            return group
    return "other"


def select_features(features: list[str], groups: list[str] | None) -> list[str]:
    if groups is None:
        return features
    return [f for f in features if group_of(f) in groups]


def difficulty(frame: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(frame[DIFFICULTY_COLUMNS].to_numpy(dtype=np.float32), nan=0.0).max(axis=1)


def sample_training_pairs(path: Path, hard_rate: float, medium_rate: float, easy_rate: float, seed: int,
                          batch_rows: int = 2_000_000, keep_candidate_ids: bool = False) -> pd.DataFrame:
    """
    Stream the pair file; keep every positive and sample negatives per
    difficulty bucket (hard at the highest rate). sample_weight = 1 / rate.
    keep_candidate_ids: keep candidate_entity_id (needed to join mined hard examples).
    """
    rng = np.random.default_rng(seed)
    kept = []
    drop = ["candidate_source"] if keep_candidate_ids else ["candidate_entity_id", "candidate_source"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows):
        frame = batch.to_pandas().drop(columns=drop)
        score = difficulty(frame)
        positive = frame[LABEL].to_numpy() == 1
        hard = ~positive & (score >= HARD_THRESHOLD)
        medium = ~positive & (score >= MEDIUM_THRESHOLD) & (score < HARD_THRESHOLD)
        easy = ~positive & (score < MEDIUM_THRESHOLD)
        draw = rng.random(len(frame))
        keep = positive | (hard & (draw < hard_rate)) | (medium & (draw < medium_rate)) | (easy & (draw < easy_rate))
        weight = np.select([positive, hard, medium, easy],
                           [1.0, 1.0 / hard_rate, 1.0 / medium_rate, 1.0 / easy_rate]).astype(np.float32)
        bucket = np.select([positive, hard, medium], ["positive", "hard", "medium"], "easy")
        frame = frame[keep].copy()
        frame["sample_weight"] = weight[keep]
        frame["difficulty_bucket"] = bucket[keep]
        kept.append(frame)
    return pd.concat(kept, ignore_index=True)


def bucket_counts(path: Path, batch_rows: int = 2_000_000) -> dict:
    """Full (unsampled) counts of positives and negative difficulty buckets."""
    counts = {"positive": 0, "hard": 0, "medium": 0, "easy": 0, "s1": set()}
    columns = DIFFICULTY_COLUMNS + [LABEL, "source1_entity_id"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=columns):
        frame = batch.to_pandas()
        score = difficulty(frame)
        positive = frame[LABEL].to_numpy() == 1
        counts["positive"] += int(positive.sum())
        counts["hard"] += int((~positive & (score >= HARD_THRESHOLD)).sum())
        counts["medium"] += int((~positive & (score >= MEDIUM_THRESHOLD) & (score < HARD_THRESHOLD)).sum())
        counts["easy"] += int((~positive & (score < MEDIUM_THRESHOLD)).sum())
        counts["s1"].update(frame["source1_entity_id"].unique())
    counts["s1"] = len(counts["s1"])
    return counts


def split_by_s1(frame: pd.DataFrame, dev_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Internal early-stopping split, grouped by Source 1 entity."""
    s1 = np.array(sorted(frame["source1_entity_id"].unique()))
    rng = np.random.default_rng(seed)
    dev_ids = set(rng.choice(s1, size=int(len(s1) * dev_fraction), replace=False))
    in_dev = frame["source1_entity_id"].isin(dev_ids).to_numpy()
    return frame[~in_dev].reset_index(drop=True), frame[in_dev].reset_index(drop=True)
