"""
Stage 8 / E001: hard-example mining from TRAIN S1 only (never validation).

1. Split the train_sample S1 into two folds; fit a quick model on each fold's
   complement and score the held-out fold over ALL its candidate pairs
   (out-of-fold, so the scores are not optimistic).
2. Hard negatives: false pairs scored >= NEG_P (confident false merges).
   Hard positives: true pairs scored < POS_P.
3. The training sample gains every hard example; hard rows are up-weighted by BOOST.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NEG_P = 0.5
POS_P = 0.5
BOOST = 3.0
PAIR_KEY = ["source1_entity_id", "candidate_entity_id"]
QUICK_PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_child_samples": 50,
                "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "n_estimators": 400,
                "random_state": 42, "n_jobs": -1, "verbosity": -1}


def fit_quick_folds(sample: pd.DataFrame, features: list[str], folds: int = 2, seed: int = 42) -> dict:
    """Quick model k is trained on the sampled pairs of every S1 NOT in fold k."""
    s1 = np.array(sorted(sample["source1_entity_id"].unique()))
    fold_of = pd.Series(np.random.default_rng(seed).integers(0, folds, len(s1)), index=s1)
    fold = fold_of.reindex(sample["source1_entity_id"]).to_numpy()
    models = {}
    for k in range(folds):
        model = lgb.LGBMClassifier(**QUICK_PARAMS)
        model.fit(sample.loc[fold != k, features], sample.loc[fold != k, "label"],
                  sample_weight=sample.loc[fold != k, "sample_weight"])
        models[k] = model
    return {"fold_of": fold_of, "models": models}


def mine(path: Path, features: list[str], quick: dict, batch_rows: int = 2_000_000) -> pd.DataFrame:
    """Out-of-fold scoring of every train_sample pair; returns the hard examples."""
    kept = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows):
        frame = batch.to_pandas().drop(columns=["candidate_source"])
        fold = quick["fold_of"].reindex(frame["source1_entity_id"]).to_numpy()
        prob = np.full(len(frame), np.nan, dtype=np.float32)
        for k, model in quick["models"].items():
            held = fold == k
            if held.any():
                prob[held] = model.predict_proba(frame.loc[held, features])[:, 1]
        label = frame["label"].to_numpy()
        hard = ((label == 0) & (prob >= NEG_P)) | ((label == 1) & (prob < POS_P))
        kept.append(frame[hard].assign(oof_probability=prob[hard]))
    return pd.concat(kept, ignore_index=True)


def add_hard_examples(sample: pd.DataFrame, hard: pd.DataFrame, boost: float = BOOST) -> tuple[pd.DataFrame, dict]:
    """
    Up-weight hard rows already in the stratified sample; append the others with
    weight `boost` (their original sampling weight would have been 1 / rate, so
    boost is a deliberate emphasis, not an unbiased weight).
    """
    hard_keys = pd.MultiIndex.from_frame(hard[PAIR_KEY])
    sample_keys = pd.MultiIndex.from_frame(sample[PAIR_KEY])
    in_sample = sample_keys.isin(hard_keys)
    sample = sample.copy()
    sample.loc[in_sample, "sample_weight"] = sample.loc[in_sample, "sample_weight"] * boost

    extra = hard[~hard_keys.isin(sample_keys)].drop(columns=["oof_probability"])
    extra = extra.assign(sample_weight=np.float32(boost), difficulty_bucket="mined_hard")
    combined = pd.concat([sample, extra[sample.columns]], ignore_index=True)
    return combined, {
        "hard_examples_mined": int(len(hard)),
        "hard_negatives": int((hard["label"] == 0).sum()),
        "hard_positives": int((hard["label"] == 1).sum()),
        "already_in_sample_upweighted": int(in_sample.sum()),
        "added_to_sample": int(len(extra)),
        "boost": boost,
    }
