"""
Master error tables for a frozen system (model predictions + decision rules)
on the validation split.

Memory-lean: of the ~61M validation pairs only the relevant ones are kept -
every retrieved true pair plus every pair with probability >= FLOOR (no
decision threshold goes lower) - together with a handful of evidence features.
True pairs Stage 4 never retrieved come from the ground truth.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from decision.decision_engine import DecisionParams, entity_outcomes, select
from decision.ranking import prepare

FLOOR = 0.02
EVIDENCE = [
    "name_token_set_ratio", "address_token_set_ratio", "name_content_idf_jaccard", "address_idf_jaccard",
    "number_conflict", "number_first_equal", "legal_conflict", "address_missing_cand", "name_cand_non_latin",
    "name_content_shared_idf_max", "hybrid_translated_rank", "candidates_for_source1", "blocking_channel_count",
]


@dataclass
class ErrorTables:
    pairs: pd.DataFrame      # relevant candidate pairs with decision + evidence
    truth: pd.DataFrame      # every true pair: retrieved?, probability, selected?
    entities: pd.DataFrame   # one row per validation S1
    s1_index: pd.Index
    true_counts: np.ndarray


def load_pairs(predictions: Path, features: Path) -> pd.DataFrame:
    prob = pq.read_table(predictions, columns=["probability"])["probability"].to_numpy()
    label = pq.read_table(predictions, columns=["label"])["label"].to_numpy()
    keep = np.flatnonzero((prob >= FLOOR) | (label == 1))
    frame = pd.DataFrame({"row": keep, "probability": prob[keep], "label": label[keep].astype(np.int8)})
    del prob, label
    for column in ["source1_entity_id", "candidate_entity_id"]:
        frame[column] = pq.read_table(predictions, columns=[column])[column].take(keep).to_pandas().to_numpy()
    for column in EVIDENCE:
        frame[column] = pq.read_table(features, columns=[column])[column].take(keep).to_numpy()
    frame["is_s2"] = frame["candidate_entity_id"].str.startswith("S2-")
    return frame


def build(predictions: Path, features: Path, ground_truth: dict, country: pd.Series,
          params: DecisionParams) -> ErrorTables:
    s1_index = pd.Index(sorted(ground_truth))
    true_counts = np.array([len(ground_truth[k]) for k in s1_index], dtype=np.float64)
    countries = country.reindex(s1_index).to_numpy()

    pairs = load_pairs(predictions, features)
    code = s1_index.get_indexer(pairs["source1_entity_id"]).astype(np.int64)
    cand_code, _ = pd.factorize(pairs["candidate_entity_id"])
    scored = prepare(code, cand_code.astype(np.int64), pairs["is_s2"].to_numpy(),
                     pairs["probability"].to_numpy(np.float32), pairs["label"].to_numpy(np.int8), countries, floor=FLOOR)
    keep = select(scored, params)

    # map decision + rank back to the pair table (scored.row = position in `pairs`)
    pairs["code"] = code
    pairs["selected"] = False
    pairs.loc[scored.row[keep], "selected"] = True
    pairs["rank_in_s1"] = np.nan
    pairs.loc[scored.row, "rank_in_s1"] = scored.rank + 1
    pairs["best_in_s1"] = np.nan
    pairs.loc[scored.row, "best_in_s1"] = scored.best
    pairs["second_in_s1"] = np.nan
    pairs.loc[scored.row, "second_in_s1"] = scored.second

    outcomes = entity_outcomes(scored, keep, true_counts)

    # every true pair, retrieved or not
    true_rows = [(k, m) for k, v in ground_truth.items() for m in v]
    truth = pd.DataFrame(true_rows, columns=["source1_entity_id", "candidate_entity_id"])
    positives = pairs[pairs["label"] == 1].drop(columns=["label"])
    truth = truth.merge(positives, on=["source1_entity_id", "candidate_entity_id"], how="left")
    truth["retrieved"] = truth["probability"].notna()
    truth["selected"] = truth["selected"].fillna(False).astype(bool)
    truth["code"] = s1_index.get_indexer(truth["source1_entity_id"])

    # entity table
    n = len(s1_index)
    n_candidates = np.zeros(n)
    retrieved_true = np.bincount(truth.loc[truth["retrieved"], "code"], minlength=n)
    top = np.zeros(n)
    second = np.zeros(n)
    top[scored.code[scored.rank == 0]] = scored.prob[scored.rank == 0]
    second[scored.code[scored.rank == 1]] = scored.prob[scored.rank == 1]
    counts_above = {t: np.bincount(scored.code[scored.prob >= t], minlength=n) for t in (0.9, 0.95, 0.99)}
    cand_per_s1 = pairs.drop_duplicates("code").set_index("code")["candidates_for_source1"]
    n_candidates[cand_per_s1.index.to_numpy()] = cand_per_s1.to_numpy()

    truth_sources = truth.groupby("code")["candidate_entity_id"].agg(lambda ids: "".join(sorted({i[:2] for i in ids})))
    composition = pd.Series("empty", index=np.arange(n))
    composition.loc[truth_sources.index] = truth_sources.map({"S2": "only_S2", "S3": "only_S3", "S2S3": "S2_and_S3"})

    entities = pd.DataFrame({
        "s1_id": s1_index, "country": countries,
        "num_truth": true_counts.astype(int), "num_predicted": outcomes["n_pred"].astype(int),
        "tp": outcomes["tp"].astype(int),
        "fp": (outcomes["n_pred"] - outcomes["tp"]).astype(int),
        "fn": (true_counts - outcomes["tp"]).astype(int),
        "precision": outcomes["precision"], "recall": outcomes["recall"], "f05": outcomes["f05"],
        "retrieved_true": retrieved_true,
        "candidate_recall": np.divide(retrieved_true, true_counts, out=np.ones(n), where=true_counts > 0),
        "candidate_count": n_candidates,
        "top_probability": top, "second_probability": second, "margin": top - second,
        "above_0.9": counts_above[0.9], "above_0.95": counts_above[0.95], "above_0.99": counts_above[0.99],
        "truth_composition": composition.to_numpy(),
    })
    return ErrorTables(pairs, truth, entities, s1_index, true_counts)
