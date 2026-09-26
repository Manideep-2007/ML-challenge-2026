"""
Hard cases, prioritised as in the Stage 8 plan:

  Type A  retrieved true matches with strong evidence but a low model score (model)
  Type B  true matches never retrieved (blocking)
  Type C  true matches just below the threshold (decision)

plus validation hard negatives (confident false positives) and hard positives
(true matches scored low). Validation cases are for diagnosis only; any
retraining on hard negatives must mine them from TRAIN S1.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW = ["name_raw", "address_raw"]


def attach_raw(frame: pd.DataFrame, normalized: Path) -> pd.DataFrame:
    s1 = pq.read_table(normalized / "train_source1.parquet", columns=["entity_id"] + RAW).to_pandas()
    s1 = s1[s1["entity_id"].isin(set(frame["source1_entity_id"]))].set_index("entity_id")
    wanted = set(frame["candidate_entity_id"])
    cands = []
    for source in ("train_source2", "train_source3"):
        t = pq.read_table(normalized / f"{source}.parquet", columns=["entity_id"] + RAW).to_pandas()
        cands.append(t[t["entity_id"].isin(wanted)])
    cand = pd.concat(cands).set_index("entity_id")
    return frame.assign(
        s1_name=s1.reindex(frame["source1_entity_id"])["name_raw"].values,
        s1_address=s1.reindex(frame["source1_entity_id"])["address_raw"].values,
        cand_name=cand.reindex(frame["candidate_entity_id"])["name_raw"].values,
        cand_address=cand.reindex(frame["candidate_entity_id"])["address_raw"].values,
    )


def hard_cases(fn: pd.DataFrame, threshold: float, per_type: int = 200) -> pd.DataFrame:
    strong = (fn["name_token_set_ratio"].fillna(0) + fn["address_token_set_ratio"].fillna(0)) / 2
    type_a = fn[fn["retrieved"] & (fn["probability"] < 0.5)].assign(strength=strong).nlargest(per_type, "strength")
    type_b = fn[~fn["retrieved"]].sample(min(per_type, int((~fn["retrieved"]).sum())), random_state=0)
    near = fn[fn["retrieved"] & (fn["probability"] >= 0.5) & (fn["probability"] < threshold)]
    type_c = near.nlargest(per_type, "probability")
    return pd.concat([type_a.assign(hard_type="A_model_missed_strong_evidence"),
                      type_b.assign(hard_type="B_not_retrieved"),
                      type_c.assign(hard_type="C_just_below_threshold")], ignore_index=True)


def hard_negatives(fp: pd.DataFrame, min_probability: float = 0.9) -> pd.DataFrame:
    return fp[fp["probability"] >= min_probability].sort_values("probability", ascending=False)


def hard_positives(truth: pd.DataFrame, max_probability: float = 0.5) -> pd.DataFrame:
    return truth[truth["retrieved"] & (truth["probability"] < max_probability)].sort_values("probability")


def empty_entities(entities: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    empty = entities[entities["num_truth"] == 0]
    top = pairs[pairs["rank_in_s1"] == 1].set_index("code")
    return empty.assign(
        top_candidate=top.reindex(empty.index)["candidate_entity_id"].values,
        top_name_similarity=top.reindex(empty.index)["name_token_set_ratio"].values,
        top_address_similarity=top.reindex(empty.index)["address_token_set_ratio"].values,
        predicted_match=empty["num_predicted"] > 0,
    ).sort_values("top_probability", ascending=False)


def multi_match(entities: pd.DataFrame) -> dict:
    multi = entities[entities["num_truth"] >= 2]
    return {
        "s1": int(len(multi)),
        "macro_f05": round(float(multi["f05"].mean()), 6),
        "all_true_retrieved_share": round(float((multi["retrieved_true"] == multi["num_truth"]).mean()), 4),
        "under_predicted_s1": int((multi["num_predicted"] < multi["num_truth"]).sum()),
        "over_predicted_s1": int((multi["num_predicted"] > multi["num_truth"]).sum()),
        "exact_count_s1": int((multi["num_predicted"] == multi["num_truth"]).sum()),
        "missed_links": int(multi["fn"].sum()), "extra_links": int(multi["fp"].sum()),
    }


def case_report(cases: pd.DataFrame, path: Path, limit: int = 60):
    lines = []
    for i, row in enumerate(cases.head(limit).itertuples(), 1):
        prob = "not retrieved" if pd.isna(row.probability) else f"{row.probability:.4f}"
        lines += [
            f"CASE #{i:03d}  [{row.hard_type}]  {row.category}", "=" * 70,
            f"S1   {row.source1_entity_id}: {row.s1_name}  |  {row.s1_address}",
            f"TRUE {row.candidate_entity_id}: {row.cand_name}  |  {row.cand_address}",
            f"retrieved: {row.retrieved}   probability: {prob}   rank in S1: {row.rank_in_s1}",
            f"name token-set {row.name_token_set_ratio}   address token-set {row.address_token_set_ratio}   "
            f"number conflict {row.number_conflict}   hybrid rank {row.hybrid_translated_rank}", "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
