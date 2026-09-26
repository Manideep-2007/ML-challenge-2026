"""
False-positive taxonomy along two axes:

structure (what the wrong match did to the entity)
  FP_ON_EMPTY_TRUTH     S1 has no true match: a single FP drops it from 1.0 to 0.0
  FP_EXTRA_MATCH        S1 also received true matches (over-prediction)
  FP_WRONG_MATCH        S1 has true matches but received none of them

evidence (why the wrong candidate looked right)
  NEAR_DUPLICATE        name and address both highly similar
  NAME_COLLISION        same name, different address
  ADDRESS_COLLISION     same address, different name
  NUMBER_COLLISION      shared house number carries it
  WEAK_EVIDENCE         none of the above strongly similar
"""

import numpy as np
import pandas as pd

LOW, HIGH = 0.6, 0.9


def classify(pairs: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    fp = pairs[pairs["selected"] & (pairs["label"] == 0)].copy()
    ent = entities.iloc[fp["code"].to_numpy()]
    fp["num_truth"] = ent["num_truth"].to_numpy()
    fp["entity_tp"] = ent["tp"].to_numpy()
    fp["structure"] = np.select(
        [fp["num_truth"] == 0, fp["entity_tp"] > 0], ["FP_ON_EMPTY_TRUTH", "FP_EXTRA_MATCH"], "FP_WRONG_MATCH")

    name = fp["name_token_set_ratio"].fillna(0)
    address = fp["address_token_set_ratio"].fillna(0)
    fp["evidence"] = np.select(
        [(name >= HIGH) & (address >= HIGH), (name >= HIGH) & (address < LOW),
         (address >= HIGH) & (name < LOW), fp["number_first_equal"] == 1],
        ["NEAR_DUPLICATE", "NAME_COLLISION", "ADDRESS_COLLISION", "NUMBER_COLLISION"], "WEAK_EVIDENCE")
    fp["margin"] = fp["best_in_s1"] - fp["second_in_s1"]
    fp["high_confidence"] = fp["probability"] >= 0.99
    fp["category"] = fp["structure"] + ":" + fp["evidence"]
    return fp
