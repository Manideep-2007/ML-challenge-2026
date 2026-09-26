"""
False-negative funnel: for every missed true pair, where in the pipeline it was
lost, then what the evidence looked like.

  retrieved?  no  -> FN_NOT_RETRIEVED                      (blocking)
  p < 0.5         -> FN_LOW_MODEL_SCORE:<evidence type>    (features / model)
  p < threshold   -> FN_THRESHOLD                          (decision threshold)
  otherwise       -> FN_DECISION_RULE                      (exclusivity / caps)
"""

import numpy as np
import pandas as pd

LOW, HIGH = 0.6, 0.9


def evidence_type(frame: pd.DataFrame) -> np.ndarray:
    name = frame["name_token_set_ratio"].fillna(0)
    address = frame["address_token_set_ratio"].fillna(0)
    return np.select(
        [frame["address_missing_cand"] == 1,
         frame["name_cand_non_latin"] == 1,
         (name < LOW) & (address < LOW),
         (name < LOW) & (address >= LOW),
         (address < LOW) & (name >= LOW),
         frame["number_conflict"] == 1],
        ["MISSING_DATA", "TRANSLITERATION", "BOTH_NOISY", "NAME_NOISE", "ADDRESS_NOISE", "NUMBER_NOISE"],
        "STRONG_EVIDENCE_LOW_SCORE")


def classify(truth: pd.DataFrame, threshold: float) -> pd.DataFrame:
    fn = truth[~truth["selected"]].copy()
    prob = fn["probability"].fillna(-1)
    stage = np.select([~fn["retrieved"], prob < 0.5, prob < threshold],
                      ["NOT_RETRIEVED", "LOW_MODEL_SCORE", "THRESHOLD"], "DECISION_RULE")
    fn["component"] = pd.Series(stage, index=fn.index).map({
        "NOT_RETRIEVED": "blocking", "LOW_MODEL_SCORE": "features/model",
        "THRESHOLD": "decision threshold", "DECISION_RULE": "decision rule"}).to_numpy()
    detail = np.where(stage == "LOW_MODEL_SCORE", evidence_type(fn), "")
    fn["category"] = np.where(detail != "", "FN_LOW_MODEL_SCORE:" + detail, "FN_" + stage)
    return fn
