"""Macro F0.5, precision, recall and candidate recall per subgroup."""

import numpy as np
import pandas as pd

CANDIDATE_BINS = [0, 5, 10, 25, 50, 100, 500, np.inf]
CANDIDATE_LABELS = ["1-5", "6-10", "11-25", "26-50", "51-100", "101-500", "500+"]
LENGTH_BINS = [0, 5, 10, 20, 30, np.inf]
LENGTH_LABELS = ["1-5", "6-10", "11-20", "21-30", "31+"]


def summarize(entities: pd.DataFrame, group: pd.Series, name: str) -> pd.DataFrame:
    frame = entities.assign(**{name: group.to_numpy()})
    out = frame.groupby(name, observed=True).agg(
        s1=("f05", "size"), macro_f05=("f05", "mean"), macro_precision=("precision", "mean"),
        macro_recall=("recall", "mean"), candidate_recall=("candidate_recall", "mean"),
        fp=("fp", "sum"), fn=("fn", "sum"),
        f05_points_lost=("f05", lambda f: float((1 - f).sum())))
    out["share_of_total_loss"] = (out["f05_points_lost"] / (1 - entities["f05"]).sum()).round(4)
    return out.round(6).reset_index()


def all_subgroups(entities: pd.DataFrame, s1_name_length: pd.Series) -> dict[str, pd.DataFrame]:
    match_bin = np.where(entities["num_truth"] >= 4, "4+", entities["num_truth"].astype(str))
    return {
        "match_count": summarize(entities, pd.Series(match_bin), "true_matches"),
        "country": summarize(entities, entities["country"], "country"),
        "source": summarize(entities, entities["truth_composition"], "truth_sources"),
        "candidate_count": summarize(entities, pd.cut(entities["candidate_count"], CANDIDATE_BINS,
                                                      labels=CANDIDATE_LABELS), "candidates"),
        "name_length": summarize(entities, pd.cut(s1_name_length, LENGTH_BINS, labels=LENGTH_LABELS),
                                 "s1_name_length"),
    }


def name_address_matrix(pairs: pd.DataFrame) -> pd.DataFrame:
    """Pair-level decision behaviour per (name similarity, address similarity) bucket."""
    def bucket(s):
        return pd.cut(s.fillna(-1), [-2, -0.5, 0.6, 0.9, 1.01], labels=["missing", "low", "medium", "high"])
    frame = pairs.assign(name_bucket=bucket(pairs["name_token_set_ratio"]),
                         address_bucket=bucket(pairs["address_token_set_ratio"]))
    out = frame.groupby(["name_bucket", "address_bucket"], observed=True).agg(
        pairs=("label", "size"), true_pairs=("label", "sum"), selected=("selected", "sum"),
        correct_selected=("selected", lambda s: int((s & (frame.loc[s.index, "label"] == 1)).sum())))
    out["positive_rate"] = (out["true_pairs"] / out["pairs"]).round(4)
    out["selected_precision"] = (out["correct_selected"] / out["selected"].replace(0, np.nan)).round(4)
    out["true_pair_recall"] = (out["correct_selected"] / out["true_pairs"].replace(0, np.nan)).round(4)
    return out.reset_index()
