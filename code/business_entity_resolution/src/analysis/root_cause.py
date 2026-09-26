"""
Root-cause ranking by F0.5 actually lost.

Every imperfect entity lost (1 - F0.5) points; that loss is shared equally
among its error pairs (its FNs and FPs). Summing per category gives the
macro-F0.5 gain available if that category were fixed completely.
"""

import numpy as np
import pandas as pd


def rank(fn: pd.DataFrame, fp: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    errors = entities["fn"].to_numpy() + entities["fp"].to_numpy()
    loss = 1.0 - entities["f05"].to_numpy()
    share = np.divide(loss, errors, out=np.zeros_like(loss), where=errors > 0)
    n = len(entities)

    frames = []
    for kind, frame, cause, component in [
        ("false_negative", fn, "category", "component"),
        ("false_positive", fp, "category", None),
    ]:
        f = pd.DataFrame({
            "kind": kind,
            "root_cause": frame[cause].to_numpy(),
            "component": frame[component].to_numpy() if component else "features/model + decision",
            "code": frame["code"].to_numpy(),
        })
        f["f05_points"] = share[f["code"]]
        frames.append(f)
    all_errors = pd.concat(frames, ignore_index=True)
    table = all_errors.groupby(["kind", "root_cause", "component"]).agg(
        error_pairs=("code", "size"), entities=("code", "nunique"), f05_points=("f05_points", "sum")).reset_index()
    table["macro_f05_gain_if_fixed"] = (table["f05_points"] / n).round(6)
    table["share_of_total_loss"] = (table["f05_points"] / loss.sum()).round(4)
    return table.sort_values("f05_points", ascending=False).reset_index(drop=True)


def by_component(table: pd.DataFrame) -> pd.DataFrame:
    return table.groupby("component").agg(
        error_pairs=("error_pairs", "sum"), f05_points=("f05_points", "sum"),
        macro_f05_gain_if_fixed=("macro_f05_gain_if_fixed", "sum"),
        share_of_total_loss=("share_of_total_loss", "sum")).sort_values("f05_points", ascending=False).reset_index()
