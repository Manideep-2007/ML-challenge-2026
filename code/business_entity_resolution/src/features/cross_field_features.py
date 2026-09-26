"""
Cross-field combinations and relative (within-S1) evidence.

Relative features compare a candidate with the other candidates of the same
S1 entity: its rank on name / address / combined evidence and the margin to
the best candidate. A pair at rank 1 by a wide margin is different evidence
from one tied with 30 others (chain businesses).
"""

import numpy as np
import pandas as pd


def cross_field_features(f: dict) -> dict:
    name = f["name_token_set_ratio"]
    address = f["address_token_set_ratio"]
    stacked = np.vstack([name, address])
    with np.errstate(invalid="ignore"):
        out = {
            "name_address_min": np.nanmin(np.where(np.isnan(stacked), np.inf, stacked), axis=0),
            "name_address_product": name * address,
            "name_address_mean": np.nanmean(stacked, axis=0),
        }
    out["name_address_min"][np.isinf(out["name_address_min"])] = np.nan
    evidence = np.nan_to_num(f["name_content_idf_jaccard"]) + np.nan_to_num(f["address_idf_jaccard"])
    out["combined_idf_evidence"] = evidence.astype(np.float32)
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


RELATIVE_SCORES = ["name_token_set_ratio", "address_token_set_ratio", "combined_idf_evidence", "name_content_idf_jaccard"]


def relative_features(group_ids: np.ndarray, f: dict) -> dict:
    """
    group_ids: S1 id per pair; all candidates of an S1 must be in the same call.
    Missing scores count as -1 (below any real score) for ranking.
    """
    codes, _ = pd.factorize(group_ids)
    n_groups = codes.max() + 1 if len(codes) else 0
    out = {}
    for score in RELATIVE_SCORES:
        values = np.nan_to_num(f[score], nan=-1.0).astype(np.float64)
        order = np.lexsort((-values, codes))
        sorted_codes, sorted_values = codes[order], values[order]
        starts = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
        position = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))

        best = np.full(n_groups, -1.0)
        best[sorted_codes[position == 0]] = sorted_values[position == 0]
        second = np.full(n_groups, np.nan)
        second[sorted_codes[position == 1]] = sorted_values[position == 1]

        # rank with ties sharing the best position ("min" method)
        rank_sorted = np.empty(len(order))
        new_value = np.r_[True, (sorted_codes[1:] != sorted_codes[:-1]) | (sorted_values[1:] != sorted_values[:-1])]
        first_of_run = np.maximum.accumulate(np.where(new_value, np.arange(len(order)), 0))
        rank_sorted = position[first_of_run] + 1
        rank = np.empty(len(order), dtype=np.float32)
        rank[order] = rank_sorted

        out[f"{score}_rank_in_s1"] = rank
        out[f"{score}_margin_to_best"] = (best[codes] - values).astype(np.float32)
        is_best = values == best[codes]
        lead = values - np.nan_to_num(second[codes], nan=-1.0)
        out[f"{score}_lead_over_second"] = np.where(is_best, lead, np.nan).astype(np.float32)

    exact_name = np.nan_to_num(f["name_content_compact_exact"])
    out["s1_candidates_with_exact_content_name"] = np.bincount(codes, weights=exact_name, minlength=n_groups)[codes].astype(np.float32)
    return out
