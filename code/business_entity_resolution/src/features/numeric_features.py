"""
Number evidence (canonical integers from Stage 3). Conflicts are explicit:
both sides have numbers but share none is strong negative evidence, which is
different from "no numbers to compare" (NaN).
"""

import numpy as np

from .token_features import overlap


def first_token(text: str) -> str:
    return text.split(" ", 1)[0] if text else ""


def long_numbers(text: str, min_digits: int = 4) -> str:
    return " ".join(n for n in text.split() if len(n) >= min_digits)


def numeric_features(s: dict, c: dict, t: dict) -> dict:
    f = overlap(*t["number"], "number")
    both = (f["number_count_s1"] > 0) & (f["number_count_cand"] > 0)
    f["number_conflict"] = np.where(both, (f["number_overlap_count"] == 0).astype(np.float32), np.nan).astype(np.float32)
    f["number_set_exact"] = np.where(both, (f["number_jaccard"] == 1).astype(np.float32), np.nan).astype(np.float32)
    f["number_first_equal"] = np.where(both, (s["first_number"] == c["first_number"]).astype(np.float32), np.nan).astype(np.float32)

    long = overlap(*t["long_number"], "long_number")
    both_long = (long["long_number_count_s1"] > 0) & (long["long_number_count_cand"] > 0)
    f["long_number_overlap"] = np.where(both_long, (long["long_number_overlap_count"] > 0).astype(np.float32), np.nan).astype(np.float32)

    name_num = overlap(*t["name_number"], "name_number")
    both_name = (name_num["name_number_count_s1"] > 0) & (name_num["name_number_count_cand"] > 0)
    f["name_number_conflict"] = np.where(both_name, (name_num["name_number_overlap_count"] == 0).astype(np.float32), np.nan).astype(np.float32)
    f["name_number_s1_only"] = ((name_num["name_number_count_s1"] > 0) & (name_num["name_number_count_cand"] == 0)).astype(np.float32)
    return f
