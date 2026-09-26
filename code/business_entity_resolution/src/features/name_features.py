"""
Name evidence. S1 side = normalized S1 views; candidate side = S2/S3 views,
with an extra "translated" content view (de-leet + learned S2/S3 -> S1 token
translation, the same transform Stage 4 used for that query set).
"""

import numpy as np

from normalization.country_rules import get_country_rule

from .string_features import exact, lengths, pairwise, prefix_match, suffix_match
from .token_features import overlap


def initials(text: str) -> str:
    tokens = text.split()
    return "".join(t[0] for t in tokens) if len(tokens) >= 2 else ""


def legal_tokens(texts, countries) -> list[str]:
    rules = {c: get_country_rule(c)["legal_forms"] for c in set(countries)}
    return [" ".join(t for t in text.split() if t in rules[c]) for text, c in zip(texts, countries)]


def name_features(s: dict, c: dict, t: dict, indexes: dict) -> dict:
    """s / c: aligned view arrays for S1 / candidate; t: view -> (S1 rows, candidate rows) token matrices."""
    f = {}

    f["name_basic_exact"] = exact(s["name_basic"], c["name_basic"])
    f["name_compact_exact"] = exact(s["name_compact"], c["name_compact"])
    f["name_content_compact_exact"] = exact(s["name_content_compact"], c["name_content_compact"])
    f["name_fingerprint_exact"] = exact(s["name_fingerprint"], c["name_fingerprint"])

    for scorer in ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio", "jaro_winkler"]:
        f[f"name_{scorer}"] = pairwise(s["name_basic"], c["name_basic"], scorer)
    f["name_compact_ratio"] = pairwise(s["name_compact"], c["name_compact"], "ratio")
    f["name_content_ratio"] = pairwise(s["name_content"], c["name_content"], "ratio")
    f["name_content_token_set_ratio"] = pairwise(s["name_content"], c["name_content_x"], "token_set_ratio")
    f["name_content_compact_ratio"] = pairwise(s["name_content_compact"], c["name_content_compact"], "ratio")

    f["name_length_difference"], f["name_length_ratio"] = lengths(s["name_compact"], c["name_compact"])
    f["name_prefix_match"] = prefix_match(s["name_compact"], c["name_compact"])
    f["name_suffix_match"] = suffix_match(s["name_compact"], c["name_compact"])

    f.update(overlap(*t["name_tok"], "name_tok"))
    f.update(overlap(*t["name_content"], "name_content", indexes["name_content"]))

    legal = overlap(*t["legal"], "legal")
    both_legal = (legal["legal_count_s1"] > 0) & (legal["legal_count_cand"] > 0)
    f["legal_both_present"] = both_legal.astype(np.float32)
    f["legal_conflict"] = np.where(both_legal, (legal["legal_overlap_count"] == 0).astype(np.float32), np.nan).astype(np.float32)
    f["legal_s1_present"] = (legal["legal_count_s1"] > 0).astype(np.float32)
    f["legal_cand_present"] = (legal["legal_count_cand"] > 0).astype(np.float32)

    f["name_cand_is_s1_acronym"] = ((s["initials"] != "") & (s["initials"] == c["name_compact"])).astype(np.float32)
    f["name_s1_is_cand_acronym"] = ((c["initials"] != "") & (c["initials"] == s["name_compact"])).astype(np.float32)

    f["name_cand_non_latin"] = (c["name_script"] != "Latin").astype(np.float32)
    f["name_cand_mixed_script"] = c["name_script_mixed"].astype(np.float32)
    return f
