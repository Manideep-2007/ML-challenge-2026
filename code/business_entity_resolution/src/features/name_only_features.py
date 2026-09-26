"""
Stage 8 / E002 (feature set v2): name evidence for candidates that have no
address. Stage 7 error analysis: 26% of missed true pairs have an empty
candidate address, leaving only a noisy name (typos, leetspeak, truncation).

  name_char3_jaccard        Jaccard of character 3-gram sets (typo / spacing robust)
  name_deleet_ratio         edit similarity after undoing leetspeak on the candidate
  name_deleet_jaro_winkler
  name_wratio               RapidFuzz WRatio (best of partial / token ratios)
  name_common_prefix_share  shared leading characters / shorter name length
  name_evidence_no_address  name evidence only where the candidate address is empty (else NaN)
  char3_no_address          same for the 3-gram Jaccard
"""

import numpy as np
from rapidfuzz import fuzz, process

from blocking.token_blocks import deleet

from .string_features import pairwise, present


def char_ngrams(text: str, n: int = 3) -> frozenset:
    return frozenset(text[i:i + n] for i in range(max(len(text) - n + 1, 1))) if text else frozenset()


def char_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if x and y:
            gx, gy = char_ngrams(x), char_ngrams(y)
            out[i] = len(gx & gy) / len(gx | gy)
    return out


def common_prefix_share(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if x and y:
            n = 0
            for cx, cy in zip(x, y):
                if cx != cy:
                    break
                n += 1
            out[i] = n / min(len(x), len(y))
    return out


def name_only_features(s: dict, c: dict, f: dict) -> dict:
    s_compact = s["name_compact"]
    c_deleet = np.array([deleet(t).replace(" ", "") for t in c["name_basic"]], dtype=object)

    out = {
        "name_char3_jaccard": char_jaccard(s_compact, c["name_compact"]),
        "name_deleet_ratio": pairwise(s_compact, c_deleet, "ratio"),
        "name_deleet_jaro_winkler": pairwise(s_compact, c_deleet, "jaro_winkler"),
        "name_common_prefix_share": common_prefix_share(s_compact, c["name_compact"]),
    }
    wratio = process.cpdist(list(s["name_basic"]), list(c["name_basic"]), scorer=fuzz.WRatio, workers=-1)
    out["name_wratio"] = np.where(present(s["name_basic"], c["name_basic"]), wratio / 100.0, np.nan).astype(np.float32)

    no_address = c["address_missing"].astype(bool)
    out["name_evidence_no_address"] = np.where(no_address, f["name_token_set_ratio"], np.nan).astype(np.float32)
    out["char3_no_address"] = np.where(no_address, out["name_char3_jaccard"], np.nan).astype(np.float32)
    return out
