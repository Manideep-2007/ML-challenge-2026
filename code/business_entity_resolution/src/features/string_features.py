"""
Vectorized pairwise string similarities.

Missing evidence is NaN, not 0: if either side is empty the similarity is
unknown rather than "different" (LightGBM handles NaN natively).
Scores come from RapidFuzz process.cpdist (C++, multi-threaded) because
per-pair Python calls are too slow for tens of millions of pairs.
"""

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler


SCORERS = {
    "ratio": fuzz.ratio,
    "partial_ratio": fuzz.partial_ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "token_set_ratio": fuzz.token_set_ratio,
    "jaro_winkler": JaroWinkler.normalized_similarity,
}
SCALE = {"jaro_winkler": 1.0}


def present(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a != "") & (b != "")


def pairwise(a: np.ndarray, b: np.ndarray, scorer: str) -> np.ndarray:
    scores = process.cpdist(list(a), list(b), scorer=SCORERS[scorer], workers=-1).astype(np.float32)
    scores /= 100.0 if scorer not in SCALE else SCALE[scorer]
    scores[~present(a, b)] = np.nan
    return scores


def exact(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = (a == b).astype(np.float32)
    out[~present(a, b)] = np.nan
    return out


def prefix_match(a: np.ndarray, b: np.ndarray, n: int = 4) -> np.ndarray:
    """First n characters equal (on compact views)."""
    pa = np.array([s[:n] for s in a], dtype=object)
    pb = np.array([s[:n] for s in b], dtype=object)
    return exact(pa, pb)


def suffix_match(a: np.ndarray, b: np.ndarray, n: int = 4) -> np.ndarray:
    pa = np.array([s[-n:] for s in a], dtype=object)
    pb = np.array([s[-n:] for s in b], dtype=object)
    return exact(pa, pb)


def lengths(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(absolute length difference, min/max length ratio); NaN if either empty."""
    la = np.fromiter((len(s) for s in a), dtype=np.float32, count=len(a))
    lb = np.fromiter((len(s) for s in b), dtype=np.float32, count=len(b))
    diff = np.abs(la - lb)
    ratio = np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)
    missing = ~present(a, b)
    diff[missing] = np.nan
    ratio[missing] = np.nan
    return diff, ratio
