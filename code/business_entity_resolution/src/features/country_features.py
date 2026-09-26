"""
Country, source and missingness indicators. Missing evidence is represented
explicitly so the model never confuses "no information" with "mismatch".
Country is not one-hot encoded (test contains an unseen country).
"""

import numpy as np


def country_features(s: dict, c: dict, candidate_source: np.ndarray) -> dict:
    s_country, c_country = s["country_key"], c["country_key"]
    return {
        "country_equal": (s_country == c_country).astype(np.float32),
        "country_missing_s1": (s_country == "").astype(np.float32),
        "country_missing_cand": (c_country == "").astype(np.float32),
        "candidate_source_is_s2": (candidate_source == "S2").astype(np.float32),
        "name_missing_s1": s["name_missing"].astype(np.float32),
        "name_missing_cand": c["name_missing"].astype(np.float32),
        "address_missing_s1": s["address_missing"].astype(np.float32),
        "address_missing_cand": c["address_missing"].astype(np.float32),
        "both_address_missing": (s["address_missing"] & c["address_missing"]).astype(np.float32),
    }
