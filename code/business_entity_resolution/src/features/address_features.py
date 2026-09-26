"""
Address evidence. Addresses are treated separately from names: S2/S3
addresses are truncated, reordered and use state codes/names/native script,
so order-invariant and translated token evidence matter most.
"""

from .string_features import exact, lengths, pairwise
from .token_features import overlap


def address_features(s: dict, c: dict, t: dict, indexes: dict) -> dict:
    f = {}
    f["address_basic_exact"] = exact(s["address_basic"], c["address_basic"])
    f["address_compact_exact"] = exact(s["address_compact"], c["address_compact"])
    f["address_fingerprint_exact"] = exact(s["address_fingerprint"], c["address_fingerprint"])

    for scorer in ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"]:
        f[f"address_{scorer}"] = pairwise(s["address_basic"], c["address_basic"], scorer)
    f["address_translated_token_set_ratio"] = pairwise(s["address_basic"], c["address_basic_x"], "token_set_ratio")

    f["address_length_difference"], f["address_length_ratio"] = lengths(s["address_basic"], c["address_basic"])
    f.update(overlap(*t["address"], "address", indexes["address"]))
    return f
