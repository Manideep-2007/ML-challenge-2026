"""
Stage 9 — collective (second-stage) matcher.

The pairwise model scores every (S1, candidate) pair in isolation. Records of
one business agree with each other, not only with S1: a candidate whose name
and street agree with the S1's confident matches is likely a match even when
its own S1 similarity is noisy, and a candidate that disagrees with them is
likely a sibling business. This stage re-scores the plausible pairs
(stage-1 p >= FLOOR) with:

  * probability structure inside the S1 (rank, gap to best, how many
    confident candidates, rank within the candidate's source), and
  * agreement with the S1's anchors (other candidates with p >= ANCHOR_P):
    name / address / street similarity, exact-name and house-number agreement,
  * direct street-level S1-candidate evidence (street_features).

Country-agnostic by design (France has no labels). Trained only on
validation-split pairs, whose stage-1 probabilities are out-of-sample.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from features.street_features import add_street_views, street_features

FLOOR = 0.02
ANCHOR_P = 0.9
MAX_ANCHORS = 5
TEXT_COLUMNS = ["entity_id", "country_key", "name_basic", "name_content_compact", "address_basic",
                "address_nfkc", "name_nfkc", "address_missing"]


def text_views(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[TEXT_COLUMNS].copy()
    add_street_views(frame)
    return frame.drop(columns=["address_nfkc", "name_nfkc"]).set_index("entity_id")


def _sim(a, b, scorer) -> np.ndarray:
    a, b = list(a), list(b)
    out = process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32) / 100.0
    empty = np.array([not x or not y for x, y in zip(a, b)])
    out[empty] = np.nan
    return out


def probability_structure(pairs: pd.DataFrame) -> pd.DataFrame:
    """pairs sorted by (s1, -p); returns per-pair structure features."""
    p = pairs["p"].to_numpy(np.float32)
    g = pairs.groupby("s1", sort=False)["p"]
    f = pd.DataFrame(index=pairs.index)
    f["p"] = p
    f["logit_p"] = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    f["rank"] = g.cumcount().to_numpy()
    best = g.transform("max").to_numpy()
    f["best"] = best
    f["p_over_best"] = p / np.maximum(best, 1e-6)
    f["gap_best"] = best - p
    second = pairs["p"].where(f["rank"] == 1).groupby(pairs["s1"], sort=False).transform("max").fillna(0.0).to_numpy()
    f["second"] = second
    f["lead_over_second"] = np.where(f["rank"].to_numpy() == 0, p - second, p - best)
    for t in (0.5, 0.8, 0.95):
        f[f"n_ge_{t}"] = (pairs["p"] >= t).groupby(pairs["s1"], sort=False).transform("sum").to_numpy()
    f["n_kept"] = g.transform("size").to_numpy()
    is_s2 = pairs["cand"].str.startswith("S2-").to_numpy()
    f["is_s2"] = is_s2.astype(np.float32)
    src = pairs["s1"] + np.where(is_s2, "|2", "|3")
    f["source_rank"] = pairs.groupby(src, sort=False).cumcount().to_numpy()
    f["source_n_ge_0.8"] = (pairs["p"] >= 0.8).groupby(src, sort=False).transform("sum").to_numpy()
    return f


def anchor_agreement(pairs: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    anchors = pairs[pairs["p"] >= ANCHOR_P][["s1", "cand", "p"]]
    anchors = anchors[anchors.groupby("s1", sort=False).cumcount() < MAX_ANCHORS]
    link = pairs[["s1", "cand"]].reset_index().merge(anchors, on="s1", suffixes=("", "_a"))
    link = link[link["cand"] != link["cand_a"]]
    a = ref.reindex(link["cand"].to_numpy())
    b = ref.reindex(link["cand_a"].to_numpy())
    sims = pd.DataFrame({
        "idx": link["index"].to_numpy(),
        "anc_p": link["p"].to_numpy(),
        "name_tsr": _sim(a["name_basic"], b["name_basic"], fuzz.token_set_ratio),
        "addr_tsr": _sim(a["address_basic"], b["address_basic"], fuzz.token_set_ratio),
        "street": _sim(a["street_core"], b["street_core"], fuzz.ratio),
        "name_exact": (a["name_content_compact"].to_numpy() == b["name_content_compact"].to_numpy())
                      & (a["name_content_compact"].to_numpy() != ""),
        "number_equal": (a["street_number"].to_numpy() == b["street_number"].to_numpy())
                        & (a["street_number"].to_numpy() != ""),
    })
    agg = sims.groupby("idx").agg(
        n_anchors=("anc_p", "size"), anc_p_max=("anc_p", "max"),
        anc_name_tsr_max=("name_tsr", "max"), anc_name_tsr_mean=("name_tsr", "mean"),
        anc_addr_tsr_max=("addr_tsr", "max"), anc_addr_tsr_mean=("addr_tsr", "mean"),
        anc_street_max=("street", "max"), anc_street_min=("street", "min"),
        anc_name_exact_any=("name_exact", "max"), anc_number_equal_any=("number_equal", "max"),
        anc_number_equal_share=("number_equal", "mean"),
    )
    out = agg.reindex(pairs.index)
    out["n_anchors"] = out["n_anchors"].fillna(0)
    return out.astype(np.float32)


def direct_evidence(pairs: pd.DataFrame, s1: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    a = s1.reindex(pairs["s1"].to_numpy())
    b = ref.reindex(pairs["cand"].to_numpy())
    f = pd.DataFrame(index=pairs.index)
    f["d_name_tsr"] = _sim(a["name_basic"], b["name_basic"], fuzz.token_set_ratio)
    f["d_addr_tsr"] = _sim(a["address_basic"], b["address_basic"], fuzz.token_set_ratio)
    f["cand_address_missing"] = b["address_missing"].to_numpy().astype(np.float32)
    cols = ["street_core", "street_number", "legal_dotted"]
    street = street_features({c: a[c].to_numpy() for c in cols}, {c: b[c].to_numpy() for c in cols})
    for k, v in street.items():
        f[k] = v
    return f.astype(np.float32)


def collective_features(pairs: pd.DataFrame, s1: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """pairs: columns s1, cand, p (any order). Returns pairs sorted by (s1, -p) with features."""
    pairs = pairs[pairs["p"] >= FLOOR].sort_values(["s1", "p"], ascending=[True, False]).reset_index(drop=True)
    parts = [pairs, probability_structure(pairs).drop(columns=["p"]), anchor_agreement(pairs, ref),
             direct_evidence(pairs, s1, ref)]
    return pd.concat(parts, axis=1)


def feature_names(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("s1", "cand", "label", "country", "fold")]


# ------------------------------------------------------------------ E007: universe name statistics
def universe_counts(s1_paths, ref_paths) -> dict:
    """
    How common a name is in the whole record universe of the split (no labels):
    S1 records per (country, name), S1 per (country, name, house number) and S2/S3
    records per (country, name). Validation uses the train universe, test the test
    universe -- the same kind of statistic on both sides.
    """
    import pyarrow.parquet as pq
    from features.street_features import first_number
    cols = ["country_key", "name_content_compact", "address_nfkc"]
    s1 = pd.concat([pq.read_table(p, columns=cols).to_pandas() for p in s1_paths], ignore_index=True)
    s1["number"] = [first_number(a) for a in s1["address_nfkc"]]
    ref = pd.concat([pq.read_table(p, columns=cols[:2]).to_pandas() for p in ref_paths], ignore_index=True)
    return {
        "s1_name": s1.groupby(["country_key", "name_content_compact"]).size(),
        "s1_name_number": s1.groupby(["country_key", "name_content_compact", "number"]).size(),
        "ref_name": ref.groupby(["country_key", "name_content_compact"]).size(),
    }


def universe_features(pairs: pd.DataFrame, s1: pd.DataFrame, ref: pd.DataFrame, counts: dict) -> pd.DataFrame:
    a = s1.reindex(pairs["s1"].to_numpy())
    b = ref.reindex(pairs["cand"].to_numpy())

    def look(table, keys):
        idx = pd.MultiIndex.from_arrays(keys)
        return table.reindex(idx).fillna(0).to_numpy(np.float32)

    ca, na, nb = a["country_key"].to_numpy(), a["name_content_compact"].to_numpy(), b["name_content_compact"].to_numpy()
    f = pd.DataFrame(index=pairs.index)
    f["u_s1_name_count"] = look(counts["s1_name"], [ca, na])
    f["u_s1_name_number_count"] = look(counts["s1_name_number"], [ca, na, a["street_number"].to_numpy()])
    f["u_ref_name_count"] = look(counts["ref_name"], [ca, na])
    f["u_cand_name_s1_count"] = look(counts["s1_name"], [ca, nb])     # S1 carrying the candidate's name
    f["u_refs_per_s1_name"] = f["u_ref_name_count"] / np.maximum(f["u_s1_name_count"], 1)
    empty = na == ""
    f.loc[empty, ["u_s1_name_count", "u_s1_name_number_count", "u_ref_name_count", "u_refs_per_s1_name"]] = np.nan
    f.loc[nb == "", "u_cand_name_s1_count"] = np.nan
    return f


# ------------------------------------------------------------------ E008: transliterated names
_VOWELS = str.maketrans("", "", "aeiouy")


def transliterated(text: str) -> str:
    """Latin rendering of any script (anyascii, ISC licence), casefolded alphanumerics."""
    from anyascii import anyascii
    from features.street_features import _NON_ALNUM
    return " ".join(_NON_ALNUM.sub(" ", anyascii(text or "").casefold()).split())


def skeleton(text: str) -> str:
    """Consonant skeleton: 'sri bijnes praivet' -> 'srbjnsprvt' (vowel spelling varies across scripts)."""
    out = text.replace(" ", "").translate(_VOWELS)
    return "".join(c for i, c in enumerate(out) if i == 0 or c != out[i - 1])


def transliteration_features(pairs: pd.DataFrame, s1_raw_names: pd.Series, ref_raw_names: pd.Series) -> pd.DataFrame:
    """Name similarity after transliterating the candidate; NaN when the candidate name is Latin."""
    a = s1_raw_names.reindex(pairs["s1"].to_numpy()).fillna("").to_numpy()
    b = ref_raw_names.reindex(pairs["cand"].to_numpy()).fillna("").to_numpy()
    non_latin = np.array([any(ord(ch) > 0x24F for ch in x) for x in b])
    ta = np.array([transliterated(x) for x in a], dtype=object)
    tb = np.array([transliterated(x) if nl else "" for x, nl in zip(b, non_latin)], dtype=object)
    sa = np.array([skeleton(x) for x in ta], dtype=object)
    sb = np.array([skeleton(x) for x in tb], dtype=object)
    f = pd.DataFrame(index=pairs.index)
    f["t_name_ratio"] = _sim(ta, tb, fuzz.ratio)
    f["t_name_token_set_ratio"] = _sim(ta, tb, fuzz.token_set_ratio)
    f["t_skeleton_ratio"] = _sim(sa, sb, fuzz.ratio)
    f["t_skeleton_partial_ratio"] = _sim(sa, sb, fuzz.partial_ratio)
    for c in f.columns:
        f.loc[~non_latin, c] = np.nan
    f["t_cand_non_latin"] = non_latin.astype(np.float32)
    return f.astype(np.float32)
