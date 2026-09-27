"""
Stage 8 / E004 (feature set v3): street-level address evidence.

Test error analysis (France, unlabeled): sibling businesses share name, house
number, city and region and differ only in legal form and street name
("Pessac Amicale SAS, 31 Rue de Flandre" vs "Pessac Amicale SARL, 31 Rue
Socrate"). Whole-address token overlap stays high for such pairs because
city / region tokens dominate short addresses, so the model cannot see that
the street itself disagrees. These features isolate the street:

  street core = tokens of the comma component that holds the first house
  number (else the component with a street-type word), minus numbers,
  street-type words and particles ("31 Rue de Flandre" -> "flandre").

  street_core_ratio            edit similarity of the two street cores
  street_core_token_set_ratio  order-invariant token similarity
  street_core_conflict         both cores present and token_set_ratio < 0.6
  street_same_number_diff_street  first house number equal but streets conflict
  street_house_match           first house number equal and streets agree
  legal_conflict_dotted        legal-form conflict that also reads dotted forms
                               ("S.A.S", "S.A.R.L.") collapsed from single letters
"""

import re
import unicodedata

import numpy as np

from normalization.country_rules import get_country_rule

from .string_features import pairwise

STREET_TYPES = {
    # French
    "rue", "r", "avenue", "av", "ave", "bd", "boulevard", "blvd", "cours", "crs", "allee", "all",
    "quai", "q", "impasse", "imp", "place", "pl", "chemin", "ch", "che", "passage", "pass", "square",
    "sq", "route", "rte", "voie", "residence", "res", "lotissement", "lot", "faubourg", "fbg", "cite",
    "esplanade", "promenade", "parvis", "rond", "point", "sentier", "hameau", "zone", "za", "zi",
    # English
    "road", "rd", "street", "st", "str", "lane", "ln", "drive", "dr", "court", "ct", "circle", "cir",
    "way", "parkway", "pkwy", "highway", "hwy", "terrace", "ter", "trail", "trl", "boulevard", "plaza",
    "plz", "alley", "aly", "loop", "path", "pike", "run", "row", "walk", "crescent", "cres", "close",
    "unit", "suite", "ste", "apt", "no", "n", "hno", "sno", "h", "house", "plot", "p", "floor", "shop", "bis", "ter",
    "po", "box", "door", "flat", "building", "bldg",
    # Indian
    "marg", "nagar", "main", "cross", "colony", "sector", "block", "phase", "layout", "near", "opp",
}
PARTICLES = {"de", "du", "des", "la", "le", "les", "l", "d", "of", "the", "a", "b", "c", "et", "and"}
_NUMBER = re.compile(r"\d")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def street_core(address: str) -> str:
    """Street name of a raw address, '' when no street component is found."""
    if not address:
        return ""
    components = [_NON_ALNUM.sub(" ", fold(part)).split() for part in address.split(",")]
    chosen = next((c for c in components if any(_NUMBER.search(t) for t in c)), None)
    if chosen is None:
        chosen = next((c for c in components if any(t in STREET_TYPES for t in c)), None)
    if chosen is None:
        return ""
    core = [t for t in chosen if not _NUMBER.search(t) and t not in STREET_TYPES and t not in PARTICLES]
    return " ".join(core)


def first_number(address: str) -> str:
    for token in _NON_ALNUM.sub(" ", fold(address or "")).split():
        digits = re.match(r"\d+", token)
        if digits:
            return digits.group(0).lstrip("0") or "0"
    return ""


def dotted_legal(name_raw: str, country: str) -> str:
    """Legal-form tokens, also reading dotted forms: 'S.A.R.L.' -> 'sarl'."""
    forms = get_country_rule(country)["legal_forms"]
    tokens = _NON_ALNUM.sub(" ", fold(name_raw or "")).split()
    out, run = set(), ""
    for t in tokens + [""]:
        if len(t) == 1:
            run += t
            continue
        if len(run) >= 2 and run in forms:
            out.add(run)
        run = ""
        if t in forms:
            out.add(t)
    return " ".join(sorted(out))


def add_street_views(frame) -> None:
    """Per-record views for feature set v3 (needs address_nfkc, name_nfkc, country_key)."""
    import pandas as pd
    frame["street_core"] = pd.Series([street_core(a) for a in frame["address_nfkc"]], dtype="str", index=frame.index)
    frame["street_number"] = pd.Series([first_number(a) for a in frame["address_nfkc"]], dtype="str", index=frame.index)
    frame["legal_dotted"] = pd.Series([dotted_legal(n, c) for n, c in zip(frame["name_nfkc"], frame["country_key"])],
                                      dtype="str", index=frame.index)


def street_features(s: dict, c: dict) -> dict:
    f = {}
    f["street_core_ratio"] = pairwise(s["street_core"], c["street_core"], "ratio")
    tsr = pairwise(s["street_core"], c["street_core"], "token_set_ratio")
    f["street_core_token_set_ratio"] = tsr
    both = ~np.isnan(tsr)
    conflict = np.where(both, (tsr < 0.6).astype(np.float32), np.nan).astype(np.float32)
    f["street_core_conflict"] = conflict
    number_equal = np.array([a != "" and a == b for a, b in zip(s["street_number"], c["street_number"])])
    f["street_same_number_diff_street"] = np.where(both, (number_equal & (conflict == 1)).astype(np.float32), np.nan)
    f["street_house_match"] = np.where(both, (number_equal & (tsr >= 0.85)).astype(np.float32), np.nan)
    ls, lc = s["legal_dotted"], c["legal_dotted"]
    both_legal = np.array([bool(a) and bool(b) for a, b in zip(ls, lc)])
    disjoint = np.array([not (set(a.split()) & set(b.split())) for a, b in zip(ls, lc)])
    f["legal_conflict_dotted"] = np.where(both_legal, disjoint.astype(np.float32), np.nan).astype(np.float32)
    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}
