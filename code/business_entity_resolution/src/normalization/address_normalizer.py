"""
Address views. Deliberately NO abbreviation expansion (Road -> Rd etc.) in
any view yet: Stage 1 showed wrong expansions in the data itself
("Street" -> "SAINT"), so equivalences are left to similarity features.
Postal codes are not extracted: Stage 1 found 5-digit US numbers are
house numbers (86% in the first component) and PIN/postcodes appear in
<0.6% of India/France addresses. The numbers view carries that signal.
"""

import pyarrow as pa
import pyarrow.compute as pc

from .country_rules import PLACEHOLDERS
from .tokenization import canonical_numbers, map_column, unique_sorted
from .unicode_utils import (
    as_text_array,
    dominant_script,
    fold,
    nfkc,
    punctuation_to_space,
    scalar,
    whitespace_normalize,
)


def drop_placeholder_components(text: str) -> tuple[str, bool]:
    """ "3800, null, clearfield, ut" -> ("3800, clearfield, ut", True) """
    parts = text.split(",")
    kept = [p for p in parts if p.strip() not in PLACEHOLDERS]
    if len(kept) == len(parts):
        return text, False
    return ",".join(kept), True


def normalize_addresses(addresses) -> dict:
    """Batch address normalization. Returns column name -> values."""

    raw = as_text_array(addresses)
    folded = whitespace_normalize(fold(raw)).to_pylist()

    cleaned, removed = [], []
    for text in folded:
        kept, dropped = drop_placeholder_components(text)
        cleaned.append(kept)
        removed.append(dropped)

    basic = whitespace_normalize(punctuation_to_space(pa.array(cleaned, type=pa.string())))
    compact = pc.replace_substring(basic, pattern=" ", replacement="")
    basic_list = basic.to_pylist()
    script, mixed = dominant_script(basic)

    return {
        "address_raw": raw,
        "address_missing": pc.equal(basic, ""),
        "address_nfkc": whitespace_normalize(nfkc(raw)),
        "address_basic": basic,
        "address_compact": compact,
        "address_fingerprint": map_column(unique_sorted, basic_list),
        "address_numbers": map_column(canonical_numbers, basic_list),
        "address_placeholder_removed": removed,
        "address_script": script,
        "address_script_mixed": mixed,
    }


def normalize_address(text) -> dict:
    """Single-record convenience wrapper around normalize_addresses (same code path)."""
    out = {k: scalar(v) for k, v in normalize_addresses([text]).items()}
    out["address_tokens"] = out["address_basic"].split()
    out["address_unique_tokens"] = out["address_fingerprint"].split()
    out["address_number_list"] = out["address_numbers"].split()
    return out
