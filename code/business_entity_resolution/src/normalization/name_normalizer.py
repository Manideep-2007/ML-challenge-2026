import pyarrow.compute as pc

from .country_rules import country_key, get_country_rule
from .tokenization import canonical_numbers, drop_tokens, map_column, unique_sorted
from .unicode_utils import (
    as_text_array,
    dominant_script,
    fold,
    nfkc,
    punctuation_to_space,
    scalar,
    whitespace_normalize,
)


def content_view(basic: list[str], countries: list[str]) -> tuple[list[str], list[bool]]:
    """Drop legal forms / connectors / web tokens; fall back to basic if nothing is left."""
    rules = {k: get_country_rule(k)["name_noise_tokens"] for k in set(countries)}
    content, fallback = [], []
    for text, key in zip(basic, countries):
        kept = drop_tokens(text, rules[key])
        empty = not kept and bool(text)
        content.append(text if empty else kept)
        fallback.append(empty)
    return content, fallback


def normalize_names(names, countries) -> dict:
    """Batch name normalization. Returns column name -> values (same length as input)."""

    raw = as_text_array(names)
    country_keys = [country_key(c) for c in as_text_array(countries).to_pylist()]

    # No placeholder rule for names: every "NA" name in train is an acronym
    # of its true match (e.g. "National Apple Company"), not a missing value.
    basic = whitespace_normalize(punctuation_to_space(fold(raw)))
    missing = pc.equal(basic, "")
    compact = pc.replace_substring(basic, pattern=" ", replacement="")

    basic_list = basic.to_pylist()
    content, fallback = content_view(basic_list, country_keys)
    script, mixed = dominant_script(basic)

    return {
        "name_raw": raw,
        "name_missing": missing,
        "name_nfkc": whitespace_normalize(nfkc(raw)),
        "name_basic": basic,
        "name_compact": compact,
        "name_fingerprint": map_column(unique_sorted, basic_list),
        "name_content": content,
        "name_content_compact": [c.replace(" ", "") for c in content],
        "name_content_fallback": fallback,
        "name_numbers": map_column(canonical_numbers, basic_list),
        "name_script": script,
        "name_script_mixed": mixed,
    }


def normalize_name(text, country=None) -> dict:
    """Single-record convenience wrapper around normalize_names (same code path)."""
    row = {k: scalar(v) for k, v in normalize_names([text], [country]).items()}
    row["name_tokens"] = row["name_basic"].split()
    row["name_unique_tokens"] = row["name_fingerprint"].split()
    row["name_content_tokens"] = row["name_content"].split()
    return row
