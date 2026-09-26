"""
Token helpers on already-normalized views (single-space separated tokens).
Plain Python: set/sort/filter work does not vectorize in Arrow.
"""

import re


DIGIT_RUN = re.compile(r"\d+")  # str pattern: also matches non-ASCII digits (e.g. Devanagari)


def tokens(text: str) -> list[str]:
    return text.split()


def unique_sorted(text: str) -> str:
    """Order-invariant fingerprint: "abc pvt ltd abc" -> "abc ltd pvt"."""
    return " ".join(sorted(set(text.split())))


def drop_tokens(text: str, drop: frozenset) -> str:
    return " ".join(t for t in text.split() if t not in drop)


def canonical_numbers(text: str) -> str:
    """
    Digit runs as integers, order kept: "00930 Park, Unit 7" -> "930 7".
    int() strips zero padding (seen in S2/S3) and maps non-ASCII digits to ASCII.
    """
    return " ".join(str(int(d)) for d in DIGIT_RUN.findall(text))


def map_column(func, values, *args) -> list[str]:
    return [func(v, *args) for v in values]
