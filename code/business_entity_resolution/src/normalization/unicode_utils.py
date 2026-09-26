"""
Unicode primitives on Arrow string arrays.

Unicode normalization (NFKC/NFC, casefold, diacritics) uses Python's
unicodedata: pyarrow 25's utf8_normalize does not compose (NFC of "e"+U+0301
stays decomposed, as do Tamil/Bengali/Telugu two-part vowels). Arrow is used
only for the regex steps.

Script safety: names/addresses include Devanagari, Malayalam, Bengali, etc.
Only Latin combining diacritics (U+0300-U+036F) are stripped, and "word"
characters are letters/numbers/marks of ANY script, so Indic vowel signs,
viramas and nuktas survive.
"""

import re
import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc


LATIN_COMBINING_MARKS = re.compile(r"[̀-ͯ]")
NON_WORD_RUN = r"[^\p{L}\p{N}\p{M}]+"
WHITESPACE_RUN = r"\s+"

SCRIPTS = [
    "Latin", "Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya",
    "Tamil", "Telugu", "Kannada", "Malayalam", "Arabic",
]


def as_text_array(values) -> pa.ChunkedArray:
    """Any sequence / Series / Arrow array -> Arrow strings, nulls as ""."""
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        arr = values
    else:
        arr = pa.array(
            [None if pd.isna(v) else str(v) for v in values],
            type=pa.string(),
        )
    if not pa.types.is_string(arr.type) and not pa.types.is_large_string(arr.type):
        arr = pc.cast(arr, pa.string())
    return pc.fill_null(arr, "")


def scalar(values):
    """First element of an Arrow array / numpy array / list as a Python value."""
    value = values[0]
    if hasattr(value, "as_py"):
        return value.as_py()
    if hasattr(value, "item"):
        return value.item()
    return value


def _map(func, arr) -> pa.Array:
    return pa.array([func(v) for v in arr.to_pylist()], type=pa.string())


def fold_text(text: str) -> str:
    """NFKC -> casefold -> strip Latin diacritics -> NFC."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = LATIN_COMBINING_MARKS.sub("", unicodedata.normalize("NFKD", text))
    return unicodedata.normalize("NFC", text)


def nfkc(arr):
    return _map(lambda v: unicodedata.normalize("NFKC", v), arr)


def fold(arr):
    """Keeps punctuation; see fold_text."""
    return _map(fold_text, arr)


def whitespace_normalize(arr):
    return pc.utf8_trim_whitespace(
        pc.replace_substring_regex(arr, pattern=WHITESPACE_RUN, replacement=" ")
    )


def punctuation_to_space(arr):
    return pc.replace_substring_regex(arr, pattern=NON_WORD_RUN, replacement=" ")


def dominant_script(arr) -> tuple[np.ndarray, np.ndarray]:
    """
    Per value: the script with the most letters ("none" if no letters from
    SCRIPTS, "other" if letters exist only in other scripts), and whether
    two or more scripts are present.
    """
    counts = np.column_stack([
        pc.count_substring_regex(arr, pattern=rf"\p{{{s}}}").to_numpy(zero_copy_only=False)
        for s in SCRIPTS
    ])
    letters = pc.count_substring_regex(arr, pattern=r"\p{L}").to_numpy(zero_copy_only=False)

    names = np.array(SCRIPTS + ["other", "none"], dtype=object)
    index = counts.argmax(axis=1)
    index = np.where(counts.max(axis=1) > 0, index, np.where(letters > 0, len(SCRIPTS), len(SCRIPTS) + 1))

    mixed = (counts > 0).sum(axis=1) >= 2
    return names[index], mixed
