"""
Learned S2/S3 -> S1 token translation (train ground truth only).

S2/S3 use tokens that S1 almost never uses for the same thing: state names
vs codes ("arizona" vs "az"), native-script words ("महाराष्ट्र", "प्राइवेट",
"फॉर्च्यून"), leetspeak ("5ervices"), variant spellings ("keralam"). For each
such "foreign" token t, the S1 token s it most often co-occurs with in true
pairs (where s itself is absent from the S2/S3 side) is learned as t -> s if
P(s | t) is high. Uses only challenge training labels; the evaluated S1 set is
always excluded from learning.
"""

from collections import Counter

import numpy as np


def document_frequency(texts) -> Counter:
    df = Counter()
    for text in texts:
        df.update(set(text.split()))
    return df


def learn_translation(
    s1_texts: list[str],
    target_texts: list[str],
    s1_df: Counter,
    n_s1: int,
    target_df: Counter,
    n_target: int,
    min_count: int = 5,
    min_probability: float = 0.5,
    foreign_ratio: float = 0.05,
) -> dict[str, tuple[str, float, int]]:
    """
    s1_texts / target_texts: aligned true pairs (one field). Returns
    token -> (translation, P(translation | token), support).
    """
    def is_foreign(token: str) -> bool:
        if token.isdigit() or target_df[token] < min_count:
            return False
        return s1_df[token] / n_s1 < foreign_ratio * target_df[token] / n_target

    foreign_cache: dict[str, bool] = {}
    token_count: Counter = Counter()
    co_count: Counter = Counter()

    for s1_text, target_text in zip(s1_texts, target_texts):
        target_tokens = set(target_text.split())
        foreign = []
        for t in target_tokens:
            flag = foreign_cache.get(t)
            if flag is None:
                flag = foreign_cache[t] = is_foreign(t)
            if flag:
                foreign.append(t)
        if not foreign:
            continue
        missing = set(s1_text.split()) - target_tokens
        for t in foreign:
            token_count[t] += 1
            for s in missing:
                co_count[(t, s)] += 1

    best: dict[str, tuple[str, int]] = {}
    for (t, s), n in co_count.items():
        if n > best.get(t, ("", 0))[1]:
            best[t] = (s, n)

    return {
        t: (s, round(n / token_count[t], 4), token_count[t])
        for t, (s, n) in best.items()
        if token_count[t] >= min_count and n / token_count[t] >= min_probability
    }


def translator(dictionary: dict[str, tuple[str, float, int]]):
    """Append translations to the text (originals are kept)."""
    mapping = {t: s for t, (s, _, _) in dictionary.items()}

    def translate(text: str) -> str:
        extra = [mapping[t] for t in text.split() if t in mapping]
        return text if not extra else text + " " + " ".join(extra)

    return translate


def sample_positions(n: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(size, n), replace=False))
