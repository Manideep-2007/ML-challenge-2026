"""
Decision-rule search with 2-fold cross-fitting on validation S1 entities.

Each experiment tunes a set of rule families greedily (one family at a time,
in order) on fold A and scores fold B, then the reverse; the out-of-fold mean
is the honest estimate. Final parameters are re-tuned on all validation S1.
"""

from dataclasses import asdict, replace

import numpy as np

from .decision_engine import DecisionParams, score, select

GRID = np.round(np.r_[np.arange(0.5, 0.9, 0.05), np.arange(0.9, 0.999, 0.005)], 4)
MARGINS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
RELATIVE = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
CAPS = [(99, 99), (5, 6), (4, 5), (3, 4), (2, 3), (2, 2), (1, 1)]


def _evaluate(s, p, true_counts, subset) -> float:
    return score(s, select(s, p), true_counts, subset)["macro_f05"]


def _family_candidates(family: str, p: DecisionParams, countries):
    if family == "global":
        return [replace(p, t_first=float(t), t_rest=float(t)) for t in GRID]
    if family == "first_rest":
        firsts = GRID[(GRID >= p.t_first - 0.3) & (GRID <= p.t_first + 0.02)]
        rests = GRID[GRID >= p.t_rest - 0.05]
        return [replace(p, t_first=float(a), t_rest=float(b)) for a in firsts for b in rests if b >= a]
    if family == "margin":
        return [replace(p, margin_min=m, ambiguous=a) for m in MARGINS for a in ("top1", "empty")]
    if family == "relative":
        return [replace(p, relative=r) for r in RELATIVE]
    if family == "caps":
        return [replace(p, cap_s2=a, cap_s3=b) for a, b in CAPS]
    if family == "exclusive":
        return [replace(p, exclusive=e) for e in (False, True)]
    raise ValueError(family)


def tune(s, true_counts, subset, families: list[str], countries, log=None):
    p, trace = DecisionParams(), []
    for family in families:
        if family == "country":
            country_t = {}
            for country in countries:
                firsts = GRID[np.abs(GRID - p.t_first) <= 0.1]
                rests = GRID[np.abs(GRID - p.t_rest) <= 0.03]
                options = [replace(p, country_t={**country_t, country: (float(a), float(b))})
                           for a in firsts for b in rests if b >= a]
                p = max(options, key=lambda o: _evaluate(s, o, true_counts, subset))
                country_t = p.country_t
            f = _evaluate(s, p, true_counts, subset)
        else:
            results = [(_evaluate(s, o, true_counts, subset), o) for o in _family_candidates(family, p, countries)]
            f, p = max(results, key=lambda x: x[0])
        trace.append({"family": family, "macro_f05": round(f, 6)})
        if log:
            log(f"      {family:<12} {f:.6f}")
    return p, trace


EXPERIMENTS = {
    "D1_global_threshold": ["global"],
    "D2_threshold_plus_margin": ["global", "margin"],
    "D3_threshold_plus_no_match": ["global", "first_rest"],
    "D4_threshold_plus_ambiguity": ["global", "first_rest", "relative", "caps"],
    "D5_combined": ["global", "first_rest", "relative", "margin", "caps", "exclusive", "country"],
}


def cross_fit(s, true_counts, countries, families, seed: int = 42, log=None) -> dict:
    fold = np.random.default_rng(seed).integers(0, 2, size=len(true_counts))
    held_out, fold_params = [], []
    for k in (0, 1):
        p, _ = tune(s, true_counts, fold == k, families, countries)
        fold_params.append(asdict(p))
        held_out.append(score(s, select(s, p), true_counts, fold != k)["macro_f05"])
    final, trace = tune(s, true_counts, None, families, countries, log=log)
    return {
        "out_of_fold_macro_f05": float(np.mean(held_out)),
        "out_of_fold_by_fold": [round(x, 6) for x in held_out],
        "fold_params": fold_params,
        "final_params": final,
        "final_in_sample_macro_f05": score(s, select(s, final), true_counts)["macro_f05"],
        "trace": trace,
    }
