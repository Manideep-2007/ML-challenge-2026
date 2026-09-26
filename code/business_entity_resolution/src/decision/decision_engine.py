"""
Per-entity decision engine: ranked pair probabilities -> per-S1 match lists.

Rules (each justified by the data audit, all tuned on validation):
  t_first      threshold for an S1's best candidate (also the no-match threshold:
               if the best candidate fails it, the S1 gets an empty prediction)
  t_rest       threshold for the other candidates (F0.5 punishes extra false merges)
  relative     keep a non-best candidate only if p >= relative * best p (probability tails)
  margin_min   S1 whose top-vs-second margin is below this is "ambiguous" ...
  ambiguous    ... and gets only its top candidate ("top1") or nothing ("empty")
  cap_s2/s3    max matches per source (ground truth: <= 5 S2, <= 6 S3 per S1)
  exclusive    each S2/S3 record goes to at most one S1 (true in the ground truth)
  country_t    optional per-country (t_first, t_rest); unseen countries use the global values
"""

from dataclasses import dataclass, field

import numpy as np

from .ranking import Scored


@dataclass
class DecisionParams:
    t_first: float = 0.5
    t_rest: float = 0.5
    relative: float = 0.0
    margin_min: float = 0.0
    ambiguous: str = "top1"
    cap_s2: int = 99
    cap_s3: int = 99
    exclusive: bool = False
    country_t: dict = field(default_factory=dict)   # country -> (t_first, t_rest)


def select(s: Scored, p: DecisionParams) -> np.ndarray:
    """Boolean mask over s: pairs predicted as matches."""
    t_first = np.full(len(s.prob), p.t_first, dtype=np.float32)
    t_rest = np.full(len(s.prob), p.t_rest, dtype=np.float32)
    if p.country_t:
        pair_country = s.country[s.code]
        for country, (tf, tr) in p.country_t.items():
            in_country = pair_country == country
            t_first[in_country] = tf
            t_rest[in_country] = tr

    first = s.rank == 0
    keep = np.where(first, s.prob >= t_first, (s.prob >= t_rest) & (s.prob >= p.relative * s.best))

    if p.margin_min > 0:
        ambiguous = (s.best - s.second) < p.margin_min
        if p.ambiguous == "top1":
            keep &= ~ambiguous | first
        else:
            keep &= ~ambiguous

    keep &= np.where(s.is_s2, s.source_rank < p.cap_s2, s.source_rank < p.cap_s3)

    if p.exclusive and keep.any():
        idx = np.flatnonzero(keep)
        order = idx[np.lexsort((-s.prob[idx], s.cand[idx]))]
        winner = np.r_[True, s.cand[order][1:] != s.cand[order][:-1]]
        keep[:] = False
        keep[order[winner]] = True
    return keep


def entity_outcomes(s: Scored, keep: np.ndarray, true_counts: np.ndarray) -> dict:
    """Per-S1 tp / predicted / precision / recall / F0.5 (exact Stage 2 definition)."""
    n = len(true_counts)
    n_pred = np.bincount(s.code, weights=keep, minlength=n)
    tp = np.bincount(s.code, weights=keep & (s.label == 1), minlength=n)
    precision = np.divide(tp, n_pred, out=np.zeros(n), where=n_pred > 0)
    recall = np.divide(tp, true_counts, out=np.zeros(n), where=true_counts > 0)
    denom = 0.25 * precision + recall
    f = np.divide(1.25 * precision * recall, denom, out=np.zeros(n), where=denom > 0)
    empty_truth = true_counts == 0
    f = np.where(empty_truth, (n_pred == 0).astype(float), f)
    precision = np.where(n_pred == 0, np.where(empty_truth, 1.0, 0.0), precision)
    recall = np.where(empty_truth, (n_pred == 0).astype(float), recall)
    return {"n_pred": n_pred, "tp": tp, "precision": precision, "recall": recall, "f05": f}


def score(s: Scored, keep: np.ndarray, true_counts: np.ndarray, subset: np.ndarray | None = None) -> dict:
    o = entity_outcomes(s, keep, true_counts)
    mask = np.ones(len(true_counts), bool) if subset is None else subset
    empty_truth = true_counts == 0
    return {
        "macro_f05": float(o["f05"][mask].mean()),
        "macro_precision": float(o["precision"][mask].mean()),
        "macro_recall": float(o["recall"][mask].mean()),
        "empty_truth_f05": float(o["f05"][mask & empty_truth].mean()) if (mask & empty_truth).any() else float("nan"),
        "matched_f05": float(o["f05"][mask & ~empty_truth].mean()),
        "mean_predicted": float(o["n_pred"][mask].mean()),
        "per_entity": o["f05"],
    }
