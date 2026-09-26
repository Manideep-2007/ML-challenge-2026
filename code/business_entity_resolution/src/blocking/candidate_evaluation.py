"""
Candidate-set evaluation against ground truth.

Besides recall and size statistics, reports the F0.5 CEILING: the macro F0.5
a perfect matcher would reach if it predicted exactly the true matches
present in the candidate set (precision 1, recall = retrieved share).
Singletons score 1.0 (a perfect matcher predicts nothing for them).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Truth:
    query_idx: np.ndarray      # per true pair
    ref_idx: np.ndarray
    keys: np.ndarray           # query_idx * n_ref + ref_idx
    true_counts: np.ndarray    # per query


def build_truth(ground_truth: dict[str, list[str]], query_ids: np.ndarray, ref_ids: np.ndarray) -> Truth:
    q_pos = pd.Series(np.arange(len(query_ids)), index=query_ids)
    r_pos = pd.Series(np.arange(len(ref_ids)), index=ref_ids)
    s1, target = [], []
    for s1_id, matches in ground_truth.items():
        for m in matches:
            s1.append(s1_id)
            target.append(m)
    q = q_pos.reindex(s1).to_numpy()
    r = r_pos.reindex(target).to_numpy()
    if np.isnan(q.astype(float)).any() or np.isnan(r.astype(float)).any():
        raise ValueError("Ground-truth IDs missing from query or reference records")
    q, r = q.astype(np.int64), r.astype(np.int64)
    return Truth(q, r, q * len(ref_ids) + r, np.bincount(q, minlength=len(query_ids)))


def found_mask(truth: Truth, keys: np.ndarray) -> np.ndarray:
    """keys must be sorted unique pair keys."""
    if len(keys) == 0:
        return np.zeros(len(truth.keys), dtype=bool)
    pos = np.searchsorted(keys, truth.keys)
    pos = np.minimum(pos, len(keys) - 1)
    return keys[pos] == truth.keys


def evaluate_keys(keys: np.ndarray, truth: Truth, n_queries: int, n_ref: int, countries: np.ndarray | None = None) -> dict:
    counts = np.bincount(keys // n_ref, minlength=n_queries)
    found = found_mask(truth, keys)
    found_counts = np.bincount(truth.query_idx[found], minlength=n_queries)

    matched = truth.true_counts > 0
    recall = np.divide(found_counts, truth.true_counts, out=np.zeros(n_queries), where=matched)
    ceiling = np.where(matched, np.divide(1.25 * recall, 0.25 + recall, out=np.zeros(n_queries), where=recall > 0), 1.0)

    result = {
        "pairs": int(len(keys)),
        "pair_recall": round(float(found.mean()), 6),
        "entity_all_found": round(float((found_counts[matched] == truth.true_counts[matched]).mean()), 6),
        "entity_any_found": round(float((found_counts[matched] > 0).mean()), 6),
        "f05_ceiling": round(float(ceiling.mean()), 6),
        "queries_without_candidates": int((counts == 0).sum()),
        "cand_min": int(counts.min()),
        "cand_median": float(np.median(counts)),
        "cand_mean": round(float(counts.mean()), 2),
        "cand_p90": float(np.percentile(counts, 90)),
        "cand_p95": float(np.percentile(counts, 95)),
        "cand_p99": float(np.percentile(counts, 99)),
        "cand_max": int(counts.max()),
    }
    if countries is not None:
        for country in np.unique(countries):
            in_country = countries == country
            pair_in_country = in_country[truth.query_idx]
            result[f"pair_recall_{country}"] = round(float(found[pair_in_country].mean()), 6)
            result[f"f05_ceiling_{country}"] = round(float(ceiling[in_country].mean()), 6)
            result[f"cand_mean_{country}"] = round(float(counts[in_country].mean()), 2)
    return result


def size_distribution(keys: np.ndarray, n_queries: int, n_ref: int) -> pd.DataFrame:
    counts = np.bincount(keys // n_ref, minlength=n_queries)
    edges = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, np.inf]
    labels = ["0", "1", "2-4", "5-9", "10-19", "20-49", "50-99", "100-199",
              "200-499", "500-999", "1000-1999", "2000-4999", "5000+"]
    bins = pd.cut(counts, bins=edges, right=False, labels=labels)
    table = bins.value_counts().reindex(labels).rename_axis("candidates").reset_index(name="queries")
    table["pct"] = (100 * table["queries"] / n_queries).round(3)
    return table


def miss_reasons(missed_q: np.ndarray, missed_r: np.ndarray, queries: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Why a true pair shares no strong key: per-pair diagnostic flags."""
    rows = []
    q_frame = queries.iloc[missed_q].reset_index(drop=True)
    r_frame = reference.iloc[missed_r].reset_index(drop=True)
    for q, r in zip(q_frame.itertuples(index=False), r_frame.itertuples(index=False)):
        q_name, r_name = set(q.name_content.split()), set(r.name_content.split())
        q_addr, r_addr = set(q.address_basic.split()), set(r.address_basic.split())
        q_num, r_num = set(q.address_numbers.split()), set(r.address_numbers.split())
        rows.append({
            "target_name_script": r.name_script,
            "target_address_missing": r.address_basic == "",
            "shares_name_token": bool(q_name & r_name),
            "shares_address_token": bool(q_addr & r_addr),
            "shares_address_number": bool(q_num & r_num),
        })
    return pd.DataFrame(rows)
