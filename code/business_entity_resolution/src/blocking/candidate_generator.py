"""
Runs blocking channels per country, measures each channel on its own, and
merges them into one CandidateUnion with provenance.
"""

import ctypes
import sys
import time

import numpy as np

from .candidate_evaluation import Truth, evaluate_keys
from .candidate_union import CandidateUnion
from .indexes import Records, country_partitions


def peak_memory_mb() -> float:
    """Peak working set of this process (Windows) or max RSS (Unix)."""
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return round(counters.PeakWorkingSetSize / 2**20)
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


K_SWEEP = [5, 10, 25, 50, 100, 200]


def run_channel(name, channel, queries: Records, reference: Records, union: CandidateUnion):
    start = time.time()

    q_frame = queries.columns(channel.columns)
    r_frame = reference.columns(channel.columns)

    q_parts, r_parts, rank_parts, stats = [], [], [], {}
    for country, q_pos, r_pos in country_partitions(queries, reference):
        q, r, channel_stats = channel(
            q_frame.iloc[q_pos].reset_index(drop=True),
            r_frame.iloc[r_pos].reset_index(drop=True),
        )
        q_parts.append(q_pos[q])
        r_parts.append(r_pos[r])
        rank = channel_stats.pop("rank", None)
        if rank is not None:
            rank_parts.append(rank)
        stats[country] = channel_stats
    del q_frame, r_frame

    q_all, r_all = np.concatenate(q_parts), np.concatenate(r_parts)
    ranked = None
    if len(rank_parts) == len(q_parts):
        ranked = (q_all, r_all, np.concatenate(rank_parts))
    keys = union.pair_keys(q_all, r_all)
    run = {"seconds": round(time.time() - start, 1), "process_peak_mb": peak_memory_mb(), "stats": stats}
    return keys, ranked, run


def recall_at_k(ranked, union: CandidateUnion, truth: Truth, n_queries: int, countries) -> list[dict]:
    """Recall / ceiling / size if the channel kept only its top K per query."""
    q, r, rank = ranked
    rows = []
    for k in K_SWEEP:
        if k > int(rank.max()) + 1:
            break
        keep = rank < k
        metrics = evaluate_keys(union.pair_keys(q[keep], r[keep]), truth, n_queries, union.n_ref, countries)
        rows.append({"k": k, **{m: metrics[m] for m in ("pair_recall", "f05_ceiling", "cand_mean", "cand_p95")}})
    return rows


def union_at_k(union: CandidateUnion, ranked_store: dict, truth: Truth, n_queries: int, countries) -> list[dict]:
    """Union recall when every ranked channel keeps only its top K (unranked channels kept whole)."""
    unranked = [n for n in union.channel_names if n not in ranked_store]
    base = union.subset(unranked) if unranked else np.empty(0, np.int64)
    max_rank = max(int(r[2].max()) for r in ranked_store.values()) + 1
    rows = []
    for k in K_SWEEP:
        if k > max_rank:
            break
        parts = [base] + [union.pair_keys(q[rank < k], r[rank < k]) for q, r, rank in ranked_store.values()]
        keys = np.unique(np.concatenate(parts))
        metrics = evaluate_keys(keys, truth, n_queries, union.n_ref, countries)
        rows.append({"k": k, **{m: v for m, v in metrics.items() if not m.startswith("cand_") or m in
                                ("cand_mean", "cand_median", "cand_p95", "cand_p99", "cand_max")}})
    return rows


def generate_candidates(channels: dict, queries: Records, reference: Records, truth: Truth | None = None,
                        log=print, keep_ranks: bool = False):
    union = CandidateUnion(len(reference), list(channels))
    channel_metrics = {}
    ranked_store = {}

    for name, channel in channels.items():
        keys, ranked, run = run_channel(name, channel, queries, reference, union)
        metrics = {"channel": name, **{k: v for k, v in run.items() if k != "stats"}}
        if truth is not None:
            metrics.update(evaluate_keys(keys, truth, len(queries), len(reference), queries.country))
            if ranked is not None:
                metrics["recall_at_k"] = recall_at_k(ranked, union, truth, len(queries), queries.country)
        if keep_ranks and ranked is not None:
            ranked_store[name] = ranked
        del ranked
        metrics["channel_stats"] = run["stats"]
        channel_metrics[name] = metrics
        union.add(keys, name)
        log(f"  {name:<22} pairs {len(keys):>12,}  "
            + (f"pair recall {metrics['pair_recall']:.4f}  ceiling {metrics['f05_ceiling']:.4f}  "
               f"mean cands {metrics['cand_mean']:.1f}  " if truth is not None else "")
            + f"{run['seconds']}s  | union {len(union.keys):,}")
        del keys

    return union, channel_metrics, ranked_store


def union_ranks(union: CandidateUnion, ranked: tuple) -> np.ndarray:
    """Per union pair: the channel's rank for it (-1 if the channel did not retrieve it)."""
    q, r, rank = ranked
    keys = q.astype(np.int64) * union.n_ref + r.astype(np.int64)
    out = np.full(len(union.keys), -1, dtype=np.int32)
    out[np.searchsorted(union.keys, keys)] = rank
    return out
