"""
End-to-end inference: normalized records -> candidates -> features ->
probabilities -> per-S1 decisions.

Reference-side structures are built once (translators, per-country hybrid
retrievers, feature token indexes); S1 queries are processed in chunks so
~240M test candidate pairs never have to be in memory at once. Uses exactly
the Stage 4 / 5 / 6 / 7 components.
"""

from pathlib import Path
import gc
import json
import time

import lightgbm as lgb
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from blocking.candidate_union import CandidateUnion
from blocking.candidate_generator import union_ranks
from blocking.exact_blocks import ExactIndex
from blocking.token_blocks import RareTokenRetriever, deleet
from features.feature_builder import (
    SIDE_COLUMNS, add_record_views, batch_features, build_indexes, load_translator, load_views,
)
from decision.decision_engine import DecisionParams, select
from decision.ranking import prepare

def load_member(model_dir: Path) -> dict:
    """One trained model: LightGBM (model.txt) or XGBoost (model.json), optional calibrator."""
    config = json.loads((model_dir / "config.json").read_text())
    if (model_dir / "model.txt").exists():
        kind, booster = "lightgbm", lgb.Booster(model_file=str(model_dir / "model.txt"))
    else:
        kind, booster = "xgboost", xgb.Booster()
        booster.load_model(str(model_dir / "model.json"))
    calibrator_path = model_dir / "calibrator.joblib"
    return {"kind": kind, "booster": booster, "features": config["features"],
            "calibrator": joblib.load(calibrator_path) if calibrator_path.exists() else None}


def predict_member(member: dict, features: pd.DataFrame) -> np.ndarray:
    X = features[member["features"]]
    if member["kind"] == "lightgbm":
        prob = member["booster"].predict(X)
    else:
        # Early stopping trains past the best round; predict with the best rounds only,
        # exactly as XGBClassifier.predict_proba does.
        best = member["booster"].attr("best_iteration")
        rounds = (0, int(best) + 1) if best is not None else (0, 0)
        prob = member["booster"].predict(xgb.DMatrix(X), iteration_range=rounds)
    prob = np.asarray(prob, dtype=np.float32)
    return member["calibrator"].transform(prob) if member["calibrator"] is not None else prob


EXACT_CHANNELS = {
    "content_name": "name_content_compact",
    "fingerprint_name": "name_fingerprint",
    "fingerprint_address": "address_fingerprint",
}
CHANNELS = list(EXACT_CHANNELS) + ["hybrid_translated"]


class Pipeline:
    def __init__(self, s1_paths, reference_paths, translations_dir: Path, model_dir: Path,
                 blocking_config: dict, s1_ids: set | None = None, log=print,
                 country: str | None = None, global_df: dict | None = None):
        """
        country: restrict to one country (exact: no true link crosses countries and every
        channel is country-scoped); global_df then supplies IDF over the full reference universe.
        """
        self.log = log
        start = time.time()
        self.s1 = load_views(s1_paths, s1_ids, country)
        self.reference = load_views(reference_paths, country=country)
        add_record_views(self.s1, None)
        add_record_views(self.reference, translations_dir)
        log(f"  records + views: {len(self.s1):,} S1, {len(self.reference):,} reference ({time.time() - start:.0f}s)")

        self.indexes = build_indexes(self.s1, self.reference, global_df)
        self.s1_pos = pd.Index(self.s1["entity_id"])
        self.ref_pos = pd.Index(self.reference["entity_id"])
        log(f"  feature token indexes ({time.time() - start:.0f}s)")

        name_tr = load_translator(translations_dir / "token_translation_name.csv")
        address_tr = load_translator(translations_dir / "token_translation_address.csv")
        self.config = blocking_config
        self.ref_country = self.reference["country_key"].to_numpy()
        self.retrievers = {}
        for country in np.unique(self.ref_country):
            part = np.flatnonzero(self.ref_country == country)
            retriever = RareTokenRetriever(
                ["name_content", "address_basic"], blocking_config["hybrid_max_df"], blocking_config["token_top_k"],
                transforms={"name_content": deleet},
                ref_transforms={"name_content": name_tr, "address_basic": address_tr},
            ).fit(self.reference.iloc[part][["name_content", "address_basic"]].reset_index(drop=True))
            exact = {name: ExactIndex(self.reference[view].iloc[part].reset_index(drop=True), blocking_config["exact_max_block"])
                     for name, view in EXACT_CHANNELS.items()}
            self.retrievers[country] = (part, retriever, exact)
        log(f"  hybrid retrievers for {len(self.retrievers)} countries ({time.time() - start:.0f}s)")

        # Columns only needed to build the token indexes / retrievers: drop them so the
        # streaming phase fits in memory (1.7M S1 x 10M references on 16 GB).
        keep = ["entity_id"] + [c for c in SIDE_COLUMNS if c != "entity_id"]
        self.s1 = self.s1[[c for c in keep if c in self.s1.columns]]
        self.reference = self.reference[[c for c in keep if c in self.reference.columns]]
        gc.collect()

        model_config = json.loads((model_dir / "config.json").read_text())
        member_dirs = [model_dir.parent / m for m in model_config["members"]] if "members" in model_config else [model_dir]
        self.members = [load_member(d) for d in member_dirs]
        log(f"  model: {model_dir.name} ({len(self.members)} member(s))")

    # ---------------------------------------------------------------
    def candidates(self, s1_positions: np.ndarray) -> pd.DataFrame:
        """Stage 4 candidate union (with provenance and ranks) for a chunk of S1 records."""
        n_ref = len(self.reference)
        union = CandidateUnion(n_ref, CHANNELS)
        ranked = None
        chunk = self.s1.iloc[s1_positions].reset_index(drop=True)
        chunk_country = chunk["country_key"].to_numpy()
        per_channel = {name: ([], []) for name in CHANNELS}
        rank_parts = []

        for country, (part, retriever, exact) in self.retrievers.items():
            q_local = np.flatnonzero(chunk_country == country)
            if len(q_local) == 0:
                continue
            queries = chunk.iloc[q_local].reset_index(drop=True)
            for name, view in EXACT_CHANNELS.items():
                q, r = exact[name].query(queries[view])
                per_channel[name][0].append(q_local[q])
                per_channel[name][1].append(part[r])
            q, r, rank = retriever.query(queries[["name_content", "address_basic"]])
            per_channel["hybrid_translated"][0].append(q_local[q])
            per_channel["hybrid_translated"][1].append(part[r])
            rank_parts.append(rank)

        for name, (qs, rs) in per_channel.items():
            if qs:
                union.add(union.pair_keys(np.concatenate(qs), np.concatenate(rs)), name)
        if rank_parts:
            q_h, r_h = per_channel["hybrid_translated"]
            ranked = (np.concatenate(q_h), np.concatenate(r_h), np.concatenate(rank_parts))

        q_idx, r_idx = union.query_ref()
        labels, n_channels = union.channel_labels()
        counts = np.bincount(q_idx, minlength=len(chunk))
        candidate_ids = self.reference["entity_id"].take(r_idx).to_numpy()   # only the selected rows
        return pd.DataFrame({
            "source1_entity_id": chunk["entity_id"].to_numpy()[q_idx],
            "candidate_entity_id": candidate_ids,
            "candidate_source": np.array([i[:2] for i in candidate_ids], dtype=object),
            "blocking_channels": labels,
            "num_blocking_channels": n_channels,
            "hybrid_translated_rank": union_ranks(union, ranked) if ranked is not None else -1,
            "candidates_for_source1": counts[q_idx].astype(np.int32),
        })

    def score(self, candidates: pd.DataFrame) -> tuple[np.ndarray, float]:
        started = time.time()
        features = batch_features(candidates, self.s1, self.reference, self.s1_pos, self.ref_pos,
                                  self.indexes, CHANNELS)
        feature_seconds = time.time() - started
        prob = np.mean([predict_member(m, features) for m in self.members], axis=0).astype(np.float32)
        return prob, feature_seconds

    def run(self, out_dir: Path, chunk_size: int = 100_000) -> dict:
        """Candidates + probabilities for every S1, one parquet part per chunk.

        Parts are written atomically (tmp + rename) and finished parts are skipped,
        so an interrupted run resumes at the first missing chunk.
        """
        parts = out_dir / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        start, pairs = time.time(), 0
        for begin in range(0, len(self.s1), chunk_size):
            positions = np.arange(begin, min(begin + chunk_size, len(self.s1)))
            cand_path, pred_path = parts / f"candidates_{begin:08d}.parquet", parts / f"predictions_{begin:08d}.parquet"
            if pred_path.exists() and cand_path.exists():
                pairs += pq.ParquetFile(pred_path).metadata.num_rows
                self.log(f"  S1 {positions[-1] + 1:,}/{len(self.s1):,}  reused part")
                continue
            t0 = time.time()
            cands = self.candidates(positions)
            t1 = time.time()
            if len(cands) == 0:
                self.log(f"  S1 {positions[-1] + 1:,}/{len(self.s1):,}  no candidates")
                continue
            prob, feature_seconds = self.score(cands)
            t2 = time.time()
            cand_table = pa.Table.from_pandas(cands, preserve_index=False)
            pred_table = pa.table({"source1_entity_id": cands["source1_entity_id"].to_numpy(),
                                   "candidate_entity_id": cands["candidate_entity_id"].to_numpy(),
                                   "probability": prob})
            for table, path in ((cand_table, cand_path), (pred_table, pred_path)):
                tmp = path.with_suffix(".tmp")
                pq.write_table(table, tmp, compression="zstd")
                tmp.replace(path)
            pairs += len(cands)
            self.log(f"  S1 {positions[-1] + 1:,}/{len(self.s1):,}  pairs {pairs:,}  {time.time() - start:.0f}s  "
                     f"(candidates {t1 - t0:.0f}s, features {feature_seconds:.0f}s, model {t2 - t1 - feature_seconds:.0f}s)")
        return {"s1": len(self.s1), "pairs": pairs, "seconds": round(time.time() - start, 1)}


def part_files(out_dir: Path, kind: str) -> list[Path]:
    """Chunk parts in S1 order; falls back to a single legacy file."""
    parts = sorted((out_dir / "parts").glob(f"{kind}_*.parquet"))
    legacy = out_dir / f"{kind}.parquet"
    return parts or ([legacy] if legacy.exists() else [])


def decide(out_dirs, s1_ids: np.ndarray, params: DecisionParams,
           s1_country: pd.Series) -> dict[str, list[str]]:
    """Apply the Stage 7 decision rules to all predictions (exclusivity across every S1)."""
    out_dirs = [out_dirs] if isinstance(out_dirs, Path) else list(out_dirs)
    # Exact pre-filter: a pair below every threshold can never be kept, and dropping
    # the lowest-probability pairs of an S1 does not change the ranks of the others
    # (the margin rule needs the true second-best, so it disables the filter).
    floor = min([params.t_first, params.t_rest] + [t for pair in params.country_t.values() for t in pair])
    floor = floor if params.margin_min == 0 else 0.0
    table = pd.concat([pq.read_table(f, filters=[("probability", ">=", floor)]).to_pandas()
                       for d in out_dirs for f in part_files(d, "predictions")], ignore_index=True)
    s1_index = pd.Index(s1_ids)
    code = s1_index.get_indexer(table["source1_entity_id"]).astype(np.int64)
    cand_code, cand_ids = pd.factorize(table["candidate_entity_id"])
    is_s2 = table["candidate_entity_id"].str.startswith("S2-").to_numpy()
    country = s1_country.reindex(s1_index).to_numpy()
    scored = prepare(code, cand_code.astype(np.int64), is_s2, table["probability"].to_numpy(np.float32),
                     np.zeros(len(table), np.int8), country)
    keep = select(scored, params)
    matches = {k: [] for k in s1_ids}
    for c, r in zip(scored.code[keep], scored.cand[keep]):
        matches[s1_ids[c]].append(cand_ids[r])
    return matches
