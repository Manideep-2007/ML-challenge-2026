"""
Stage 9 — collective second-stage matcher: cross-fitted evaluation, final fit, test apply.

    python src/stacking/run_stage9.py --evaluate      # 4-fold cross-fit on validation vs frozen baseline
    python src/stacking/run_stage9.py --fit           # final model on all validation pairs
    python src/stacking/run_stage9.py --apply-test    # re-score test pairs, write output/matching_results.tsv

Stage-1 probabilities on validation are out-of-sample (stage 1 never saw
validation S1), so stage 2 is trained on them exactly as it will be applied to
test. Folds split by S1; the decision threshold is chosen on the other folds'
out-of-fold scores (never on the fold being scored).
"""

from pathlib import Path
from datetime import date
import argparse
import glob
import hashlib
import json
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from analysis.experiment_runner import compare, log_experiment  # noqa: E402
from decision.decision_engine import DecisionParams, entity_outcomes, select  # noqa: E402
from decision.ranking import prepare  # noqa: E402
from stacking.collective import FLOOR, TEXT_COLUMNS, collective_features, feature_names, text_views  # noqa: E402

ROOT = SRC.parents[2]
NORMALIZED = ROOT / "artifacts" / "normalized"
WORK = ROOT / "artifacts" / "stacking"
REPORT = ROOT / "experiments" / "stage9"
MODEL_ARTIFACTS = SRC.parent / "model_artifacts"
OUT_MODEL = MODEL_ARTIFACTS / "collective"
VARIANT = {"name": "v1", "suffix": "", "experiment": "E005", "evidence": False, "universe": False}


def configure(variant: str):
    """v1: collective features (E005). v2: + pair-local stage-1 v2 / street evidence (E006)."""
    global OUT_MODEL, REPORT
    if variant == "v2":
        VARIANT.update({"name": "v2", "suffix": "_v2", "experiment": "E006", "evidence": True})
        OUT_MODEL = MODEL_ARTIFACTS / "collective_v2"
        REPORT = ROOT / "experiments" / "stage9" / "v2"
    if variant == "v3c":   # v3b + transliterated-name features (E008)
        VARIANT.update({"name": "v3c", "suffix": "_v3c", "experiment": "E008", "evidence": False,
                        "universe": True, "translit": True})
        OUT_MODEL = MODEL_ARTIFACTS / "collective_v3c"
        REPORT = ROOT / "experiments" / "stage9" / "v3c"
    if variant in ("v3", "v3b"):
        # v3: v2 + universe name statistics (E007); v3b: v1 + universe (if E006 is rejected)
        VARIANT.update({"name": variant, "suffix": f"_{variant}", "experiment": "E007" if variant == "v3" else "E007b",
                        "evidence": variant == "v3", "universe": True})
        OUT_MODEL = MODEL_ARTIFACTS / f"collective_{variant}"
        REPORT = ROOT / "experiments" / "stage9" / variant
VALIDATION_PRED = ROOT / "artifacts" / "predictions" / "E001_lgbm_xgb_validation.parquet"
VALIDATION_GT = ROOT / "experiments" / "stage2" / "validation_ground_truth.tsv"
DECISION = (MODEL_ARTIFACTS if (MODEL_ARTIFACTS / "decision_params.json").exists()
            else ROOT / "experiments" / "stage7") / "decision_params.json"
TEST_FINAL = ROOT / "artifacts" / "final" / "test"
OUTPUT = ROOT / "output"
THRESHOLDS = np.round(np.arange(0.30, 0.96, 0.025), 3)
FOLDS = 4
PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 100,
          "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
          "random_state": 42, "n_jobs": -1, "verbosity": -1}


def ownership(s1: np.ndarray, cand: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """
    Exclusivity-aware probability: every S2/S3 record belongs to at most one S1, so the
    competing S1 of one record are alternatives, not independent events. With p_i the
    pairwise probability of S1 i for the record,
        P(record belongs to S1 a) = odds_a / (1 + sum_i odds_i),   odds = p / (1 - p).
    Equals p_a when only one S1 wants the record. Test has every S1 present (validation
    only 20% of them), so this matters most on test -- sibling businesses in particular.
    """
    p = np.clip(prob.astype(np.float64), 1e-6, 1 - 1e-6)
    odds = p / (1 - p)
    total = pd.Series(odds).groupby(cand).transform("sum").to_numpy()
    return (odds / (1 + total)).astype(np.float32)


def base_params() -> DecisionParams:
    raw = json.loads(DECISION.read_text())
    raw.pop("model", None)
    raw["country_t"] = {}
    return DecisionParams(**raw)


def with_threshold(t: float) -> DecisionParams:
    p = base_params()
    p.t_first = p.t_rest = float(t)
    return p


def fold_of(ids: pd.Series, salt: str = "", folds: int = FOLDS) -> np.ndarray:
    return np.array([int(hashlib.md5((salt + i).encode()).hexdigest()[:8], 16) % folds for i in ids])


# ------------------------------------------------------------------ data
def load_text(paths, ids) -> pd.DataFrame:
    frames = [pq.read_table(p, columns=TEXT_COLUMNS).to_pandas() for p in paths]
    frame = pd.concat(frames, ignore_index=True)
    return text_views(frame[frame["entity_id"].isin(ids)])


def validation_frame() -> pd.DataFrame:
    path = WORK / f"validation_collective{VARIANT['suffix']}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    if VARIANT.get("translit"):   # extend the cached v3b frame
        frame = pd.read_parquet(WORK / "validation_collective_v3b.parquet")
        frame = with_translit(frame, "validation")
        frame.to_parquet(path)
        return frame
    if VARIANT["universe"]:   # extend the cached base variant
        base = WORK / f"validation_collective{'_v2' if VARIANT['evidence'] else ''}.parquet"
        frame = pd.read_parquet(base)
        frame = with_universe(frame, "validation")
        frame.to_parquet(path)
        return frame
    WORK.mkdir(parents=True, exist_ok=True)
    start = time.time()
    pred = pq.read_table(VALIDATION_PRED, filters=[("probability", ">=", FLOOR)]).to_pandas()
    pairs = pred.rename(columns={"source1_entity_id": "s1", "candidate_entity_id": "cand", "probability": "p"})
    s1 = load_text([NORMALIZED / "train_source1.parquet"], set(pairs["s1"]))
    ref = load_text([NORMALIZED / "train_source2.parquet", NORMALIZED / "train_source3.parquet"], set(pairs["cand"]))
    print(f"  text views ({time.time() - start:.0f}s)")
    label = pairs.set_index(["s1", "cand"])["label"]
    frame = collective_features(pairs[["s1", "cand", "p"]], s1, ref)
    frame["label"] = label.reindex(pd.MultiIndex.from_frame(frame[["s1", "cand"]])).to_numpy().astype(np.int8)
    frame["country"] = s1["country_key"].reindex(frame["s1"]).to_numpy()
    if VARIANT["evidence"]:
        frame = with_evidence(frame, "validation")
    frame.to_parquet(path)
    print(f"  validation collective features: {len(frame):,} pairs ({time.time() - start:.0f}s)")
    return frame


def ground_truth():
    gt = pd.read_csv(VALIDATION_GT, sep="\t", dtype=str, keep_default_na=False)
    ids = pd.Index(gt["source1_entity_id"])
    true_counts = gt["matched_entity_ids"].map(lambda s: len(s.split(",")) if s else 0).to_numpy()
    country = pd.read_parquet(NORMALIZED / "train_source1.parquet", columns=["entity_id", "country_key"]) \
        .set_index("entity_id")["country_key"].reindex(ids).to_numpy()
    return ids, true_counts, country


def with_translit(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    from stacking.collective import transliteration_features
    prefix = "train" if split == "validation" else "test"
    s1 = pd.read_parquet(NORMALIZED / f"{prefix}_source1.parquet", columns=["entity_id", "name_raw"]).set_index("entity_id")["name_raw"]
    ref = pd.concat([pd.read_parquet(NORMALIZED / f"{prefix}_source{i}.parquet", columns=["entity_id", "name_raw"])
                     for i in (2, 3)]).set_index("entity_id")["name_raw"]
    s1 = s1[s1.index.isin(set(frame["s1"]))]
    ref = ref[ref.index.isin(set(frame["cand"]))]
    t = transliteration_features(frame[["s1", "cand"]], s1, ref)
    return pd.concat([frame.reset_index(drop=True), t.reset_index(drop=True)], axis=1)


_UNIVERSE = {}


def with_universe(frame: pd.DataFrame, split: str, s1=None, ref=None) -> pd.DataFrame:
    """Append universe name statistics (stacking/collective.universe_features, E007)."""
    from stacking.collective import universe_counts, universe_features
    prefix = "train" if split == "validation" else "test"
    if split not in _UNIVERSE:
        _UNIVERSE[split] = universe_counts([NORMALIZED / f"{prefix}_source1.parquet"],
                                           [NORMALIZED / f"{prefix}_source2.parquet", NORMALIZED / f"{prefix}_source3.parquet"])
    if s1 is None:
        s1 = load_text([NORMALIZED / f"{prefix}_source1.parquet"], set(frame["s1"]))
        ref = load_text([NORMALIZED / f"{prefix}_source2.parquet", NORMALIZED / f"{prefix}_source3.parquet"],
                        set(frame["cand"]))
    u = universe_features(frame[["s1", "cand"]], s1, ref, _UNIVERSE[split])
    return pd.concat([frame.reset_index(drop=True), u.reset_index(drop=True)], axis=1)


def with_evidence(frame: pd.DataFrame, split: str, country: str | None = None) -> pd.DataFrame:
    """Append pair-local stage-1 v2 + street features (stacking/pair_evidence.py)."""
    from stacking.pair_evidence import PROVENANCE, pair_evidence
    from inference.pipeline import part_files
    from features.feature_builder import reference_document_frequencies
    keys = frame[["s1", "cand"]].rename(columns={"s1": "source1_entity_id", "cand": "candidate_entity_id"})
    if split == "validation":
        translations = ROOT / "experiments" / "stage4"
        s1_paths = [NORMALIZED / "train_source1.parquet"]
        ref_paths = [NORMALIZED / "train_source2.parquet", NORMALIZED / "train_source3.parquet"]
        df_path = WORK / "train_reference_document_frequencies.joblib"
        if not df_path.exists():
            joblib.dump(reference_document_frequencies(ref_paths, translations), df_path)
        global_df = joblib.load(df_path)
        candidate_files = [ROOT / "artifacts" / "candidates" / "validation_candidates.parquet"]
    else:
        translations = MODEL_ARTIFACTS / "translations"
        s1_paths = [NORMALIZED / "test_source1.parquet"]
        ref_paths = [NORMALIZED / "test_source2.parquet", NORMALIZED / "test_source3.parquet"]
        global_df = joblib.load(TEST_FINAL / "reference_document_frequencies.joblib")
        candidate_files = part_files(TEST_FINAL / country, "candidates")
    wanted = set(keys["source1_entity_id"])
    columns = ["source1_entity_id", "candidate_entity_id", "candidate_source"] + PROVENANCE
    prov = pd.concat([pq.read_table(f, columns=columns, filters=[("source1_entity_id", "in", list(wanted))]).to_pandas()
                      for f in candidate_files], ignore_index=True)
    pairs = keys.merge(prov, on=["source1_entity_id", "candidate_entity_id"], how="left")
    if pairs["candidate_source"].isna().any():
        raise ValueError("stage-2 pairs missing from the candidate file")
    evidence = pair_evidence(pairs, s1_paths, ref_paths, translations, global_df, country)
    return pd.concat([frame.reset_index(drop=True), evidence], axis=1)


# ------------------------------------------------------------------ model
def fit(frame: pd.DataFrame, features: list[str]) -> lgb.Booster:
    unique = frame["s1"].drop_duplicates()
    dev = fold_of(unique, salt="dev", folds=10) == 0   # inner early-stopping split by S1 (independent hash)
    dev_ids = set(unique[dev])
    is_dev = frame["s1"].isin(dev_ids).to_numpy()
    train = lgb.Dataset(frame.loc[~is_dev, features], frame.loc[~is_dev, "label"])
    valid = lgb.Dataset(frame.loc[is_dev, features], frame.loc[is_dev, "label"])
    return lgb.train(PARAMS, train, num_boost_round=4000, valid_sets=[valid],
                     callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(500)])


def entity_table(frame: pd.DataFrame, prob: np.ndarray, t: float, ids, true_counts, country) -> pd.DataFrame:
    code = ids.get_indexer(frame["s1"]).astype(np.int64)
    cand, _ = pd.factorize(frame["cand"])
    scored = prepare(code, cand.astype(np.int64), frame["is_s2"].to_numpy() == 1, prob.astype(np.float32),
                     frame["label"].to_numpy(np.int8), country, floor=0.0)
    o = entity_outcomes(scored, select(scored, with_threshold(t)), true_counts)
    return pd.DataFrame({"s1_id": ids, "f05": o["f05"], "precision": o["precision"], "recall": o["recall"],
                         "fp": (o["n_pred"] - o["tp"]).astype(int), "fn": (true_counts - o["tp"]).astype(int),
                         "country": country})


def evaluate():
    REPORT.mkdir(parents=True, exist_ok=True)
    frame = validation_frame()
    features = feature_names(frame)
    ids, true_counts, country = ground_truth()
    frame["fold"] = fold_of(frame["s1"])
    oof = np.zeros(len(frame), np.float32)
    importance = []
    oof_path = WORK / f"validation_oof{VARIANT['suffix']}.parquet"
    if oof_path.exists():   # resume: out-of-fold scores already computed
        saved = pd.read_parquet(oof_path)
        if saved[["s1", "cand"]].equals(frame[["s1", "cand"]]):
            print("  reusing saved out-of-fold scores")
            oof = saved["p2"].to_numpy(np.float32)
    for k in range(FOLDS if not oof.any() else 0):
        test = frame["fold"].to_numpy() == k
        model = fit(frame[~test], features)
        oof[test] = model.predict(frame.loc[test, features], num_iteration=model.best_iteration)
        importance.append(pd.Series(model.feature_importance("gain"), index=features))
        print(f"  fold {k}: best_iteration {model.best_iteration}")
    frame["p2"] = oof
    frame[["s1", "cand", "p", "p2", "label", "fold"]].to_parquet(oof_path)

    # threshold per fold chosen on the OTHER folds' out-of-fold scores
    s1_fold = pd.Series(fold_of(pd.Series(ids)), index=ids)
    curves, chosen = [], {}
    for t in THRESHOLDS:
        table = entity_table(frame, oof, t, ids, true_counts, country)
        table["fold"] = s1_fold.to_numpy()
        curves.append({"t": t, **{f"fold{k}": table.loc[table["fold"] == k, "f05"].mean() for k in range(FOLDS)},
                       "all": table["f05"].mean()})
    curves = pd.DataFrame(curves)
    curves.to_csv(REPORT / "threshold_curve.csv", index=False)
    parts = []
    for k in range(FOLDS):
        others = curves[[f"fold{j}" for j in range(FOLDS) if j != k]].mean(axis=1)
        chosen[k] = float(curves.loc[others.idxmax(), "t"])
        table = entity_table(frame, oof, chosen[k], ids, true_counts, country)
        parts.append(table[s1_fold.to_numpy() == k])
    experiment = pd.concat(parts).set_index("s1_id").reindex(ids).rename_axis("s1_id").reset_index()
    baseline = entity_table(frame, frame["p"].to_numpy(), 0.80, ids, true_counts, country)

    result = compare(baseline, experiment)
    by_country = {c: {"baseline": round(baseline.loc[baseline["country"] == c, "f05"].mean(), 6),
                      "collective": round(experiment.loc[experiment["country"] == c, "f05"].mean(), 6)}
                  for c in sorted(set(country))}
    final_t = float(curves.loc[curves["all"].idxmax(), "t"])
    row = {"experiment_id": VARIANT["experiment"], "date": date.today().isoformat(), "baseline_version": "baseline_v1",
           "change": "collective second-stage matcher (anchor agreement + probability structure + street)"
                     + (" + pair-local stage-1 v2 / street evidence" if VARIANT["evidence"] else ""),
           "hypothesis": "records of one business agree with each other; siblings disagree with the anchors",
           "affected_component": "model (stage 2)",
           **{k: result[k] for k in ("s1", "baseline_macro_f05", "experiment_macro_f05", "delta_f05", "delta_p05",
                                      "delta_p95", "entities_improved", "entities_degraded", "fp_change", "fn_change")},
           "precision": round(float(experiment["precision"].mean()), 6),
           "recall": round(float(experiment["recall"].mean()), 6),
           "status": "KEEP" if result["delta_p05"] > 0 else ("REJECT" if result["delta_f05"] <= 0 else "INCONCLUSIVE"),
           "notes": json.dumps({"fold_thresholds": chosen, "final_threshold": final_t, "by_country": by_country})}
    log_experiment(ROOT / "experiments" / "stage8" / "improvement_experiments.csv", row)
    if importance:
        imp = pd.concat(importance, axis=1).mean(axis=1).sort_values(ascending=False)
        (100 * imp / imp.sum()).round(3).to_csv(REPORT / "feature_importance_gain_pct.csv", header=["gain_pct"])
    (REPORT / "evaluation.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2))


FIRST_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
REST_GRID = [0.60, 0.65, 0.70, 0.725, 0.75, 0.80]
# France has no labels and ~5x more mid-probability pairs than validation (sibling businesses):
# F0.5 punishes false merges, so its best-candidate threshold never goes below the single
# threshold E005 selected (0.65) -- a deliberate, conservative choice for an unmeasured segment.
FRANCE_MIN_FIRST = 0.65


def tune_decision():
    """(t_first, t_rest) on stage-2 out-of-fold scores; reported cross-fold, chosen on all folds."""
    oof = pd.read_parquet(WORK / f"validation_oof{VARIANT['suffix']}.parquet")
    ids, true_counts, country = ground_truth()
    code = ids.get_indexer(oof["s1"]).astype(np.int64)
    cand, _ = pd.factorize(oof["cand"])
    owned = ownership(oof["s1"].to_numpy(), oof["cand"].to_numpy(), oof["p2"].to_numpy())
    scored = prepare(code, cand.astype(np.int64), oof["cand"].str.startswith("S2-").to_numpy(),
                     owned, oof["label"].to_numpy(np.int8), country, floor=0.0)
    fold = fold_of(pd.Series(ids))
    rows = []
    for tf in FIRST_GRID:
        for tr in REST_GRID:
            p = base_params()
            p.t_first, p.t_rest = tf, tr
            f = entity_outcomes(scored, select(scored, p), true_counts)["f05"]
            rows.append({"t_first": tf, "t_rest": tr, "all": f.mean(),
                         **{f"fold{k}": f[fold == k].mean() for k in range(FOLDS)}})
    grid = pd.DataFrame(rows)
    grid.to_csv(REPORT / "decision_grid.csv", index=False)
    honest = 0.0
    for k in range(FOLDS):
        others = grid[[f"fold{j}" for j in range(FOLDS) if j != k]].mean(axis=1)
        honest += grid.loc[others.idxmax(), f"fold{k}"] * (fold == k).sum()
    best = grid.loc[grid["all"].idxmax()]
    result = {"t_first": float(best["t_first"]), "t_rest": float(best["t_rest"]),
              "validation_f05": round(float(best["all"]), 6), "cross_fold_f05": round(honest / len(fold), 6),
              "country_t": {"france": [max(float(best["t_first"]), FRANCE_MIN_FIRST), float(best["t_rest"])]}}
    (REPORT / "decision.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def fit_final():
    frame = validation_frame()
    features = feature_names(frame)
    decision = tune_decision()
    model = fit(frame, features)
    OUT_MODEL.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUT_MODEL / "model.txt"), num_iteration=model.best_iteration)
    (OUT_MODEL / "config.json").write_text(json.dumps(
        {"features": features, "ownership": True, "t_first": decision["t_first"], "t_rest": decision["t_rest"],
         "country_t": decision["country_t"], "floor": FLOOR,
         "trained_on": "validation-split pairs (stage-1 E001_lgbm_xgb out-of-sample probabilities)",
         "best_iteration": model.best_iteration}, indent=2))
    print(f"saved {OUT_MODEL} ({decision})")


# ------------------------------------------------------------------ test
def decision_from_config(config: dict) -> DecisionParams:
    p = base_params()
    if "t_first" in config:
        p.t_first, p.t_rest = config["t_first"], config["t_rest"]
        p.country_t = {k: tuple(v) for k, v in config.get("country_t", {}).items()}
    else:   # first release: one threshold
        p.t_first = p.t_rest = config["threshold"]
    return p


def apply_test(rescore: bool = True):
    """Stage-2 scores per country are cached in artifacts/stacking/test_<country>.parquet;
    rescore=False only re-applies the decision rules."""
    from inference.pipeline import part_files
    config = json.loads((OUT_MODEL / "config.json").read_text())
    model = lgb.Booster(model_file=str(OUT_MODEL / "model.txt"))
    params = decision_from_config(config)
    s1_all = pd.read_parquet(NORMALIZED / "test_source1.parquet", columns=["entity_id", "country_key"])
    matches = {k: [] for k in s1_all["entity_id"]}
    for country in sorted(s1_all["country_key"].unique()):
        start = time.time()
        cache = WORK / f"test_{country}{VARIANT['suffix']}.parquet"
        if rescore or not cache.exists():
            pred = pd.concat([pq.read_table(f, filters=[("probability", ">=", FLOOR)]).to_pandas()
                              for f in part_files(TEST_FINAL / country, "predictions")], ignore_index=True)
            pairs = pred.rename(columns={"source1_entity_id": "s1", "candidate_entity_id": "cand", "probability": "p"})
            s1 = load_text([NORMALIZED / "test_source1.parquet"], set(pairs["s1"]))
            ref = load_text([NORMALIZED / "test_source2.parquet", NORMALIZED / "test_source3.parquet"],
                            set(pairs["cand"]))
            frame = collective_features(pairs, s1, ref)
            if VARIANT["evidence"]:
                frame = with_evidence(frame, "test", country)
            if VARIANT["universe"]:
                frame = with_universe(frame, "test", s1, ref)
            if VARIANT.get("translit"):
                frame = with_translit(frame, "test")
            frame["p2"] = model.predict(frame[config["features"]]).astype(np.float32)
            frame = frame[["s1", "cand", "is_s2", "p", "p2"]]
            frame.to_parquet(cache)
        else:
            frame = pd.read_parquet(cache)
        ids = pd.Index(s1_all.loc[s1_all["country_key"] == country, "entity_id"])
        code = ids.get_indexer(frame["s1"]).astype(np.int64)
        cand, cand_ids = pd.factorize(frame["cand"])
        prob = frame["p2"].to_numpy(np.float32)
        if config.get("ownership"):
            prob = ownership(frame["s1"].to_numpy(), frame["cand"].to_numpy(), prob)
        scored = prepare(code, cand.astype(np.int64), frame["is_s2"].to_numpy() == 1, prob,
                         np.zeros(len(frame), np.int8), np.full(len(ids), country, dtype=object), floor=0.0)
        keep = select(scored, params)
        for c, r in zip(scored.code[keep], scored.cand[keep]):
            matches[ids[c]].append(cand_ids[r])
        print(f"  {country}: {len(frame):,} pairs, {int(keep.sum()):,} matches ({time.time() - start:.0f}s)")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for k, v in matches.items():
            f.write(f"{k}\t{','.join(dict.fromkeys(v))}\n")
    print(f"wrote {OUTPUT / 'matching_results.tsv'}: {sum(map(len, matches.values())):,} matches, "
          f"{sum(bool(v) for v in matches.values()):,} S1 with matches")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["v1", "v2", "v3", "v3b", "v3c"], default="v1")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--apply-test", action="store_true")
    parser.add_argument("--decide-only", action="store_true", help="re-apply decision rules to cached test scores")
    args = parser.parse_args()
    configure(args.variant)
    if args.evaluate:
        evaluate()
    if args.fit:
        fit_final()
    if args.apply_test or args.decide_only:
        apply_test(rescore=not args.decide_only)


if __name__ == "__main__":
    main()
