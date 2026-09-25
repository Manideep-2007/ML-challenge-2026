from pathlib import Path
import json
import random
import sys
import tempfile

import pandas as pd
from sklearn.metrics import fbeta_score as sklearn_fbeta

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))

from evaluation.metrics import BETA, entity_metrics, macro_fbeta  # noqa: E402
from evaluation.evaluator import (  # noqa: E402
    evaluate_predictions,
    load_ground_truth,
    load_predictions,
    prediction_issues,
)

STAGE2_DIR = ROOT / "experiments" / "stage2"
VALIDATION_GT = STAGE2_DIR / "validation_ground_truth.tsv"
PROFILE_PATH = STAGE2_DIR / "validation_profile.json"
S1_PATH = ROOT / "challenge" / "dataset" / "train" / "train_source1.tsv"

TOLERANCE = 1e-9


# ============================================================
# UNIT TESTS
# ============================================================

ENTITY_CASES = [
    ("perfect_match", ["S2-1"], ["S2-1"], 1.0),
    ("missed_match", ["S2-1"], [], 0.0),
    ("wrong_match", ["S2-1"], ["S2-999"], 0.0),
    ("correct_plus_false_positive", ["S2-1"], ["S2-1", "S2-999"], 0.625 / 1.125),
    ("correct_empty", [], [], 1.0),
    ("false_positive_on_singleton", [], ["S2-1"], 0.0),
    # Worked example from the challenge README: P=2/3, R=1 -> 5/7.
    ("challenge_readme_example", ["S2-00047", "S3-00812"],
     ["S2-00047", "S2-00193", "S3-00812"], 5 / 7),
    ("one_of_four_found", ["S2-1", "S2-2", "S3-1", "S3-2"], ["S3-1"], 0.625),
    ("duplicate_prediction_ids", ["S2-1"], ["S2-1", "S2-1"], 1.0),
    ("order_invariant", ["S2-1", "S3-2"], ["S3-2", "S2-1"], 1.0),
]


def check(name, actual, expected, results):
    passed = abs(actual - expected) < TOLERANCE
    results.append((name, actual, expected, passed))
    print(f"  {name:<40} {actual:.6f}  expected {expected:.6f}  {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise AssertionError(f"Failed: {name}")


def test_entity_cases(results):
    print("\nEntity-level metric tests:")
    for name, truth, prediction, expected in ENTITY_CASES:
        check(name, entity_metrics(prediction, truth)["f_beta"], expected, results)


def test_macro_averaging(results):
    print("\nMacro-averaging tests:")

    truth = {
        "S1-A": ["S2-1"],
        "S1-B": [],
        "S1-C": ["S2-2"],
        "S1-D": ["S2-3", "S3-3"],
    }
    predictions = {
        "S1-A": ["S2-1"],          # 1.0
        "S1-B": [],                # 1.0 (correct singleton)
        "S1-C": [],                # 0.0
        "S1-D": ["S2-3"],          # P=1, R=0.5 -> 0.8333...
        "S1-EXTRA": ["S2-9"],      # not in truth -> ignored
    }
    expected = (1.0 + 1.0 + 0.0 + (1.25 * 0.5 / 0.75)) / 4
    check("macro_mixed_entities", macro_fbeta(predictions, truth)["macro_f_beta"], expected, results)

    # Entity absent from predictions counts as an empty prediction.
    check(
        "missing_prediction_entity_is_empty",
        macro_fbeta({}, {"S1-A": ["S2-1"], "S1-B": []})["macro_f_beta"],
        0.5,
        results,
    )

    # Every entity weighs the same regardless of how many matches it has.
    check(
        "equal_weight_per_entity",
        macro_fbeta(
            {"S1-A": ["S2-1"]},
            {"S1-A": ["S2-1"], "S1-B": [f"S2-{i}" for i in range(100)]},
        )["macro_f_beta"],
        0.5,
        results,
    )


def test_against_sklearn(results, cases=500, seed=0):
    # Independent check of the F-beta formula on random non-empty cases.
    print("\nCross-check against sklearn.metrics.fbeta_score:")
    rng = random.Random(seed)
    worst = 0.0
    for _ in range(cases):
        universe = [f"S2-{i}" for i in range(rng.randint(2, 12))]
        truth = set(rng.sample(universe, rng.randint(1, len(universe))))
        predicted = set(rng.sample(universe, rng.randint(0, len(universe))))
        y_true = [u in truth for u in universe]
        y_pred = [u in predicted for u in universe]
        expected = sklearn_fbeta(y_true, y_pred, beta=BETA, zero_division=0.0)
        actual = entity_metrics(predicted, truth)["f_beta"]
        worst = max(worst, abs(actual - expected))
    check(f"sklearn_agreement_{cases}_random_cases (max abs diff)", worst, 0.0, results)


def test_issue_detection(results):
    print("\nSubmission-rule issue detection:")
    issues = prediction_issues(
        {"S1-A": ["S2-1", "S2-1"], "S1-B": ["S1-7"], "S1-X": []},
        {"S1-A": ["S2-1"], "S1-B": [], "S1-C": []},
    )
    check("detects_duplicate_ids", issues["lists_with_duplicate_ids"], 1, results)
    check("detects_invalid_prefix", issues["ids_with_invalid_prefix"], 1, results)
    check("detects_missing_entity", issues["entities_missing_from_predictions"], 1, results)
    check("detects_extra_entity", issues["prediction_entities_not_in_ground_truth"], 1, results)


# ============================================================
# REAL-DATA SANITY BASELINES (validation split, no model)
# ============================================================

def baselines(ground_truth):
    fake = iter(range(10**9))
    return {
        "oracle (predict ground truth)": ground_truth,
        "all empty (predict no matches)": {k: [] for k in ground_truth},
        "one correct match per entity": {k: v[:1] for k, v in ground_truth.items()},
        "half of the true matches (rounded up)": {
            k: v[: (len(v) + 1) // 2] for k, v in ground_truth.items()
        },
        "oracle + 1 false positive per entity": {
            k: v + [f"S2-FAKE{next(fake)}"] for k, v in ground_truth.items()
        },
    }


def summarize(result):
    return {
        "macro_f0.5": round(result["macro_f_beta"], 6),
        "macro_precision": round(result["macro_precision"], 6),
        "macro_recall": round(result["macro_recall"], 6),
        "singleton_macro_f0.5": round(result["by_truth_type"]["singleton"]["macro_f_beta"], 6),
        "matched_macro_f0.5": round(result["by_truth_type"]["matched"]["macro_f_beta"], 6),
        "by_country": {
            c: round(r["macro_f_beta"], 6) for c, r in result.get("by_group", {}).items()
        },
    }


def run_baselines(ground_truth, countries, results):
    print("\nReference baselines on the validation split:")
    scores = {}
    for name, predictions in baselines(ground_truth).items():
        scores[name] = summarize(
            evaluate_predictions(predictions, ground_truth, groups=countries)
        )
        s = scores[name]
        print(f"  {name:<40} macro F0.5 {s['macro_f0.5']:.6f}  "
              f"(singletons {s['singleton_macro_f0.5']:.4f}, "
              f"matched {s['matched_macro_f0.5']:.4f})")

    singleton_share = sum(1 for v in ground_truth.values() if not v) / len(ground_truth)
    check("oracle_scores_1", scores["oracle (predict ground truth)"]["macro_f0.5"], 1.0, results)
    check(
        "all_empty_equals_singleton_share",
        scores["all empty (predict no matches)"]["macro_f0.5"],
        round(singleton_share, 6),
        results,
    )
    return scores


def test_file_round_trip(ground_truth, results):
    # Write predictions in the exact matching_results.tsv format and score the file.
    print("\nFile round trip (matching_results.tsv format):")
    predictions = {k: v[:1] for k, v in ground_truth.items()}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "matching_results.tsv"
        pd.DataFrame({
            "source1_entity_id": list(predictions),
            "matched_entity_ids": [",".join(v) for v in predictions.values()],
        }).to_csv(path, sep="\t", index=False)
        loaded = load_predictions(path)

    check("round_trip_entities_preserved", len(loaded), len(predictions), results)
    check(
        "round_trip_score_identical",
        macro_fbeta(loaded, ground_truth)["macro_f_beta"],
        macro_fbeta(predictions, ground_truth)["macro_f_beta"],
        results,
    )


# ============================================================
# REPORT
# ============================================================

def write_report(profile, results, baseline_scores):
    v = profile["validation"]
    lines = [
        "STAGE 2 VALIDATION REPORT",
        "=" * 70,
        "",
        f"Total Source 1 entities:   {profile['total_entities']:,}",
        f"Training entities:         {profile['train_entities']:,}",
        f"Validation entities:       {profile['validation_entities']:,}",
        f"Validation fraction:       {profile['validation_fraction']}",
        f"Random state:              {profile['random_state']}",
        f"Validation IDs sha256:     {profile['validation_ids_sha256']}",
        f"Split strategy:            {profile['split_strategy']}",
        "",
        "Validation:",
        f"  Empty GT (singletons):   {v['singletons']:,} ({v['singleton_pct']}%)",
        f"  Single-match GT:         {v['single_match']:,}",
        f"  Multi-match GT:          {v['multi_match']:,}",
        f"  Positive links:          {v['positive_links']:,}",
        f"  S2 matches:              {v['s2_links']:,}",
        f"  S3 matches:              {v['s3_links']:,}",
        f"  Average matches/entity:  {v['average_matches']}",
        f"  Maximum matches/entity:  {v['maximum_matches']}",
        f"  Countries:               {v['country_distribution']}",
        "",
        "Metric:",
        "  F-beta:     F = (1 + b^2) P R / (b^2 P + R)",
        f"  Beta:       {BETA}",
        "  Averaging:  per Source 1 entity, macro-averaged over ALL entities",
        "  Empty GT + empty prediction = 1.0; IDs compared as sets",
        "",
        f"Tests: {sum(r[3] for r in results)}/{len(results)} passed",
    ]
    lines += [f"  {'PASS' if ok else 'FAIL'}  {name}" for name, _, _, ok in results]
    lines += ["", "Reference baselines (validation split, no model):"]
    for name, s in baseline_scores.items():
        lines.append(
            f"  {name:<40} macro F0.5 {s['macro_f0.5']:.6f}  "
            f"P {s['macro_precision']:.4f}  R {s['macro_recall']:.4f}  "
            f"singletons {s['singleton_macro_f0.5']:.4f}  matched {s['matched_macro_f0.5']:.4f}  "
            f"by country {s['by_country']}"
        )

    text = "\n".join(lines)
    (STAGE2_DIR / "stage2_report.txt").write_text(text, encoding="utf-8")
    with open(STAGE2_DIR / "baseline_scores.json", "w", encoding="utf-8") as f:
        json.dump(baseline_scores, f, indent=2)
    return text


def main():
    results = []

    test_entity_cases(results)
    test_macro_averaging(results)
    test_against_sklearn(results)
    test_issue_detection(results)
    print("\nAll metric tests passed.")

    ground_truth = load_ground_truth(VALIDATION_GT)
    s1 = pd.read_csv(S1_PATH, sep="\t", engine="pyarrow", usecols=["entity_id", "country"])
    countries = dict(zip(s1["entity_id"], s1["country"]))
    del s1

    with open(PROFILE_PATH, encoding="utf-8") as f:
        profile = json.load(f)
    check("validation_file_matches_profile", len(ground_truth), profile["validation_entities"], results)

    baseline_scores = run_baselines(ground_truth, countries, results)
    test_file_round_trip(ground_truth, results)

    write_report(profile, results, baseline_scores)

    print(f"\n{len(results)}/{len(results)} checks passed.")
    print(f"Saved: {STAGE2_DIR / 'stage2_report.txt'}")
    print(f"Saved: {STAGE2_DIR / 'baseline_scores.json'}")
    print("\nStage 2 evaluation engine is operational.")


if __name__ == "__main__":
    main()
