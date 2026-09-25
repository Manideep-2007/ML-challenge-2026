from pathlib import Path

import pandas as pd

from .metrics import BETA, macro_fbeta


def parse_match_ids(value) -> list[str]:
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [x.strip() for x in value.split(",") if x.strip()]


def load_id_lists(
    path: Path,
    list_column: str = "matched_entity_ids",
    key_column: str = "source1_entity_id",
) -> dict[str, list[str]]:
    """
    Load a `source1_entity_id <tab> comma-separated IDs` file:
    ground truth, matching_results.tsv, or candidate_pairs.tsv
    (list_column="candidate_entity_ids").
    """

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    return dict(zip(df[key_column], map(parse_match_ids, df[list_column])))


def load_ground_truth(path: Path) -> dict[str, list[str]]:
    return load_id_lists(path, "matched_entity_ids")


def load_predictions(path: Path) -> dict[str, list[str]]:
    return load_id_lists(path, "matched_entity_ids")


def prediction_issues(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
) -> dict:
    # The metric uses sets, so these would not change the score,
    # but the official submission rules reject duplicates and S1 self-matches.
    return {
        "lists_with_duplicate_ids": sum(
            len(ids) != len(set(ids)) for ids in predictions.values()
        ),
        "ids_with_invalid_prefix": sum(
            not (i.startswith("S2-") or i.startswith("S3-"))
            for ids in predictions.values()
            for i in ids
        ),
        "entities_missing_from_predictions": sum(
            entity_id not in predictions for entity_id in ground_truth
        ),
        "prediction_entities_not_in_ground_truth": sum(
            entity_id not in ground_truth for entity_id in predictions
        ),
    }


def evaluate_predictions(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    beta: float = BETA,
    groups: dict[str, str] | None = None,
) -> dict:
    """
    Competition score plus breakdowns.

    groups: optional source1_entity_id -> label (e.g. country) for a
    per-label macro F-beta.
    """

    result = macro_fbeta(predictions, ground_truth, beta)

    singletons = {k: v for k, v in ground_truth.items() if not v}
    matched = {k: v for k, v in ground_truth.items() if v}

    result["by_truth_type"] = {
        "singleton": macro_fbeta(predictions, singletons, beta),
        "matched": macro_fbeta(predictions, matched, beta),
    }

    if groups is not None:
        labels = sorted({groups.get(k, "<unknown>") for k in ground_truth})
        result["by_group"] = {
            label: macro_fbeta(
                predictions,
                {k: v for k, v in ground_truth.items()
                 if groups.get(k, "<unknown>") == label},
                beta,
            )
            for label in labels
        }

    result["issues"] = prediction_issues(predictions, ground_truth)

    return result
