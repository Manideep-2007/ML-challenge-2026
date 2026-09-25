from typing import Iterable


BETA = 0.5


def fbeta_score(
    precision: float,
    recall: float,
    beta: float = BETA,
) -> float:
    """F-beta; beta=0.5 weights precision more than recall."""

    denominator = (beta ** 2) * precision + recall

    if denominator == 0:
        return 0.0

    return (1 + beta ** 2) * precision * recall / denominator


def entity_metrics(
    predicted_ids: Iterable[str],
    ground_truth_ids: Iterable[str],
    beta: float = BETA,
) -> dict:
    """Metrics for one Source 1 entity. IDs are compared as sets."""

    predicted = set(predicted_ids)
    ground_truth = set(ground_truth_ids)

    # Challenge rule: a correctly predicted singleton scores 1.0.
    if not predicted and not ground_truth:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f_beta": 1.0,
        }

    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f_beta": fbeta_score(precision, recall, beta),
    }


def macro_fbeta(
    predictions: dict[str, Iterable[str]],
    ground_truth: dict[str, Iterable[str]],
    beta: float = BETA,
) -> dict:
    """
    Entity-level F-beta, macro-averaged over every Source 1 entity in
    ground_truth. Entities missing from predictions count as an empty
    prediction; prediction keys absent from ground_truth are ignored.
    """

    scores = []
    precisions = []
    recalls = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for entity_id, truth_ids in ground_truth.items():

        result = entity_metrics(
            predictions.get(entity_id, []),
            truth_ids,
            beta,
        )

        scores.append(result["f_beta"])
        precisions.append(result["precision"])
        recalls.append(result["recall"])

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

    n = len(scores)

    micro_precision = (
        total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    )
    micro_recall = (
        total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    )

    return {
        "macro_f_beta": sum(scores) / n if n else 0.0,
        "macro_precision": sum(precisions) / n if n else 0.0,
        "macro_recall": sum(recalls) / n if n else 0.0,
        "micro_f_beta": fbeta_score(micro_precision, micro_recall, beta),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "num_entities": n,
    }
