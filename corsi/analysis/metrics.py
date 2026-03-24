"""Metrics and error analysis for the minimal Corsi sequence model."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence


def token_accuracy(predictions, targets, mask) -> float:
    correct = ((predictions == targets) & mask).sum().item()
    total = mask.sum().item()
    return float(correct / total) if total > 0 else 0.0


def full_sequence_accuracy(predictions, targets, mask) -> float:
    if predictions.size(0) == 0:
        return 0.0
    sequence_correct = ((predictions == targets) | (~mask)).all(dim=1)
    return float(sequence_correct.float().mean().item())


def accuracy_by_length(predictions, targets, lengths, mask) -> Dict[int, float]:
    per_length: Dict[int, List[float]] = defaultdict(list)
    sequence_correct = ((predictions == targets) | (~mask)).all(dim=1)
    for index, length in enumerate(lengths.tolist()):
        per_length[int(length)].append(float(sequence_correct[index].item()))
    return {length: sum(values) / len(values) for length, values in sorted(per_length.items())}


def estimated_span(length_accuracy: Dict[int, float], threshold: float = 0.5) -> int:
    valid_lengths = [length for length, acc in length_accuracy.items() if acc >= threshold]
    return max(valid_lengths) if valid_lengths else 0


def classify_sequence_error(target_sequence: Sequence[int], predicted_sequence: Sequence[int]) -> str:
    target_list = list(target_sequence)
    predicted_list = list(predicted_sequence)

    if predicted_list == target_list:
        return "correct"
    if len(predicted_list) < len(target_list):
        return "early_collapse"
    if any(
        predicted_list[index] == predicted_list[index - 1]
        for index in range(1, len(predicted_list))
    ):
        return "repeated_block"
    if Counter(predicted_list) == Counter(target_list):
        return "order_error"
    return "wrong_block"


def error_type_breakdown(predictions, targets, lengths) -> Dict[str, float]:
    counts: Counter = Counter()
    total = len(lengths)
    for prediction, target, length in zip(predictions.tolist(), targets.tolist(), lengths.tolist()):
        clipped_prediction = prediction[:length]
        clipped_target = target[:length]
        counts[classify_sequence_error(clipped_target, clipped_prediction)] += 1
    if total == 0:
        return {}
    return {key: value / total for key, value in sorted(counts.items())}


def summarize_sequence_metrics(predictions, targets, lengths, mask) -> Dict[str, object]:
    by_length = accuracy_by_length(predictions, targets, lengths, mask)
    return {
        "token_accuracy": token_accuracy(predictions, targets, mask),
        "full_sequence_accuracy": full_sequence_accuracy(predictions, targets, mask),
        "accuracy_by_length": by_length,
        "estimated_span": estimated_span(by_length),
        "error_breakdown": error_type_breakdown(predictions, targets, lengths),
    }
