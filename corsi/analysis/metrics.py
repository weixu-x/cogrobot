"""Metrics and error analysis for Corsi sequence recall experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence


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


def per_length_metrics(
    predictions,
    targets,
    target_lengths,
    mask,
    *,
    min_length: int = 2,
    max_length: int = 6,
) -> Dict[str, Dict[str, float]]:
    per_length: Dict[str, Dict[str, float]] = {}
    sequence_correct = ((predictions == targets) | (~mask)).all(dim=1)

    for length in range(min_length, max_length + 1):
        length_mask = target_lengths == length
        if int(length_mask.sum().item()) == 0:
            per_length[str(length)] = {"token_acc": 0.0, "full_seq_acc": 0.0}
            continue

        length_predictions = predictions[length_mask]
        length_targets = targets[length_mask]
        length_token_mask = mask[length_mask]
        per_length[str(length)] = {
            "token_acc": token_accuracy(length_predictions, length_targets, length_token_mask),
            "full_seq_acc": float(sequence_correct[length_mask].float().mean().item()),
        }

    return per_length


def serial_position_accuracy(
    predictions,
    targets,
    target_lengths,
    *,
    min_length: int = 2,
    max_length: int = 6,
) -> Dict[str, List[float]]:
    serial_acc: Dict[str, List[float]] = {}

    for length in range(min_length, max_length + 1):
        length_mask = target_lengths == length
        if int(length_mask.sum().item()) == 0:
            serial_acc[str(length)] = [0.0] * length
            continue

        length_predictions = predictions[length_mask, :length]
        length_targets = targets[length_mask, :length]
        correct = (length_predictions == length_targets).float().mean(dim=0)
        serial_acc[str(length)] = [float(value.item()) for value in correct]

    return serial_acc


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


def error_type_breakdown_by_length(
    predictions,
    targets,
    target_lengths,
    *,
    min_length: int = 2,
    max_length: int = 6,
) -> Dict[str, Dict[str, float]]:
    per_length: Dict[str, Dict[str, float]] = {}
    for length in range(min_length, max_length + 1):
        length_mask = target_lengths == length
        if int(length_mask.sum().item()) == 0:
            per_length[str(length)] = {}
            continue
        per_length[str(length)] = error_type_breakdown(
            predictions[length_mask],
            targets[length_mask],
            target_lengths[length_mask],
        )
    return per_length


def _first_error_position(target_sequence: Sequence[int], predicted_sequence: Sequence[int]) -> int | None:
    for position, (target_token, predicted_token) in enumerate(
        zip(target_sequence, predicted_sequence),
        start=1,
    ):
        if target_token != predicted_token:
            return position
    return None


def _repeat_error_count(target_sequence: Sequence[int], predicted_sequence: Sequence[int]) -> int:
    seen_predictions = set()
    repeat_count = 0
    for index, predicted_token in enumerate(predicted_sequence):
        if predicted_token in seen_predictions and predicted_token != target_sequence[index]:
            repeat_count += 1
        seen_predictions.add(predicted_token)
    return repeat_count


def _mean_abs_displacement(target_sequence: Sequence[int], predicted_sequence: Sequence[int]) -> float:
    target_length = len(target_sequence)
    remaining_prediction_positions: Dict[int, List[int]] = defaultdict(list)
    for position, predicted_token in enumerate(predicted_sequence, start=1):
        remaining_prediction_positions[predicted_token].append(position)

    displacements: List[float] = []
    for target_position, target_token in enumerate(target_sequence, start=1):
        predicted_positions = remaining_prediction_positions[target_token]
        if predicted_positions:
            predicted_position = predicted_positions.pop(0)
            displacements.append(float(abs(predicted_position - target_position)))
        else:
            displacements.append(float(target_length))

    return sum(displacements) / len(displacements) if displacements else 0.0


def _transposition_like(target_sequence: Sequence[int], predicted_sequence: Sequence[int]) -> bool:
    return list(target_sequence) != list(predicted_sequence) and Counter(target_sequence) == Counter(predicted_sequence)


def aggregate_error_analysis(predictions, targets, target_lengths) -> Dict[str, float | None]:
    total_sequences = int(target_lengths.numel())
    if total_sequences == 0:
        return {
            "exact_match_rate": 0.0,
            "mean_first_error_pos": None,
            "mean_abs_displacement": 0.0,
            "repeat_error_rate": 0.0,
            "mean_repeat_error_count": 0.0,
            "transposition_like_rate": 0.0,
        }

    exact_matches = 0
    first_error_positions: List[float] = []
    displacement_values: List[float] = []
    repeat_sequences = 0
    repeat_error_total = 0
    transposition_sequences = 0

    for prediction, target, length in zip(predictions.tolist(), targets.tolist(), target_lengths.tolist()):
        clipped_prediction = prediction[:length]
        clipped_target = target[:length]

        if clipped_prediction == clipped_target:
            exact_matches += 1
        else:
            first_error = _first_error_position(clipped_target, clipped_prediction)
            if first_error is not None:
                first_error_positions.append(float(first_error))

        repeat_errors = _repeat_error_count(clipped_target, clipped_prediction)
        repeat_error_total += repeat_errors
        if repeat_errors > 0:
            repeat_sequences += 1

        if _transposition_like(clipped_target, clipped_prediction):
            transposition_sequences += 1

        displacement_values.append(_mean_abs_displacement(clipped_target, clipped_prediction))

    return {
        "exact_match_rate": exact_matches / total_sequences,
        "mean_first_error_pos": (
            sum(first_error_positions) / len(first_error_positions)
            if first_error_positions
            else None
        ),
        "mean_abs_displacement": (
            sum(displacement_values) / len(displacement_values)
            if displacement_values
            else 0.0
        ),
        "repeat_error_rate": repeat_sequences / total_sequences,
        "mean_repeat_error_count": repeat_error_total / total_sequences,
        "transposition_like_rate": transposition_sequences / total_sequences,
    }


def aggregate_error_analysis_by_length(
    predictions,
    targets,
    target_lengths,
    *,
    min_length: int = 2,
    max_length: int = 6,
) -> Dict[str, Dict[str, float | None]]:
    per_length: Dict[str, Dict[str, float | None]] = {}
    for length in range(min_length, max_length + 1):
        length_mask = target_lengths == length
        if int(length_mask.sum().item()) == 0:
            per_length[str(length)] = {
                "exact_match_rate": 0.0,
                "mean_first_error_pos": None,
                "mean_abs_displacement": 0.0,
                "repeat_error_rate": 0.0,
                "mean_repeat_error_count": 0.0,
                "transposition_like_rate": 0.0,
            }
            continue
        per_length[str(length)] = aggregate_error_analysis(
            predictions[length_mask],
            targets[length_mask],
            target_lengths[length_mask],
        )
    return per_length


def summarize_sequence_metrics(
    predictions,
    targets,
    lengths,
    mask,
    *,
    min_length: int = 2,
    max_length: int = 6,
) -> Dict[str, object]:
    by_length = accuracy_by_length(predictions, targets, lengths, mask)
    per_length = per_length_metrics(
        predictions,
        targets,
        lengths,
        mask,
        min_length=min_length,
        max_length=max_length,
    )
    serial_acc = serial_position_accuracy(
        predictions,
        targets,
        lengths,
        min_length=min_length,
        max_length=max_length,
    )
    error_analysis = aggregate_error_analysis(predictions, targets, lengths)
    error_analysis_by_length = aggregate_error_analysis_by_length(
        predictions,
        targets,
        lengths,
        min_length=min_length,
        max_length=max_length,
    )
    error_breakdown = error_type_breakdown(predictions, targets, lengths)
    error_breakdown_by_length = error_type_breakdown_by_length(
        predictions,
        targets,
        lengths,
        min_length=min_length,
        max_length=max_length,
    )
    return {
        "token_accuracy": token_accuracy(predictions, targets, mask),
        "full_sequence_accuracy": full_sequence_accuracy(predictions, targets, mask),
        "per_length": per_length,
        "serial_position_acc": serial_acc,
        "serial_position_accuracy_by_length": serial_acc,
        "estimated_span": estimated_span(by_length),
        "error_analysis": error_analysis,
        "error_analysis_by_length": error_analysis_by_length,
        "error_breakdown": error_breakdown,
        "error_breakdown_by_length": error_breakdown_by_length,
        "mean_first_error_pos_by_length": {
            length: stats["mean_first_error_pos"]
            for length, stats in error_analysis_by_length.items()
        },
        "transposition_like_rate_by_length": {
            length: stats["transposition_like_rate"]
            for length, stats in error_analysis_by_length.items()
        },
        "repeat_error_rate_by_length": {
            length: stats["repeat_error_rate"]
            for length, stats in error_analysis_by_length.items()
        },
        # Backward-compatible alias retained for older consumers.
        "accuracy_by_length": by_length,
    }
