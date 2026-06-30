"""Autonomous sequence metrics and causal sanity checks for Corsi V2."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def _is_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch")


def _to_numpy(value: Any) -> np.ndarray:
    if _is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def tokens_from_logits(logits_or_tokens: Any) -> np.ndarray:
    values = _to_numpy(logits_or_tokens)
    if values.ndim >= 3:
        return np.argmax(values, axis=-1).astype(np.int64)
    return values.astype(np.int64)


def trim_at_eos(tokens: Sequence[int], eos_token_id: int) -> list[int]:
    result: list[int] = []
    for token in tokens:
        value = int(token)
        result.append(value)
        if value == int(eos_token_id):
            break
    return result


def block_tokens_before_eos(tokens: Sequence[int], eos_token_id: int) -> list[int]:
    result: list[int] = []
    for token in tokens:
        value = int(token)
        if value == int(eos_token_id):
            break
        result.append(value)
    return result


def _valid_target_tokens(
    target_row: np.ndarray,
    mask_row: np.ndarray | None,
    *,
    ignore_index: int,
) -> list[int]:
    if mask_row is None:
        return [int(value) for value in target_row.tolist() if int(value) != int(ignore_index)]
    return [int(value) for value, keep in zip(target_row.tolist(), mask_row.tolist()) if bool(keep)]


def _predicted_length(pred_row: Sequence[int], eos_token_id: int) -> int:
    for index, token in enumerate(pred_row):
        if int(token) == int(eos_token_id):
            return index
    return len(pred_row)


def _classify_errors(reference: Sequence[int], prediction: Sequence[int]) -> dict[str, int]:
    ref = [int(value) for value in reference]
    pred = [int(value) for value in prediction]
    zero = (0, 0, 0, 0, 0)
    rows = len(ref) + 1
    cols = len(pred) + 1
    dp: list[list[tuple[int, int, int, int, int]]] = [[zero for _ in range(cols)] for _ in range(rows)]
    for row in range(1, rows):
        previous = dp[row - 1][0]
        dp[row][0] = (
            previous[0] + 1,
            previous[1],
            previous[2] + 1,
            previous[3],
            previous[4],
        )
    for col in range(1, cols):
        previous = dp[0][col - 1]
        dp[0][col] = (
            previous[0] + 1,
            previous[1],
            previous[2],
            previous[3] + 1,
            previous[4],
        )
    for row in range(1, rows):
        for col in range(1, cols):
            candidates = []
            keep_or_sub = dp[row - 1][col - 1]
            if ref[row - 1] == pred[col - 1]:
                candidates.append(keep_or_sub)
            else:
                candidates.append(
                    (
                        keep_or_sub[0] + 1,
                        keep_or_sub[1] + 1,
                        keep_or_sub[2],
                        keep_or_sub[3],
                        keep_or_sub[4],
                    )
                )
            delete = dp[row - 1][col]
            candidates.append((delete[0] + 1, delete[1], delete[2] + 1, delete[3], delete[4]))
            insert = dp[row][col - 1]
            candidates.append((insert[0] + 1, insert[1], insert[2], insert[3] + 1, insert[4]))
            if (
                row >= 2
                and col >= 2
                and ref[row - 2] == pred[col - 1]
                and ref[row - 1] == pred[col - 2]
            ):
                swap = dp[row - 2][col - 2]
                candidates.append((swap[0] + 1, swap[1], swap[2], swap[3], swap[4] + 1))
            dp[row][col] = min(candidates)
    _, substitutions, omissions, insertions, transpositions = dp[-1][-1]
    return {
        "substitution": int(substitutions),
        "omission": int(omissions),
        "insertion": int(insertions),
        "transposition": int(transpositions),
    }


def _mean(values: Sequence[float | int]) -> float:
    return float(np.mean(values)) if values else 0.0


def compute_sequence_metrics(
    predictions: Any,
    targets: Any,
    target_mask: Any | None = None,
    *,
    eos_token_id: int = 9,
    ignore_index: int = -100,
) -> dict[str, Any]:
    """Compute autonomous Corsi recall metrics from decoded tokens or logits."""

    pred_tokens = tokens_from_logits(predictions)
    target_tokens = _to_numpy(targets).astype(np.int64)
    if pred_tokens.ndim == 1:
        pred_tokens = pred_tokens.reshape(1, -1)
    if target_tokens.ndim == 1:
        target_tokens = target_tokens.reshape(1, -1)
    mask = None if target_mask is None else _to_numpy(target_mask).astype(bool)
    if mask is not None and mask.ndim == 1:
        mask = mask.reshape(1, -1)

    total_positions = 0
    correct_positions = 0
    full_correct = 0
    eos_correct = 0
    length_correct = 0
    per_length: dict[str, list[int]] = defaultdict(list)
    per_length_token: dict[str, list[float]] = defaultdict(list)
    per_length_eos: dict[str, list[int]] = defaultdict(list)
    per_length_predicted_length: dict[str, list[int]] = defaultdict(list)
    per_length_duplicate: dict[str, list[int]] = defaultdict(list)
    per_length_unique_blocks: dict[str, list[int]] = defaultdict(list)
    per_length_set_overlap: dict[str, list[float]] = defaultdict(list)
    per_length_error_taxonomy: dict[str, Counter[str]] = defaultdict(
        lambda: Counter({"substitution": 0, "omission": 0, "insertion": 0, "transposition": 0})
    )
    serial: dict[str, list[int]] = defaultdict(list)
    taxonomy = Counter({"substitution": 0, "omission": 0, "insertion": 0, "transposition": 0})
    failure_taxonomy = Counter(
        {
            "exact": 0,
            "eos_error": 0,
            "length_error": 0,
            "duplicate_error": 0,
            "substitution_error": 0,
            "omission_error": 0,
            "insertion_error": 0,
            "transposition_error": 0,
        }
    )
    duplicate_sequence_flags: list[int] = []
    duplicate_counts: list[int] = []
    duplicate_rates: list[float] = []
    unique_predicted_counts: list[int] = []
    set_jaccards: list[float] = []
    set_recalls: list[float] = []
    set_precisions: list[float] = []
    first_position_correct: list[int] = []
    middle_position_correct: list[int] = []
    last_position_correct: list[int] = []
    transposition_distance_counts: Counter[int] = Counter()
    rows: list[dict[str, Any]] = []

    for row_index in range(target_tokens.shape[0]):
        target_row = target_tokens[row_index]
        mask_row = None if mask is None else mask[row_index]
        ref = _valid_target_tokens(target_row, mask_row, ignore_index=ignore_index)
        pred_row = [int(value) for value in pred_tokens[row_index].tolist()]
        comparable_pred = pred_row[: len(ref)]
        matches = [int(p == t) for p, t in zip(comparable_pred, ref)]
        total_positions += len(ref)
        correct_positions += int(sum(matches))
        row_token_accuracy = float(sum(matches) / len(ref)) if ref else 0.0

        ref_blocks = block_tokens_before_eos(ref, eos_token_id)
        pred_blocks = block_tokens_before_eos(pred_row, eos_token_id)
        ref_with_eos = trim_at_eos(ref, eos_token_id)
        pred_with_eos = trim_at_eos(pred_row, eos_token_id)
        exact = int(pred_with_eos == ref_with_eos)
        full_correct += exact
        length_key = str(len(ref_blocks))
        per_length[length_key].append(exact)
        per_length_token[length_key].append(row_token_accuracy)
        for position, token in enumerate(ref_blocks):
            value = int(position < len(pred_blocks) and pred_blocks[position] == token)
            serial[str(position)].append(value)
            if position == 0:
                first_position_correct.append(value)
            elif position == len(ref_blocks) - 1:
                last_position_correct.append(value)
            else:
                middle_position_correct.append(value)
        target_positions = {int(token): index for index, token in enumerate(ref_blocks)}
        for position, token in enumerate(pred_blocks[: len(ref_blocks)]):
            target_position = target_positions.get(int(token))
            if target_position is None or target_position == position:
                continue
            transposition_distance_counts[abs(int(position) - int(target_position))] += 1

        eos_index = len(ref_blocks)
        eos_hit = int(eos_index < len(pred_row) and pred_row[eos_index] == int(eos_token_id))
        eos_correct += eos_hit
        per_length_eos[length_key].append(eos_hit)
        pred_len = _predicted_length(pred_row, eos_token_id)
        len_hit = int(pred_len == len(ref_blocks))
        length_correct += len_hit
        per_length_predicted_length[length_key].append(len_hit)
        errors = _classify_errors(ref_blocks, pred_blocks)
        taxonomy.update(errors)
        per_length_error_taxonomy[length_key].update(errors)

        duplicate_count = int(len(pred_blocks) - len(set(pred_blocks)))
        duplicate_flag = int(duplicate_count > 0)
        duplicate_rate = float(duplicate_count / max(len(pred_blocks), 1))
        unique_predicted_blocks = int(len(set(pred_blocks)))
        target_set = set(ref_blocks)
        predicted_set = set(pred_blocks)
        set_intersection = len(target_set & predicted_set)
        set_union = len(target_set | predicted_set)
        set_overlap_jaccard = float(set_intersection / set_union) if set_union else 0.0
        set_recall = float(set_intersection / len(target_set)) if target_set else 0.0
        set_precision = float(set_intersection / len(predicted_set)) if predicted_set else 0.0
        duplicate_sequence_flags.append(duplicate_flag)
        duplicate_counts.append(duplicate_count)
        duplicate_rates.append(duplicate_rate)
        unique_predicted_counts.append(unique_predicted_blocks)
        set_jaccards.append(set_overlap_jaccard)
        set_recalls.append(set_recall)
        set_precisions.append(set_precision)
        per_length_duplicate[length_key].append(duplicate_flag)
        per_length_unique_blocks[length_key].append(unique_predicted_blocks)
        per_length_set_overlap[length_key].append(set_overlap_jaccard)

        failure_flags = {
            "exact": bool(exact),
            "eos_error": not bool(eos_hit),
            "length_error": not bool(len_hit),
            "duplicate_error": bool(duplicate_flag),
            "substitution_error": bool(errors["substitution"]),
            "omission_error": bool(errors["omission"]),
            "insertion_error": bool(errors["insertion"]),
            "transposition_error": bool(errors["transposition"]),
        }
        for key, value in failure_flags.items():
            if value:
                failure_taxonomy[key] += 1
        rows.append(
            {
                "index": row_index,
                "target": ref_with_eos,
                "prediction": pred_with_eos,
                "target_tokens": ref_with_eos,
                "predicted_tokens": pred_with_eos,
                "target_blocks": ref_blocks,
                "predicted_blocks": pred_blocks,
                "target_length": len(ref_blocks),
                "predicted_length": pred_len,
                "token_accuracy": row_token_accuracy,
                "full_sequence_accuracy": float(exact),
                "exact_correct": bool(exact),
                "eos_accuracy": float(eos_hit),
                "eos_correct": bool(eos_hit),
                "predicted_length_accuracy": float(len_hit),
                "length_correct": bool(len_hit),
                "has_duplicate_prediction": bool(duplicate_flag),
                "duplicate_count": duplicate_count,
                "duplicate_rate": duplicate_rate,
                "unique_predicted_blocks": unique_predicted_blocks,
                "target_set_size": int(len(target_set)),
                "predicted_set_size": int(len(predicted_set)),
                "set_intersection": int(set_intersection),
                "set_union": int(set_union),
                "set_overlap_jaccard": set_overlap_jaccard,
                "set_recall": set_recall,
                "set_precision": set_precision,
                "substitutions": int(errors["substitution"]),
                "omissions": int(errors["omission"]),
                "insertions": int(errors["insertion"]),
                "transpositions": int(errors["transposition"]),
                "errors": errors,
                "failure_flags": failure_flags,
            }
        )

    count = max(1, target_tokens.shape[0])
    per_length_metrics = {}
    for key in sorted(per_length, key=int):
        per_length_metrics[key] = {
            "count": int(len(per_length[key])),
            "full_sequence_accuracy": _mean(per_length[key]),
            "token_accuracy": _mean(per_length_token[key]),
            "eos_accuracy": _mean(per_length_eos[key]),
            "predicted_length_accuracy": _mean(per_length_predicted_length[key]),
            "duplicate_sequence_rate": _mean(per_length_duplicate[key]),
            "mean_unique_predicted_blocks": _mean(per_length_unique_blocks[key]),
            "mean_set_overlap_jaccard": _mean(per_length_set_overlap[key]),
            "error_taxonomy": {
                error_key: int(per_length_error_taxonomy[key][error_key])
                for error_key in sorted(per_length_error_taxonomy[key])
            },
        }
    edge_values = first_position_correct + last_position_correct
    adjacent_transpositions = int(transposition_distance_counts.get(1, 0))
    far_transpositions = int(
        sum(count for distance, count in transposition_distance_counts.items() if int(distance) >= 2)
    )
    transposition_total = adjacent_transpositions + far_transpositions
    return {
        "token_accuracy": float(correct_positions / max(total_positions, 1)),
        "full_sequence_accuracy": float(full_correct / count),
        "eos_accuracy": float(eos_correct / count),
        "predicted_length_accuracy": float(length_correct / count),
        "per_length_accuracy": {
            key: {"accuracy": float(np.mean(values)), "count": int(len(values))}
            for key, values in sorted(per_length.items(), key=lambda item: int(item[0]))
        },
        "per_length_exact_accuracy": {
            key: {"accuracy": _mean(values), "count": int(len(values))}
            for key, values in sorted(per_length.items(), key=lambda item: int(item[0]))
        },
        "per_length_token_accuracy": {
            key: {"accuracy": _mean(values), "count": int(len(values))}
            for key, values in sorted(per_length_token.items(), key=lambda item: int(item[0]))
        },
        "per_length_metrics": per_length_metrics,
        "serial_position_accuracy": {
            key: {"accuracy": float(np.mean(values)), "count": int(len(values))}
            for key, values in sorted(serial.items(), key=lambda item: int(item[0]))
        },
        "error_taxonomy": {key: int(taxonomy[key]) for key in sorted(taxonomy)},
        "failure_taxonomy": {
            key: {"count": int(failure_taxonomy[key]), "rate": float(failure_taxonomy[key] / count)}
            for key in sorted(failure_taxonomy)
        },
        "duplicate_metrics": {
            "sequence_duplicate_rate": _mean(duplicate_sequence_flags),
            "mean_duplicate_count": _mean(duplicate_counts),
            "mean_duplicate_rate": _mean(duplicate_rates),
            "mean_unique_predicted_blocks": _mean(unique_predicted_counts),
        },
        "duplicate_sequence_rate": _mean(duplicate_sequence_flags),
        "duplicate_rate": _mean(duplicate_sequence_flags),
        "mean_duplicate_count": _mean(duplicate_counts),
        "mean_duplicate_rate": _mean(duplicate_rates),
        "mean_unique_predicted_blocks": _mean(unique_predicted_counts),
        "set_overlap_metrics": {
            "mean_jaccard": _mean(set_jaccards),
            "mean_recall": _mean(set_recalls),
            "mean_precision": _mean(set_precisions),
        },
        "behavior_sanity": {
            "serial_position_shape": {
                "first_accuracy": _mean(first_position_correct),
                "middle_accuracy": _mean(middle_position_correct),
                "last_accuracy": _mean(last_position_correct),
                "edge_accuracy": _mean(edge_values),
                "u_shape_score": float(_mean(edge_values) - _mean(middle_position_correct)),
                "first_count": int(len(first_position_correct)),
                "middle_count": int(len(middle_position_correct)),
                "last_count": int(len(last_position_correct)),
            },
            "transposition_distance_gradient": {
                "distance_counts": {
                    str(distance): int(count)
                    for distance, count in sorted(transposition_distance_counts.items())
                },
                "adjacent_count": adjacent_transpositions,
                "far_count": far_transpositions,
                "adjacent_fraction": float(adjacent_transpositions / max(transposition_total, 1)),
                "distance_dependent": bool(adjacent_transpositions > far_transpositions),
                "count": int(transposition_total),
            },
        },
        "mean_set_overlap_jaccard": _mean(set_jaccards),
        "sequence_count": int(target_tokens.shape[0]),
        "token_count": int(total_positions),
        "rows": rows,
    }


def clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Mapping):
            cloned[key] = clone_batch(value)
        elif _is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def apply_presentation_order_shuffle(batch: Mapping[str, Any]) -> dict[str, Any]:
    shuffled = clone_batch(batch)
    model_inputs = shuffled.get("model_inputs", {})
    for key in ("images", "segment_mask", "frame_mask"):
        value = model_inputs.get(key)
        if value is None:
            continue
        if _is_tensor(value):
            model_inputs[key] = value.flip(dims=[1])
        else:
            model_inputs[key] = np.flip(np.asarray(value), axis=1).copy()
    return shuffled


def _decode_model_output(output: Any) -> Any:
    if isinstance(output, Mapping):
        for key in ("token_logits", "logits", "tokens", "pred_tokens"):
            if key in output:
                return output[key]
    return output


def call_recall_model(model: Any, batch: Mapping[str, Any], *, intervention: str | None = None) -> Any:
    model_inputs = batch.get("model_inputs", batch)
    if hasattr(model, "decode"):
        return model.decode(model_inputs, intervention=intervention)
    if hasattr(model, "forward_recall"):
        return model.forward_recall(model_inputs, intervention=intervention)
    try:
        return model(**model_inputs, intervention=intervention)
    except TypeError:
        return model(**model_inputs)


def run_causal_sanity_checks(
    model: Any,
    batch: Mapping[str, Any],
    *,
    eos_token_id: int = 9,
    ignore_index: int = -100,
) -> dict[str, Any]:
    """Run required V2 causal interventions through a narrow model hook contract."""

    targets = batch["targets"]["tokens"]
    target_mask = batch["targets"].get("token_mask")
    normal = _decode_model_output(call_recall_model(model, batch, intervention=None))
    checks: dict[str, Any] = {
        "normal": compute_sequence_metrics(
            normal,
            targets,
            target_mask,
            eos_token_id=eos_token_id,
            ignore_index=ignore_index,
        )
    }
    for name in ("memory_zero", "memory_shuffle"):
        predictions = _decode_model_output(call_recall_model(model, batch, intervention=name))
        metrics = compute_sequence_metrics(
            predictions,
            targets,
            target_mask,
            eos_token_id=eos_token_id,
            ignore_index=ignore_index,
        )
        checks[name] = {
            "operation": name,
            "metrics": metrics,
            "token_accuracy_delta": float(
                checks["normal"]["token_accuracy"] - metrics["token_accuracy"]
            ),
            "full_sequence_accuracy_delta": float(
                checks["normal"]["full_sequence_accuracy"] - metrics["full_sequence_accuracy"]
            ),
        }

    shuffled_batch = apply_presentation_order_shuffle(batch)
    shuffled_predictions = _decode_model_output(
        call_recall_model(model, shuffled_batch, intervention="presentation_order_shuffle")
    )
    shuffled_metrics = compute_sequence_metrics(
        shuffled_predictions,
        targets,
        target_mask,
        eos_token_id=eos_token_id,
        ignore_index=ignore_index,
    )
    normal_tokens = tokens_from_logits(normal)
    shuffled_tokens = tokens_from_logits(shuffled_predictions)
    changed = bool(np.any(normal_tokens != shuffled_tokens))
    checks["presentation_order_shuffle"] = {
        "operation": "presentation_order_shuffle",
        "metrics": shuffled_metrics,
        "decoded_order_changed": changed,
    }
    return checks
