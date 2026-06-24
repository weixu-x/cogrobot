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
    serial: dict[str, list[int]] = defaultdict(list)
    taxonomy = Counter({"substitution": 0, "omission": 0, "insertion": 0, "transposition": 0})
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

        ref_blocks = block_tokens_before_eos(ref, eos_token_id)
        pred_blocks = block_tokens_before_eos(pred_row, eos_token_id)
        ref_with_eos = trim_at_eos(ref, eos_token_id)
        pred_with_eos = trim_at_eos(pred_row, eos_token_id)
        exact = int(pred_with_eos == ref_with_eos)
        full_correct += exact
        length_key = str(len(ref_blocks))
        per_length[length_key].append(exact)
        for position, token in enumerate(ref_blocks):
            value = int(position < len(pred_blocks) and pred_blocks[position] == token)
            serial[str(position)].append(value)

        eos_index = len(ref_blocks)
        eos_hit = int(eos_index < len(pred_row) and pred_row[eos_index] == int(eos_token_id))
        eos_correct += eos_hit
        pred_len = _predicted_length(pred_row, eos_token_id)
        len_hit = int(pred_len == len(ref_blocks))
        length_correct += len_hit
        errors = _classify_errors(ref_blocks, pred_blocks)
        taxonomy.update(errors)
        rows.append(
            {
                "index": row_index,
                "target": ref_with_eos,
                "prediction": pred_with_eos,
                "target_length": len(ref_blocks),
                "predicted_length": pred_len,
                "token_accuracy": float(sum(matches) / len(ref)) if ref else 0.0,
                "full_sequence_accuracy": float(exact),
                "eos_accuracy": float(eos_hit),
                "predicted_length_accuracy": float(len_hit),
                "errors": errors,
            }
        )

    count = max(1, target_tokens.shape[0])
    return {
        "token_accuracy": float(correct_positions / max(total_positions, 1)),
        "full_sequence_accuracy": float(full_correct / count),
        "eos_accuracy": float(eos_correct / count),
        "predicted_length_accuracy": float(length_correct / count),
        "per_length_accuracy": {
            key: {"accuracy": float(np.mean(values)), "count": int(len(values))}
            for key, values in sorted(per_length.items(), key=lambda item: int(item[0]))
        },
        "serial_position_accuracy": {
            key: {"accuracy": float(np.mean(values)), "count": int(len(values))}
            for key, values in sorted(serial.items(), key=lambda item: int(item[0]))
        },
        "error_taxonomy": {key: int(taxonomy[key]) for key in sorted(taxonomy)},
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
