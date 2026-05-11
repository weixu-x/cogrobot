"""Detailed error taxonomy utilities for Corsi sequence recall."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence


def first_error_position(target: Sequence[int], pred: Sequence[int]) -> Optional[int]:
    for index, (target_token, pred_token) in enumerate(zip(target, pred)):
        if target_token != pred_token:
            return index
    if len(target) != len(pred):
        return min(len(target), len(pred))
    return None


def adjacent_transposition_count(target: Sequence[int], pred: Sequence[int]) -> int:
    count = 0
    index = 0
    limit = min(len(target), len(pred)) - 1
    while index < limit:
        if pred[index] == target[index + 1] and pred[index + 1] == target[index]:
            count += 1
            index += 2
        else:
            index += 1
    return count


def compute_kendall_tau_on_intersection(target: Sequence[int], pred: Sequence[int]) -> float:
    target_pos = {token: index for index, token in enumerate(target)}
    ordered_pred = [token for token in pred if token in target_pos]
    if len(ordered_pred) < 2:
        return math.nan

    concordant = 0
    discordant = 0
    for left in range(len(ordered_pred)):
        for right in range(left + 1, len(ordered_pred)):
            if target_pos[ordered_pred[left]] < target_pos[ordered_pred[right]]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return float((concordant - discordant) / total) if total else math.nan


def compute_lcs_normalized(target: Sequence[int], pred: Sequence[int]) -> float:
    if not target:
        return 0.0
    rows = len(target) + 1
    cols = len(pred) + 1
    dp = [[0] * cols for _ in range(rows)]
    for row in range(1, rows):
        for col in range(1, cols):
            if target[row - 1] == pred[col - 1]:
                dp[row][col] = dp[row - 1][col - 1] + 1
            else:
                dp[row][col] = max(dp[row - 1][col], dp[row][col - 1])
    return float(dp[-1][-1] / len(target))


def classify_trial(target: Sequence[int], pred: Sequence[int], length: Optional[int] = None) -> Dict[str, object]:
    if length is None:
        length = len(target)
    target_list = list(target[:length])
    pred_list = list(pred[:length])
    target_set = set(target_list)
    target_counts = Counter(target_list)
    pred_counts = Counter(pred_list)

    exact_match = pred_list == target_list
    token_accuracy = (
        sum(int(pred_token == target_token) for pred_token, target_token in zip(pred_list, target_list))
        / length
        if length
        else 0.0
    )
    wrong_block_count = sum(1 for token in pred_list if token not in target_set)
    order_error_count = sum(
        1
        for index, token in enumerate(pred_list)
        if index < len(target_list) and token in target_set and token != target_list[index]
    )
    repeat_error_count = sum(
        max(0, count - target_counts.get(token, 0))
        for token, count in pred_counts.items()
    )
    omissions = [token for token in target_list if token not in pred_counts]

    return {
        "exact_match": exact_match,
        "token_accuracy": float(token_accuracy),
        "first_error_position": first_error_position(target_list, pred_list),
        "wrong_block_count": int(wrong_block_count),
        "order_error_count": int(order_error_count),
        "repeat_error_count": int(repeat_error_count),
        "omission_count": int(len(omissions)),
        "adjacent_transposition_count": int(adjacent_transposition_count(target_list, pred_list)),
        "kendall_tau": compute_kendall_tau_on_intersection(target_list, pred_list),
        "lcs_normalized": compute_lcs_normalized(target_list, pred_list),
    }


def classify_trials(targets, predictions, lengths) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, (target, pred, length) in enumerate(zip(targets, predictions, lengths)):
        row = classify_trial(target, pred, int(length))
        row["trial_index"] = index
        row["length"] = int(length)
        rows.append(row)
    return rows
