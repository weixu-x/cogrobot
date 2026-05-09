"""Attention diagnostics for visual Corsi recall experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


def _local_mass(weights: np.ndarray, step: int, window: int, valid_length: int) -> float:
    left = max(0, step - window)
    right = min(valid_length, step + window + 1)
    return float(weights[left:right].sum())


def analyze_attention_weights(
    attention_weights,
    lengths,
    *,
    encoder_lengths: Optional[object] = None,
) -> List[Dict[str, float | int]]:
    weights = np.asarray(attention_weights)
    lengths_np = np.asarray(lengths).astype(int)
    encoder_lengths_np = (
        np.asarray(encoder_lengths).astype(int) if encoder_lengths is not None else lengths_np
    )
    rows: List[Dict[str, float | int]] = []

    for trial_index in range(weights.shape[0]):
        output_length = int(lengths_np[trial_index])
        encoder_length = int(encoder_lengths_np[trial_index])
        valid_length = max(1, min(encoder_length, weights.shape[2]))
        for step in range(min(output_length, weights.shape[1])):
            step_weights = weights[trial_index, step, :valid_length]
            total = step_weights.sum()
            if total <= 0:
                normalized = np.full(valid_length, 1.0 / valid_length)
            else:
                normalized = step_weights / total
            entropy = float(-(normalized * np.log(np.clip(normalized, 1e-8, None))).sum())
            entropy_norm = float(entropy / np.log(valid_length)) if valid_length > 1 else 0.0
            peak = int(normalized.argmax())
            rows.append(
                {
                    "trial_index": int(trial_index),
                    "length": output_length,
                    "step": int(step),
                    "attention_entropy": entropy,
                    "attention_entropy_norm": entropy_norm,
                    "attention_peak_index": peak,
                    "attention_peak_displacement": int(peak - step),
                    "diagonal_mass": float(normalized[step]) if step < valid_length else 0.0,
                    "local_mass_window_1": _local_mass(normalized, step, 1, valid_length),
                    "local_mass_window_2": _local_mass(normalized, step, 2, valid_length),
                }
            )
    return rows


def summarize_attention_rows(rows: List[Dict[str, float | int]]) -> Dict[str, float]:
    if not rows:
        return {
            "mean_attention_entropy": 0.0,
            "mean_attention_entropy_norm": 0.0,
            "mean_peak_displacement_abs": 0.0,
            "mean_peak_displacement": 0.0,
            "local_mass_w1": 0.0,
            "local_mass_w2": 0.0,
            "diagonal_mass": 0.0,
        }
    return {
        "mean_attention_entropy": float(np.mean([row["attention_entropy"] for row in rows])),
        "mean_attention_entropy_norm": float(np.mean([row["attention_entropy_norm"] for row in rows])),
        "mean_peak_displacement_abs": float(np.mean([abs(row["attention_peak_displacement"]) for row in rows])),
        "mean_peak_displacement": float(np.mean([row["attention_peak_displacement"] for row in rows])),
        "local_mass_w1": float(np.mean([row["local_mass_window_1"] for row in rows])),
        "local_mass_w2": float(np.mean([row["local_mass_window_2"] for row in rows])),
        "diagonal_mass": float(np.mean([row["diagonal_mass"] for row in rows])),
    }


def summarize_attention_by_length(rows: List[Dict[str, float | int]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[int, List[Dict[str, float | int]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["length"])].append(row)
    return {str(length): summarize_attention_rows(items) for length, items in sorted(grouped.items())}
