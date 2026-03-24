"""Sequence and trial generation utilities for the Corsi task."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


BlockLayout = Dict[int, Tuple[float, float]]


@dataclass(frozen=True)
class CorsiTrial:
    """Structured metadata for a single coordinate-based Corsi trial."""

    trial_id: str
    seq_len: int
    sequence: List[int]
    coords: List[Tuple[float, float]]
    delta_coords: List[Tuple[float, float]]
    layout: BlockLayout
    mode: str


def canonical_board_layout(spacing: float = 1.0) -> BlockLayout:
    """Returns a simple row-major 3x3 Corsi layout centered around the origin."""

    coordinates = (-spacing, 0.0, spacing)
    layout: BlockLayout = {}
    index = 0
    for y in reversed(coordinates):
        for x in coordinates:
            layout[index] = (x, y)
            index += 1
    return layout


def _get_rng(rng: Optional[random.Random] = None) -> random.Random:
    return rng if rng is not None else random.Random()


def generate_block_sequence(
    length: int,
    num_blocks: int = 9,
    rng: Optional[random.Random] = None,
    forbid_immediate_repeats: bool = True,
    mode: str = "forward",
) -> List[int]:
    """Generates a block sequence with optional forward / backward variants."""

    if length < 1:
        raise ValueError("length must be at least 1")
    if num_blocks < 2 and forbid_immediate_repeats and length > 1:
        raise ValueError("num_blocks must be at least 2 when immediate repeats are forbidden")
    if mode not in {"forward", "backward"}:
        raise ValueError("mode must be 'forward' or 'backward'")

    generator = _get_rng(rng)
    sequence: List[int] = []
    for _ in range(length):
        candidates = list(range(num_blocks))
        if forbid_immediate_repeats and sequence:
            candidates.remove(sequence[-1])
        sequence.append(generator.choice(candidates))

    if mode == "backward":
        return list(reversed(sequence))
    return sequence


def sequence_to_coords(sequence: Sequence[int], layout: Mapping[int, Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Maps block indices to 2D coordinates."""

    coords: List[Tuple[float, float]] = []
    for block_id in sequence:
        if block_id not in layout:
            raise KeyError(f"Block id {block_id} is missing from the layout")
        coords.append(tuple(layout[block_id]))
    return coords


def compute_delta_coords(coords: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Computes per-step relative displacement with a zero delta at the first step."""

    if not coords:
        return []

    deltas: List[Tuple[float, float]] = [(0.0, 0.0)]
    for prev, current in zip(coords[:-1], coords[1:]):
        deltas.append((current[0] - prev[0], current[1] - prev[1]))
    return deltas


def build_coordinate_features(
    coords: Sequence[Tuple[float, float]],
    delta_coords: Sequence[Tuple[float, float]],
    feature_mode: str = "xydxdy",
) -> List[List[float]]:
    """Builds coordinate features for the minimal memory model."""

    if len(coords) != len(delta_coords):
        raise ValueError("coords and delta_coords must have the same length")
    if feature_mode not in {"xy", "xydxdy"}:
        raise ValueError("feature_mode must be 'xy' or 'xydxdy'")

    features: List[List[float]] = []
    for (x, y), (dx, dy) in zip(coords, delta_coords):
        if feature_mode == "xy":
            features.append([x, y])
        else:
            features.append([x, y, dx, dy])
    return features


def generate_coordinate_trial(
    seq_len: int,
    layout: Optional[BlockLayout] = None,
    rng: Optional[random.Random] = None,
    mode: str = "forward",
    trial_id: Optional[str] = None,
    forbid_immediate_repeats: bool = True,
) -> CorsiTrial:
    """Generates one coordinate-based Corsi trial."""

    board_layout = dict(layout) if layout is not None else canonical_board_layout()
    sequence = generate_block_sequence(
        length=seq_len,
        num_blocks=len(board_layout),
        rng=rng,
        forbid_immediate_repeats=forbid_immediate_repeats,
        mode=mode,
    )
    coords = sequence_to_coords(sequence, board_layout)
    delta_coords = compute_delta_coords(coords)
    return CorsiTrial(
        trial_id=trial_id or f"trial_len{seq_len}_{mode}_{'-'.join(map(str, sequence))}",
        seq_len=seq_len,
        sequence=sequence,
        coords=coords,
        delta_coords=delta_coords,
        layout=board_layout,
        mode=mode,
    )


def generate_trial_collection(
    num_trials: int,
    seq_len_range: Tuple[int, int] = (2, 9),
    layout: Optional[BlockLayout] = None,
    seed: Optional[int] = None,
    mode: str = "forward",
) -> List[CorsiTrial]:
    """Generates a list of trials for dataset construction."""

    min_len, max_len = seq_len_range
    if min_len < 1 or max_len < min_len:
        raise ValueError("Invalid seq_len_range")

    rng = random.Random(seed)
    trials: List[CorsiTrial] = []
    for index in range(num_trials):
        seq_len = rng.randint(min_len, max_len)
        trials.append(
            generate_coordinate_trial(
                seq_len=seq_len,
                layout=layout,
                rng=rng,
                mode=mode,
                trial_id=f"trial_{index:06d}",
            )
        )
    return trials
