"""Sequence and trial generation utilities for the Corsi task."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


BlockLayout = Dict[int, Tuple[float, float]]
BoardSize = Tuple[float, float]

STANDARD_CORSI_BOARD_SIZE: BoardSize = (255.0, 205.0)
STANDARD_CORSI_BLOCK_SIZE: BoardSize = (30.0, 30.0)
STANDARD_CORSI_ROBOSUITE_TARGET_WIDTH: float = 0.36

# Original image coordinates, measured at the bottom-left corner of each 30x30 block.
STANDARD_CORSI_LAYOUT_IMAGE: BlockLayout = {
    0: (130.0, 155.0),  # block 1
    1: (30.0, 145.0),   # block 2
    2: (180.0, 120.0),  # block 3
    3: (70.0, 110.0),   # block 4
    4: (140.0, 90.0),   # block 5
    5: (195.0, 60.0),   # block 6
    6: (15.0, 50.0),    # block 7
    7: (75.0, 20.0),    # block 8
    8: (135.0, 30.0),   # block 9
}


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


def transform_layout(
    layout: Mapping[int, Tuple[float, float]],
    *,
    board_size: BoardSize,
    origin: str = "center",
    scale: float = 1.0,
) -> BlockLayout:
    """Transforms a board layout between image / centered coordinate systems."""

    if origin not in {"image", "center"}:
        raise ValueError("origin must be either 'image' or 'center'")

    width, height = board_size
    transformed: BlockLayout = {}
    for block_id, (x, y) in layout.items():
        if origin == "image":
            tx, ty = x * scale, y * scale
        else:
            tx = (x - width / 2.0) * scale
            ty = (y - height / 2.0) * scale
        transformed[int(block_id)] = (float(tx), float(ty))
    return transformed


def standard_corsi_layout(
    *,
    origin: str = "image",
    scale: float = 1.0,
    board_size: BoardSize = STANDARD_CORSI_BOARD_SIZE,
    use_block_centers: bool = False,
) -> BlockLayout:
    """Returns the standard 9-block Corsi layout from the reference figure."""

    layout = STANDARD_CORSI_LAYOUT_IMAGE
    if use_block_centers:
        half_width = STANDARD_CORSI_BLOCK_SIZE[0] / 2.0
        half_height = STANDARD_CORSI_BLOCK_SIZE[1] / 2.0
        layout = {
            int(block_id): (float(x) + half_width, float(y) + half_height)
            for block_id, (x, y) in STANDARD_CORSI_LAYOUT_IMAGE.items()
        }

    return transform_layout(
        layout,
        board_size=board_size,
        origin=origin,
        scale=scale,
    )


def standard_corsi_robosuite_layout(
    *,
    board_width: float = STANDARD_CORSI_BOARD_SIZE[0],
    target_width: float = STANDARD_CORSI_ROBOSUITE_TARGET_WIDTH,
) -> BlockLayout:
    """Returns the standard layout mapped onto the robosuite table plane for agent-view alignment."""

    centered_layout = standard_corsi_layout(
        origin="center",
        scale=1.0,
        use_block_centers=True,
    )
    scale = target_width / board_width

    # In the standard figure, image x points right and image y points up from the board's bottom-left corner.
    # For the current robosuite agent view, screen left-right aligns with world y and screen top-bottom aligns
    # with negative world x, so we rotate the board coordinates onto the table plane accordingly.
    return {
        int(block_id): (-float(center_y) * scale, float(center_x) * scale)
        for block_id, (center_x, center_y) in centered_layout.items()
    }


def standard_corsi_robosuite_board_size(
    *,
    board_size: BoardSize = STANDARD_CORSI_BOARD_SIZE,
    target_width: float = STANDARD_CORSI_ROBOSUITE_TARGET_WIDTH,
) -> BoardSize:
    """Returns the scaled `(width, height)` of the standard Corsi outer frame in robosuite units."""

    board_width, board_height = board_size
    scale = target_width / board_width
    return (float(board_height * scale), float(board_width * scale))


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
