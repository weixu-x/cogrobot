"""Exports a robosuite visual Corsi preview dataset with EE / gripper XY metadata."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.envs.robosuite_corsi import (  # noqa: E402
    DEFAULT_GRIPPER_SETTLE_STEPS,
    FREE_CAMERA_NAME,
    create_env,
    collect_motion_state,
    init_sequence_state,
    render_tuned_free_camera_frame,
    settled_gripper_action,
    step_pointing_policy,
)
from corsi.envs.sequence_generator import (  # noqa: E402
    STANDARD_CORSI_BOARD_SIZE,
    generate_trial_collection,
    standard_corsi_layout,
    standard_corsi_robosuite_board_size,
)
from corsi.experiments.visual_base.scripts.ee_xy_dataset_utils import (  # noqa: E402
    corsi_lower_left_xy_bounds,
    finite_xy,
    norm_xy_in_bounds,
    robosuite_table_xy_to_corsi_xy,
    table_xy_to_norm,
)
from corsi.scripts.export_robosuite_corsi_frames import (  # noqa: E402
    parse_csv_list,
    save_reset_frames_with_env,
)


DEFAULT_DATASET_NAME = "freecam_ee_xy_v1_preview"
DEFAULT_OUTPUT_DIR = f"corsi_artifacts/visual_base/datasets/{DEFAULT_DATASET_NAME}"
EXHAUSTIVE_SEQUENCE_MODE = "exhaustive_no_consecutive_repeat"
SAMPLED_SEQUENCE_MODE = "sampled_no_consecutive_repeat"
LEN2_3_TRAIN_DATASET_NAME = "freecam_ee_xy_v1_len2_3_train"
LEN2_3_VAL_DATASET_NAME = "freecam_ee_xy_v1_len2_3_val"
LEN4_5_TRAIN_DATASET_NAME = "freecam_ee_xy_v1_len4_5_sampled_train"
LEN4_5_VAL_DATASET_NAME = "freecam_ee_xy_v1_len4_5_sampled_val"
LEN2_3_TOTALS = {
    "num_length_2_total": 72,
    "num_length_3_total": 576,
    "num_train": 514,
    "num_val": 134,
}
LEN2_3_SPLIT_COUNTS = {
    "train": {2: 54, 3: 460},
    "val": {2: 18, 3: 116},
}
LEN4_5_TOTALS = {
    "num_length_4_full": 4608,
    "num_length_5_full": 36864,
    "full_length_4_count": 4608,
    "full_length_5_count": 36864,
    "num_train": 1300,
    "num_val": 260,
}
LEN4_5_SAMPLED_COUNTS = {
    "train": {4: 500, 5: 800},
    "val": {4: 100, 5: 160},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split-name", type=str, default="preview")
    parser.add_argument("--cameras", type=str, default=FREE_CAMERA_NAME)
    parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument("--seq-len-range", type=str, default="2,4")
    parser.add_argument("--sequences-json", type=str, default="")
    parser.add_argument("--sequence-mode", type=str, default="random", choices=["random", EXHAUSTIVE_SEQUENCE_MODE, SAMPLED_SEQUENCE_MODE])
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--validate-against-dir", type=str, default="")
    parser.add_argument("--shard-index", "--shard-id", dest="shard_index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--speed-gain", type=float, default=0.03)
    parser.add_argument("--dwell-steps", type=int, default=24)
    parser.add_argument("--arrival-threshold", type=float, default=0.01)
    parser.add_argument("--target-height", type=float, default=0.03)
    parser.add_argument("--gripper-settle-steps", type=int, default=DEFAULT_GRIPPER_SETTLE_STEPS)
    parser.add_argument("--max-control-steps", type=int, default=5000)
    parser.add_argument("--write-rollout-videos", action="store_true")
    parser.add_argument("--keep-rollout-frames", action="store_true")
    parser.add_argument("--keep-rollout-videos", action="store_true")
    parser.add_argument("--save-local-window", action="store_true")
    parser.add_argument("--local-window-offsets", type=str, default="-2,-1,0,1,2")
    parser.add_argument("--no-save-trajectory-metadata", action="store_true")
    return parser.parse_args()


def parse_seq_len_range(text: str) -> tuple[int, int]:
    values = [int(token) for token in parse_csv_list(text)]
    if len(values) != 2:
        raise ValueError("--seq-len-range must contain exactly two integers, e.g. 2,4")
    return values[0], values[1]


def parse_int_list(text: str) -> list[int]:
    return [int(token) for token in parse_csv_list(text)]


def has_consecutive_repeat(sequence: Sequence[int]) -> bool:
    return any(int(prev) == int(curr) for prev, curr in zip(sequence[:-1], sequence[1:]))


def has_nonconsecutive_repeat(sequence: Sequence[int]) -> bool:
    positions: dict[int, list[int]] = {}
    for index, block_id in enumerate(sequence):
        positions.setdefault(int(block_id), []).append(index)
    return any(
        any(curr - prev > 1 for prev, curr in zip(indices[:-1], indices[1:]))
        for indices in positions.values()
    )


def exhaustive_no_consecutive_repeat_sequences(*, length: int, num_blocks: int = 9) -> list[list[int]]:
    if length < 1:
        raise ValueError("length must be at least 1")
    sequences = [
        [int(block_id) for block_id in sequence]
        for sequence in itertools.product(range(num_blocks), repeat=length)
        if not has_consecutive_repeat(sequence)
    ]
    return sequences


def count_no_consecutive_repeat_sequences(*, length: int, num_blocks: int = 9) -> int:
    if length < 1:
        raise ValueError("length must be at least 1")
    return int(num_blocks * (num_blocks - 1) ** (length - 1))


def random_no_consecutive_repeat_sequence(*, length: int, rng: random.Random, num_blocks: int = 9) -> tuple[int, ...]:
    if length < 1:
        raise ValueError("length must be at least 1")
    sequence = [rng.randrange(num_blocks)]
    for _ in range(1, length):
        choices = [block_id for block_id in range(num_blocks) if block_id != sequence[-1]]
        sequence.append(rng.choice(choices))
    return tuple(sequence)


def sample_unique_no_consecutive_repeat_sequences(
    *,
    length: int,
    count: int,
    rng: random.Random,
    exclude: set[tuple[int, ...]] | None = None,
    num_blocks: int = 9,
) -> list[list[int]]:
    excluded = set() if exclude is None else set(exclude)
    full_count = count_no_consecutive_repeat_sequences(length=length, num_blocks=num_blocks)
    if count + len(excluded) > full_count:
        raise ValueError(
            f"Requested {count} length-{length} samples with {len(excluded)} excluded, "
            f"but only {full_count} valid sequences exist"
        )
    selected: set[tuple[int, ...]] = set()
    while len(selected) < count:
        sequence = random_no_consecutive_repeat_sequence(length=length, rng=rng, num_blocks=num_blocks)
        if sequence in excluded or sequence in selected:
            continue
        selected.add(sequence)
    return [list(sequence) for sequence in selected]


def build_len2_3_exhaustive_split(split_seed: int) -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    rng = random.Random(int(split_seed))
    length_2 = exhaustive_no_consecutive_repeat_sequences(length=2)
    length_3 = exhaustive_no_consecutive_repeat_sequences(length=3)
    rng.shuffle(length_2)
    rng.shuffle(length_3)

    train_len2 = length_2[: LEN2_3_SPLIT_COUNTS["train"][2]]
    val_len2 = length_2[LEN2_3_SPLIT_COUNTS["train"][2] :]
    train_len3 = length_3[: LEN2_3_SPLIT_COUNTS["train"][3]]
    val_len3 = length_3[LEN2_3_SPLIT_COUNTS["train"][3] :]
    split_sequences = {
        "train": train_len2 + train_len3,
        "val": val_len2 + val_len3,
    }
    for split_name in split_sequences:
        rng.shuffle(split_sequences[split_name])

    metadata = {
        "sequence_mode": EXHAUSTIVE_SEQUENCE_MODE,
        "allow_nonconsecutive_repeats": True,
        "allow_consecutive_repeats": False,
        **LEN2_3_TOTALS,
        "split_seed": int(split_seed),
        "split_counts_by_length": {
            "train": {"2": len(train_len2), "3": len(train_len3)},
            "val": {"2": len(val_len2), "3": len(val_len3)},
        },
    }
    validate_exhaustive_split_definition(split_sequences, metadata)
    return split_sequences, metadata


def build_len4_5_sampled_split(*, split_seed: int, sample_seed: int) -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    sample_rng = random.Random(int(sample_seed))
    split_rng = random.Random(int(split_seed))

    train_len4 = sample_unique_no_consecutive_repeat_sequences(
        length=4,
        count=LEN4_5_SAMPLED_COUNTS["train"][4],
        rng=sample_rng,
    )
    train_len5 = sample_unique_no_consecutive_repeat_sequences(
        length=5,
        count=LEN4_5_SAMPLED_COUNTS["train"][5],
        rng=sample_rng,
    )
    train_len4_set = {tuple(sequence) for sequence in train_len4}
    train_len5_set = {tuple(sequence) for sequence in train_len5}
    val_len4 = sample_unique_no_consecutive_repeat_sequences(
        length=4,
        count=LEN4_5_SAMPLED_COUNTS["val"][4],
        rng=sample_rng,
        exclude=train_len4_set,
    )
    val_len5 = sample_unique_no_consecutive_repeat_sequences(
        length=5,
        count=LEN4_5_SAMPLED_COUNTS["val"][5],
        rng=sample_rng,
        exclude=train_len5_set,
    )

    split_sequences = {
        "train": train_len4 + train_len5,
        "val": val_len4 + val_len5,
    }
    for split_name in split_sequences:
        split_rng.shuffle(split_sequences[split_name])

    train_sequences = {tuple(sequence) for sequence in split_sequences["train"]}
    val_sequences = {tuple(sequence) for sequence in split_sequences["val"]}
    overlap = train_sequences & val_sequences
    metadata = {
        "sequence_mode": SAMPLED_SEQUENCE_MODE,
        "allow_nonconsecutive_repeats": True,
        "allow_consecutive_repeats": False,
        **LEN4_5_TOTALS,
        "sampled_length_4_train_count": len(train_len4),
        "sampled_length_5_train_count": len(train_len5),
        "sampled_length_4_val_count": len(val_len4),
        "sampled_length_5_val_count": len(val_len5),
        "split_seed": int(split_seed),
        "sample_seed": int(sample_seed),
        "train_val_overlap_count": len(overlap),
        "split_counts_by_length": {
            "train": {"4": len(train_len4), "5": len(train_len5)},
            "val": {"4": len(val_len4), "5": len(val_len5)},
        },
    }
    validate_sampled_split_definition(split_sequences, metadata)
    return split_sequences, metadata


def validate_sampled_split_definition(split_sequences: dict[str, list[list[int]]], metadata: dict[str, Any]) -> None:
    length_4_total = count_no_consecutive_repeat_sequences(length=4)
    length_5_total = count_no_consecutive_repeat_sequences(length=5)
    if length_4_total != LEN4_5_TOTALS["num_length_4_full"]:
        raise RuntimeError(f"Expected 4608 length-4 sequences, got {length_4_total}")
    if length_5_total != LEN4_5_TOTALS["num_length_5_full"]:
        raise RuntimeError(f"Expected 36864 length-5 sequences, got {length_5_total}")

    train_sequences = {tuple(sequence) for sequence in split_sequences["train"]}
    val_sequences = {tuple(sequence) for sequence in split_sequences["val"]}
    if len(train_sequences) != LEN4_5_TOTALS["num_train"]:
        raise RuntimeError(f"Expected {LEN4_5_TOTALS['num_train']} unique train sequences, got {len(train_sequences)}")
    if len(val_sequences) != LEN4_5_TOTALS["num_val"]:
        raise RuntimeError(f"Expected {LEN4_5_TOTALS['num_val']} unique val sequences, got {len(val_sequences)}")
    overlap = train_sequences & val_sequences
    if overlap:
        raise RuntimeError(f"Train/val sequence overlap in sampled split: {sorted(overlap)[:5]}")
    for split_name, expected_counts in LEN4_5_SAMPLED_COUNTS.items():
        counts = {4: 0, 5: 0}
        for sequence in split_sequences[split_name]:
            if len(sequence) not in counts:
                raise RuntimeError(f"{split_name} contains unsupported sequence length: {sequence}")
            if has_consecutive_repeat(sequence):
                raise RuntimeError(f"{split_name} contains consecutive repeat sequence: {sequence}")
            counts[len(sequence)] += 1
        if counts != expected_counts:
            raise RuntimeError(f"{split_name} sampled counts {counts} != expected {expected_counts}")


def validate_exhaustive_split_definition(split_sequences: dict[str, list[list[int]]], metadata: dict[str, Any]) -> None:
    length_2_total = len(exhaustive_no_consecutive_repeat_sequences(length=2))
    length_3_total = len(exhaustive_no_consecutive_repeat_sequences(length=3))
    if length_2_total != LEN2_3_TOTALS["num_length_2_total"]:
        raise RuntimeError(f"Expected 72 length-2 sequences, got {length_2_total}")
    if length_3_total != LEN2_3_TOTALS["num_length_3_total"]:
        raise RuntimeError(f"Expected 576 length-3 sequences, got {length_3_total}")

    train_sequences = {tuple(sequence) for sequence in split_sequences["train"]}
    val_sequences = {tuple(sequence) for sequence in split_sequences["val"]}
    if len(train_sequences) != metadata["num_train"]:
        raise RuntimeError(f"Expected {metadata['num_train']} unique train sequences, got {len(train_sequences)}")
    if len(val_sequences) != metadata["num_val"]:
        raise RuntimeError(f"Expected {metadata['num_val']} unique val sequences, got {len(val_sequences)}")
    overlap = train_sequences & val_sequences
    if overlap:
        raise RuntimeError(f"Train/val sequence overlap in exhaustive split: {sorted(overlap)[:5]}")
    for split_name, sequences in split_sequences.items():
        for sequence in sequences:
            if len(sequence) not in {2, 3}:
                raise RuntimeError(f"{split_name} contains unsupported sequence length: {sequence}")
            if has_consecutive_repeat(sequence):
                raise RuntimeError(f"{split_name} contains consecutive repeat sequence: {sequence}")


def load_sequences(
    sequences_json: str,
    *,
    num_trials: int,
    seq_len_range: tuple[int, int],
    seed: int,
    sequence_mode: str,
    split_name: str,
    split_seed: int,
    sample_seed: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    if sequences_json:
        payload = json.loads(Path(sequences_json).read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("sequences_json must contain a non-empty list of sequences")
        return [[int(block_id) for block_id in sequence] for sequence in payload], {
            "sequence_mode": "sequences_json",
            "allow_nonconsecutive_repeats": True,
            "allow_consecutive_repeats": False,
            "split_seed": int(split_seed),
            "sample_seed": int(sample_seed),
        }

    if sequence_mode == EXHAUSTIVE_SEQUENCE_MODE:
        if split_name not in {"train", "val"}:
            raise ValueError(f"{EXHAUSTIVE_SEQUENCE_MODE} requires --split-name train or --split-name val")
        split_sequences, split_metadata = build_len2_3_exhaustive_split(split_seed)
        return split_sequences[split_name], split_metadata

    if sequence_mode == SAMPLED_SEQUENCE_MODE:
        if split_name not in {"train", "val"}:
            raise ValueError(f"{SAMPLED_SEQUENCE_MODE} requires --split-name train or --split-name val")
        split_sequences, split_metadata = build_len4_5_sampled_split(
            split_seed=split_seed,
            sample_seed=sample_seed,
        )
        return split_sequences[split_name], split_metadata

    trials = generate_trial_collection(
        num_trials=num_trials,
        seq_len_range=seq_len_range,
        seed=seed,
    )
    return [list(trial.sequence) for trial in trials], {
        "sequence_mode": "random",
        "allow_nonconsecutive_repeats": True,
        "allow_consecutive_repeats": False,
        "split_seed": int(split_seed),
        "sample_seed": int(sample_seed),
    }


def shard_indexed_sequences(
    sequences: list[list[int]],
    *,
    shard_index: int,
    num_shards: int,
) -> tuple[list[tuple[int, list[int]]], dict[str, int]]:
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-id / --shard-index must be in [0, num_shards)")
    start_index = len(sequences) * shard_index // num_shards
    end_index = len(sequences) * (shard_index + 1) // num_shards
    indexed_sequences = list(enumerate(sequences))
    if num_shards == 1:
        return indexed_sequences, {
            "shard_start_index": 0,
            "shard_end_index": len(sequences),
            "shard_num_sequences": len(sequences),
        }
    shard = indexed_sequences[start_index:end_index]
    return shard, {
        "shard_start_index": start_index,
        "shard_end_index": end_index,
        "shard_num_sequences": len(shard),
    }


def build_keyframe_paths(sample_dir: Path, camera_names: list[str], sequence: list[int]) -> dict[str, list[str]]:
    keyframe_paths: dict[str, list[str]] = {}
    for camera_name in camera_names:
        keyframe_paths[camera_name] = [
            str(
                sample_dir
                / "rollout_keyframes"
                / camera_name
                / f"keyframe_step_{step_index:02d}_block_{block_id}_{camera_name}.png"
            )
            for step_index, block_id in enumerate(sequence)
        ]
    return keyframe_paths


def render_camera_frame(env, obs: dict[str, Any], camera_name: str) -> np.ndarray:
    if camera_name == FREE_CAMERA_NAME:
        return render_tuned_free_camera_frame(env)
    obs_key = f"{camera_name}_image"
    if obs_key not in obs:
        raise KeyError(f"Missing camera obs '{obs_key}'. Available keys: {list(obs.keys())}")
    return obs[obs_key]


def get_site_xyz_world(env, site_name: str) -> list[float]:
    site_id = env.sim.model.site_name2id(site_name)
    return [float(v) for v in np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64)]


def get_body_xyz_world(env, body_name: str) -> list[float]:
    return [float(v) for v in np.asarray(env.sim.data.get_body_xpos(body_name), dtype=np.float64)]


def get_table_center_xy_world(env) -> list[float]:
    try:
        site_id = env.sim.model.site_name2id("table_top")
        return [float(v) for v in np.asarray(env.sim.data.site_xpos[site_id][:2], dtype=np.float64)]
    except Exception:
        return [0.0, 0.0]


def world_xy_to_table_xy(world_xy: Sequence[float], table_center_xy_world: Sequence[float]) -> list[float]:
    return [
        float(world_xy[0]) - float(table_center_xy_world[0]),
        float(world_xy[1]) - float(table_center_xy_world[1]),
    ]


def world_xy_to_corsi_xy(world_xy: Sequence[float], table_center_xy_world: Sequence[float]) -> list[float]:
    table_xy = world_xy_to_table_xy(world_xy, table_center_xy_world)
    return robosuite_table_xy_to_corsi_xy(table_xy)


def block_top_xyz_world(env, block_xyz_world: Sequence[float]) -> list[float]:
    block_half_z = 0.0
    arena = getattr(getattr(env, "model", None), "mujoco_arena", None)
    if arena is not None and hasattr(arena, "block_half_size"):
        block_half_z = float(np.asarray(arena.block_half_size, dtype=np.float64)[2])
    return [float(block_xyz_world[0]), float(block_xyz_world[1]), float(block_xyz_world[2]) + block_half_z]


def project_world_points_to_freecam_pixels(
    env,
    points_xyz_world: Sequence[Sequence[float]],
    *,
    width: int = 512,
    height: int = 512,
) -> list[list[int]]:
    """Projects world points into the tuned free-camera image using the active MuJoCo scene camera."""

    render_tuned_free_camera_frame(env, width=width, height=height)
    scene_cameras = env.sim._render_context_offscreen.scn.camera
    camera = scene_cameras[0]
    camera_pos = np.mean([np.asarray(cam.pos, dtype=np.float64) for cam in scene_cameras], axis=0)
    forward = np.mean([np.asarray(cam.forward, dtype=np.float64) for cam in scene_cameras], axis=0)
    forward = forward / np.linalg.norm(forward)
    up = np.mean([np.asarray(cam.up, dtype=np.float64) for cam in scene_cameras], axis=0)
    up = up / np.linalg.norm(up)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    near = float(camera.frustum_near)
    half_height_near = float(camera.frustum_top)
    if half_height_near <= 0.0 or near <= 0.0:
        raise RuntimeError("Invalid free-camera frustum for pixel projection")
    half_width_near = half_height_near * float(width) / float(height)

    pixels: list[list[int]] = []
    for point in points_xyz_world:
        rel = np.asarray(point, dtype=np.float64) - camera_pos
        depth = float(np.dot(rel, forward))
        if depth <= 0.0:
            pixels.append([0, 0])
            continue
        horizontal = float(np.dot(rel, right))
        vertical = float(np.dot(rel, up))
        half_width = half_width_near * depth / near
        half_height = half_height_near * depth / near
        px = (0.5 + horizontal / (2.0 * half_width)) * float(width - 1)
        py = (0.5 - vertical / (2.0 * half_height)) * float(height - 1)
        pixels.append(
            [
                int(np.clip(round(px), 0, width - 1)),
                int(np.clip(round(py), 0, height - 1)),
            ]
        )
    return pixels


def projected_step_pixels(
    env,
    *,
    camera_names: Sequence[str],
    target_xyz_world: Sequence[float],
    ee_xyz_world: Sequence[float],
) -> dict[str, dict[str, list[int]]]:
    pixels: dict[str, dict[str, list[int]]] = {}
    for camera_name in camera_names:
        if camera_name != FREE_CAMERA_NAME:
            continue
        target_pixel, ee_pixel = project_world_points_to_freecam_pixels(env, [target_xyz_world, ee_xyz_world])
        pixels[camera_name] = {
            "target_block_pixel": target_pixel,
            "ee_pixel": ee_pixel,
        }
    return pixels


def collect_step_metadata(
    env,
    state: dict[str, Any],
    *,
    step_index: int,
    control_step: int,
    xy_norm_bounds: dict[str, float],
    table_center_xy_world: Sequence[float],
    camera_names: Sequence[str],
    frame_role: str,
    motion_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    block_index = int(state["block_sequence"][step_index])
    block_name = state["block_names"][block_index]
    block_xyz_world = get_body_xyz_world(env, block_name)
    block_xy_table = world_xy_to_corsi_xy(block_xyz_world[:2], table_center_xy_world)
    ee_site_name = str(state["target_site_name"])
    ee_xyz_world = get_site_xyz_world(env, ee_site_name)
    ee_xy_table = world_xy_to_corsi_xy(ee_xyz_world[:2], table_center_xy_world)
    metadata = {
        "step": int(step_index),
        "block_index": block_index,
        "target_block_xy_table": block_xy_table,
        "target_block_xy_norm": table_xy_to_norm(block_xy_table, xy_norm_bounds),
        "ee_xyz_world": ee_xyz_world,
        "ee_xy_table": ee_xy_table,
        "ee_xy_norm": table_xy_to_norm(ee_xy_table, xy_norm_bounds),
        "frame_role": frame_role,
        "control_step": int(control_step),
        "sim_time": float(env.sim.data.time),
        "image_points": projected_step_pixels(
            env,
            camera_names=camera_names,
            target_xyz_world=block_top_xyz_world(env, block_xyz_world),
            ee_xyz_world=ee_xyz_world,
        ),
    }
    if motion_state is not None:
        metadata["motion_state"] = motion_state
    return metadata


def collect_block_positions(
    env,
    state: dict[str, Any],
    *,
    xy_norm_bounds: dict[str, float],
    table_center_xy_world: Sequence[float],
) -> dict[str, dict[str, list[float]]]:
    """Records static block positions in the same frames as the motion data."""

    positions: dict[str, dict[str, list[float]]] = {}
    for block_index, block_name in enumerate(state["block_names"]):
        block_xyz_world = get_body_xyz_world(env, str(block_name))
        block_xy_table = world_xy_to_corsi_xy(block_xyz_world[:2], table_center_xy_world)
        positions[str(block_index)] = {
            "body_name": str(block_name),
            "xyz_world": [float(v) for v in block_xyz_world],
            "xy_table": [float(v) for v in block_xy_table],
            "xy_norm": table_xy_to_norm(block_xy_table, xy_norm_bounds),
        }
    return positions


def save_local_window_frame(
    *,
    local_windows_dir: Path,
    camera_name: str,
    step_index: int,
    block_index: int,
    offset: int,
    frame: np.ndarray,
) -> str:
    output_dir = local_windows_dir / camera_name
    output_dir.mkdir(parents=True, exist_ok=True)
    sign = "p" if offset >= 0 else "m"
    output_path = output_dir / f"keyframe_step_{step_index:02d}_block_{block_index}_{sign}{abs(offset):02d}_{camera_name}.png"
    imageio.imwrite(output_path, frame)
    return str(output_path)


def rollout_sequence_with_ee_xy_metadata(
    env,
    block_sequence: Sequence[int],
    *,
    video_path: Path,
    frames_dir: Path,
    keyframes_dir: Path,
    local_windows_dir: Path,
    camera_names: Sequence[str],
    fps: int,
    write_videos: bool,
    keep_frames: bool,
    save_local_window: bool,
    local_window_offsets: Sequence[int],
    save_trajectory_metadata: bool,
    xy_norm_bounds: dict[str, float],
    table_center_xy_world: Sequence[float],
    speed_gain: float,
    dwell_steps: int,
    arrival_threshold: float,
    target_height: float,
    gripper_settle_steps: int,
    max_control_steps: int,
) -> dict[str, Any]:
    cameras = list(camera_names)
    frames_dir = Path(frames_dir)
    keyframes_dir = Path(keyframes_dir)
    local_windows_dir = Path(local_windows_dir)
    shutil.rmtree(frames_dir, ignore_errors=True)
    shutil.rmtree(keyframes_dir, ignore_errors=True)
    for camera_name in cameras:
        (keyframes_dir / camera_name).mkdir(parents=True, exist_ok=True)
        if keep_frames:
            (frames_dir / camera_name).mkdir(parents=True, exist_ok=True)
    if save_local_window:
        shutil.rmtree(local_windows_dir, ignore_errors=True)
        for camera_name in cameras:
            (local_windows_dir / camera_name).mkdir(parents=True, exist_ok=True)

    writers = {}
    video_root = video_path.stem
    video_ext = video_path.suffix
    if write_videos:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writers = {
            camera_name: imageio.get_writer(video_path.parent / f"{video_root}_{camera_name}{video_ext}", fps=fps)
            for camera_name in cameras
        }

    max_negative_offset = max([abs(offset) for offset in local_window_offsets if offset < 0] or [0])
    frame_history = {camera_name: deque(maxlen=max_negative_offset + 1) for camera_name in cameras}
    pending_local_saves: list[dict[str, Any]] = []
    step_metadata: list[dict[str, Any]] = []
    trajectory_metadata: list[dict[str, Any]] = []
    local_window_warnings: list[str] = []
    saved_keyframe_steps: set[int] = set()

    try:
        state = init_sequence_state(
            env,
            block_sequence,
            speed_gain=speed_gain,
            dwell_steps=dwell_steps,
            arrival_threshold=arrival_threshold,
            target_height=target_height,
            gripper_settle_steps=gripper_settle_steps,
        )
        while not state["completed"]:
            if int(state["step"]) >= int(max_control_steps):
                raise RuntimeError(
                    "Rollout exceeded max control steps "
                    f"({max_control_steps}) for sequence {list(block_sequence)} at "
                    f"sequence_position={state['sequence_position']}"
                )
            action = step_pointing_policy(state)
            state["obs"], _, _, _ = env.step(action)
            control_step = int(state["step"])
            motion_state = collect_motion_state(env, state, action)
            frames = {camera_name: render_camera_frame(env, state["obs"], camera_name) for camera_name in cameras}

            for camera_name, frame in frames.items():
                if write_videos:
                    writers[camera_name].append_data(frame)
                if keep_frames:
                    imageio.imwrite(frames_dir / camera_name / f"frame_{control_step:05d}.png", frame)
                frame_history[camera_name].append((control_step, frame.copy()))

            if save_trajectory_metadata:
                ee_xyz_world = get_site_xyz_world(env, str(state["target_site_name"]))
                ee_xy_table = world_xy_to_corsi_xy(ee_xyz_world[:2], table_center_xy_world)
                sequence_position = int(state["sequence_position"])
                target_block_index = None
                if sequence_position < len(state["block_sequence"]):
                    target_block_index = int(state["block_sequence"][sequence_position])
                trajectory_metadata.append(
                    {
                        "control_step": control_step,
                        "sim_time": float(env.sim.data.time),
                        "sequence_position": sequence_position,
                        "target_block_index": target_block_index,
                        "ee_xyz_world": ee_xyz_world,
                        "ee_xy_table": ee_xy_table,
                        "ee_xy_norm": table_xy_to_norm(ee_xy_table, xy_norm_bounds),
                        "motion_state": motion_state,
                    }
                )

            for pending in list(pending_local_saves):
                if pending["control_step"] == control_step:
                    pending["paths"] = {
                        camera_name: save_local_window_frame(
                            local_windows_dir=local_windows_dir,
                            camera_name=camera_name,
                            step_index=pending["step_index"],
                            block_index=pending["block_index"],
                            offset=pending["offset"],
                            frame=frames[camera_name],
                        )
                        for camera_name in cameras
                    }
                    metadata_ref = pending.get("metadata")
                    if metadata_ref is not None:
                        metadata_ref.setdefault("local_window_paths", {})[str(pending["offset"])] = pending["paths"]
                    pending_local_saves.remove(pending)

            if state["dwell_counter"] == 1 and not state["completed"]:
                step_index = int(state["sequence_position"])
                if step_index in saved_keyframe_steps:
                    state["step"] += 1
                    continue
                block_index = int(state["block_sequence"][step_index])
                saved_keyframe_steps.add(step_index)
                metadata = collect_step_metadata(
                    env,
                    state,
                    step_index=step_index,
                    control_step=control_step,
                    xy_norm_bounds=xy_norm_bounds,
                    table_center_xy_world=table_center_xy_world,
                    camera_names=cameras,
                    frame_role="keyframe",
                    motion_state=motion_state,
                )
                if save_local_window:
                    metadata["local_window_paths"] = {}
                    for offset in local_window_offsets:
                        if offset <= 0:
                            source_control_step = control_step + int(offset)
                            saved = False
                            for camera_name in cameras:
                                for history_step, history_frame in frame_history[camera_name]:
                                    if history_step == source_control_step:
                                        metadata["local_window_paths"].setdefault(str(offset), {})[camera_name] = (
                                            save_local_window_frame(
                                                local_windows_dir=local_windows_dir,
                                                camera_name=camera_name,
                                                step_index=step_index,
                                                block_index=block_index,
                                                offset=int(offset),
                                                frame=history_frame,
                                            )
                                        )
                                        saved = True
                            if not saved:
                                local_window_warnings.append(
                                    f"Missing local-window frame for step {step_index}, offset {offset}"
                                )
                        else:
                            pending_local_saves.append(
                                {
                                    "control_step": control_step + int(offset),
                                    "step_index": step_index,
                                    "block_index": block_index,
                                    "offset": int(offset),
                                    "paths": {},
                                    "metadata": metadata,
                                }
                            )

                for camera_name, frame in frames.items():
                    imageio.imwrite(
                        keyframes_dir / camera_name / f"keyframe_step_{step_index:02d}_block_{block_index}_{camera_name}.png",
                        frame,
                    )
                step_metadata.append(metadata)

            state["step"] += 1
    finally:
        for camera_name, writer in writers.items():
            writer.close()
            print(f"Video saved to {video_path.parent / f'{video_root}_{camera_name}{video_ext}'}")

    for pending in pending_local_saves:
        local_window_warnings.append(
            f"Missing future local-window frame for step {pending['step_index']}, offset {pending['offset']}"
        )

    if keep_frames:
        print(f"Frames kept at {frames_dir}")
    print(f"Keyframes saved at {keyframes_dir}")
    if save_local_window:
        print(f"Local windows saved at {local_windows_dir}")

    return {
        "step_metadata": step_metadata,
        "trajectory_metadata": trajectory_metadata if save_trajectory_metadata else [],
        "local_window_warnings": local_window_warnings,
    }


def dataset_manifest_payload(
    *,
    output_dir: Path,
    dataset_name: str,
    split_name: str,
    camera_names: list[str],
    seq_len_range: tuple[int, int],
    xy_norm_bounds: dict[str, float],
    table_center_xy_world: Sequence[float],
    board_size_xy: Sequence[float],
    robosuite_board_size_xy: Sequence[float],
    split_metadata: dict[str, Any],
    shard_metadata: dict[str, Any],
    args: argparse.Namespace,
    dataset_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    length_counts: dict[str, int] = {}
    for sample in dataset_samples:
        length_key = str(int(sample["length"]))
        length_counts[length_key] = length_counts.get(length_key, 0) + 1
    return {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "camera_names": camera_names,
        "num_samples": len(dataset_samples),
        "num_sequences": len(dataset_samples),
        "length_counts": length_counts,
        **split_metadata,
        **shard_metadata,
        "samples_dir": str(output_dir / "samples"),
        "sequence_length_range": {
            "min": seq_len_range[0],
            "max": seq_len_range[1],
        },
        "xy_normalization": {
            "frame": "corsi_lower_left",
            "bounds": xy_norm_bounds,
            "table_center_xy_world": [float(v) for v in table_center_xy_world],
            "board_size_xy": [float(v) for v in board_size_xy],
            "robosuite_board_size_xy": [float(v) for v in robosuite_board_size_xy],
            "range": [-1.0, 1.0],
        },
        "debug_block_xy_norm": build_debug_block_xy_norm(xy_norm_bounds),
        "sample_structure": [
            "reset",
            "rollout_keyframes",
            "manifest.json",
        ],
        "optional_structure": [
            "rollout_frames",
            "rollout_local_windows",
        ],
        "shard_info": {
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
            **shard_metadata,
        },
        "export_params": {
            "control_freq": args.control_freq,
            "fps": args.fps,
            "speed_gain": args.speed_gain,
            "dwell_steps": args.dwell_steps,
            "arrival_threshold": args.arrival_threshold,
            "target_height": args.target_height,
            "gripper_settle_steps": args.gripper_settle_steps,
            "max_control_steps": args.max_control_steps,
            "write_rollout_videos": bool(args.write_rollout_videos),
            "keep_rollout_frames": bool(args.keep_rollout_frames),
            "keep_rollout_videos": bool(args.keep_rollout_videos),
            "save_local_window": bool(args.save_local_window),
            "local_window_offsets": parse_int_list(args.local_window_offsets),
            "save_trajectory_metadata": not bool(args.no_save_trajectory_metadata),
            "seed": args.seed,
            "split_seed": args.split_seed,
            "sample_seed": args.sample_seed,
        },
        "samples": dataset_samples,
    }


def path_exists(path_text: str) -> bool:
    return Path(path_text).exists()


def validate_dataset(output_dir: Path, *, tolerance: float = 1.0e-6) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    if not dataset_manifest_path.exists():
        return {
            "passed": False,
            "errors": [f"Missing dataset manifest: {dataset_manifest_path}"],
            "warnings": [],
        }

    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    samples = dataset_manifest.get("samples", [])
    seen_sequences: set[tuple[int, ...]] = set()
    has_allowed_nonconsecutive_repeat = False
    for sample in samples:
        trial_id = sample.get("trial_id", "<unknown>")
        sequence = tuple(int(block_id) for block_id in sample.get("sequence", []))
        if sequence in seen_sequences:
            errors.append(f"{trial_id}: duplicate sequence {list(sequence)}")
        seen_sequences.add(sequence)
        if has_consecutive_repeat(sequence):
            errors.append(f"{trial_id}: consecutive repeated blocks are not allowed: {list(sequence)}")
        if has_nonconsecutive_repeat(sequence):
            has_allowed_nonconsecutive_repeat = True
        manifest_path = Path(sample.get("manifest_path", ""))
        if not manifest_path.exists():
            errors.append(f"{trial_id}: missing manifest.json at {manifest_path}")
            continue
        trial_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        trial_sequence = tuple(int(block_id) for block_id in trial_manifest.get("sequence", []))
        if trial_sequence != sequence:
            errors.append(f"{trial_id}: manifest sequence {list(trial_sequence)} != dataset sequence {list(sequence)}")
        reset_paths = trial_manifest.get("reset_paths", {})
        if "freecam" in trial_manifest.get("camera_names", []) and not path_exists(reset_paths.get("freecam", "")):
            errors.append(f"{trial_id}: missing reset/freecam.png")

        length = int(trial_manifest.get("length", -1))
        step_metadata = trial_manifest.get("step_metadata", [])
        if len(step_metadata) != length:
            errors.append(f"{trial_id}: len(step_metadata)={len(step_metadata)} != length={length}")

        keyframe_paths = trial_manifest.get("keyframe_paths", {})
        freecam_keyframes = keyframe_paths.get("freecam", [])
        if "freecam" in trial_manifest.get("camera_names", []) and len(freecam_keyframes) != length:
            errors.append(f"{trial_id}: len(keyframe_paths['freecam'])={len(freecam_keyframes)} != length={length}")
        for keyframe_path in freecam_keyframes:
            if not path_exists(keyframe_path):
                errors.append(f"{trial_id}: missing keyframe image {keyframe_path}")

        for step in step_metadata:
            step_id = step.get("step", "?")
            ee_xy_norm = step.get("ee_xy_norm", [])
            target_xy_norm = step.get("target_block_xy_norm", [])
            if not finite_xy(ee_xy_norm):
                errors.append(f"{trial_id} step {step_id}: ee_xy_norm is not finite")
            if not finite_xy(target_xy_norm):
                errors.append(f"{trial_id} step {step_id}: target_block_xy_norm is not finite")
            if finite_xy(ee_xy_norm) and not norm_xy_in_bounds(ee_xy_norm, tolerance=tolerance):
                warnings.append(f"{trial_id} step {step_id}: ee_xy_norm outside [-1, 1]: {ee_xy_norm}")
            if finite_xy(target_xy_norm) and not norm_xy_in_bounds(target_xy_norm, tolerance=tolerance):
                warnings.append(
                    f"{trial_id} step {step_id}: target_block_xy_norm outside [-1, 1]: {target_xy_norm}"
                )

    if dataset_manifest.get("sequence_mode") in {EXHAUSTIVE_SEQUENCE_MODE, SAMPLED_SEQUENCE_MODE} and not dataset_manifest.get("is_shard"):
        expected_count = None
        if dataset_manifest.get("split_name") == "train":
            expected_count = int(dataset_manifest.get("num_train", -1))
        elif dataset_manifest.get("split_name") == "val":
            expected_count = int(dataset_manifest.get("num_val", -1))
        if expected_count is not None and len(samples) != expected_count:
            errors.append(f"Expected {expected_count} samples for split {dataset_manifest.get('split_name')}, got {len(samples)}")
        if not has_allowed_nonconsecutive_repeat:
            warnings.append("No non-consecutive repeat sequence observed in this split")

        expected_counts_by_length = dataset_manifest.get("split_counts_by_length", {}).get(
            dataset_manifest.get("split_name", ""),
            {},
        )
        if expected_counts_by_length:
            observed_counts_by_length: dict[str, int] = {}
            for sample in samples:
                length_key = str(int(sample.get("length", len(sample.get("sequence", [])))))
                observed_counts_by_length[length_key] = observed_counts_by_length.get(length_key, 0) + 1
            for length_key, expected_length_count in expected_counts_by_length.items():
                observed_count = observed_counts_by_length.get(str(length_key), 0)
                if observed_count != int(expected_length_count):
                    errors.append(
                        f"Expected {expected_length_count} length-{length_key} samples for split "
                        f"{dataset_manifest.get('split_name')}, got {observed_count}"
                    )

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def read_manifest_sequences(dataset_dir: Path) -> set[tuple[int, ...]]:
    manifest_path = Path(dataset_dir) / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        tuple(int(block_id) for block_id in sample.get("sequence", []))
        for sample in manifest.get("samples", [])
    }


def validate_sequence_overlap(dataset_dir: Path, other_dataset_dir: Path) -> dict[str, Any]:
    sequences = read_manifest_sequences(Path(dataset_dir))
    other_sequences = read_manifest_sequences(Path(other_dataset_dir))
    overlap = sequences & other_sequences
    return {
        "passed": not overlap,
        "num_overlap": len(overlap),
        "overlap_examples": [list(sequence) for sequence in sorted(overlap)[:10]],
        "dataset_dir": str(dataset_dir),
        "other_dataset_dir": str(other_dataset_dir),
    }


def build_debug_block_xy_norm(xy_norm_bounds: dict[str, float]) -> dict[str, list[float]]:
    layout = standard_corsi_layout(origin="image", use_block_centers=True)
    return {
        str(block_index): table_xy_to_norm(layout[block_index], xy_norm_bounds)
        for block_index in sorted(layout)
    }


def write_preview_summary(output_dir: Path, dataset_manifest: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    samples = dataset_manifest.get("samples", [])
    example_sample = samples[0] if samples else {}
    example_trial_manifest = {}
    if example_sample.get("manifest_path") and Path(example_sample["manifest_path"]).exists():
        example_trial_manifest = json.loads(Path(example_sample["manifest_path"]).read_text(encoding="utf-8"))

    example_camera = "freecam"
    example_keyframe_path = ""
    keyframe_paths = example_trial_manifest.get("keyframe_paths", {})
    if keyframe_paths.get(example_camera):
        example_keyframe_path = keyframe_paths[example_camera][0]

    summary = {
        "dataset_name": dataset_manifest.get("dataset_name"),
        "output_root": str(output_dir),
        "number_of_trials": int(dataset_manifest.get("num_samples", 0)),
        "length_range": dataset_manifest.get("sequence_length_range"),
        "example_trial_ids": [sample.get("trial_id") for sample in samples[:5]],
        "example_sequence": example_trial_manifest.get("sequence", []),
        "example_keyframe_path": example_keyframe_path,
        "example_step_metadata": (example_trial_manifest.get("step_metadata") or [{}])[0],
        "debug_block_xy_norm": dataset_manifest.get("debug_block_xy_norm", {}),
        "sequence_mode": dataset_manifest.get("sequence_mode"),
        "length_counts": dataset_manifest.get("length_counts", {}),
        "validation": validation,
    }
    summary_path = output_dir / "preview_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    camera_names = parse_csv_list(args.cameras)
    env_camera_names = [camera_name for camera_name in camera_names if camera_name != FREE_CAMERA_NAME]
    if not env_camera_names:
        env_camera_names = ["agentview"]

    output_dir = Path(args.output_dir)
    split_name = args.split_name or "preview"
    if args.sequence_mode == EXHAUSTIVE_SEQUENCE_MODE and args.output_dir == DEFAULT_OUTPUT_DIR:
        if split_name == "train":
            output_dir = Path("corsi_artifacts/visual_base/datasets") / LEN2_3_TRAIN_DATASET_NAME
        elif split_name == "val":
            output_dir = Path("corsi_artifacts/visual_base/datasets") / LEN2_3_VAL_DATASET_NAME
    if args.sequence_mode == SAMPLED_SEQUENCE_MODE and args.output_dir == DEFAULT_OUTPUT_DIR:
        if split_name == "train":
            output_dir = Path("corsi_artifacts/visual_base/datasets") / LEN4_5_TRAIN_DATASET_NAME
        elif split_name == "val":
            output_dir = Path("corsi_artifacts/visual_base/datasets") / LEN4_5_VAL_DATASET_NAME
    base_output_dir = output_dir
    dataset_name = args.dataset_name or base_output_dir.name
    if args.sequence_mode in {EXHAUSTIVE_SEQUENCE_MODE, SAMPLED_SEQUENCE_MODE} and args.dataset_name == DEFAULT_DATASET_NAME:
        dataset_name = base_output_dir.name
    if args.num_shards > 1:
        output_dir = base_output_dir / "shards" / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    seq_len_range = parse_seq_len_range(args.seq_len_range)
    local_window_offsets = parse_int_list(args.local_window_offsets)
    save_trajectory_metadata = not bool(args.no_save_trajectory_metadata)
    board_size_xy = STANDARD_CORSI_BOARD_SIZE
    robosuite_board_size_xy = standard_corsi_robosuite_board_size()
    xy_norm_bounds = corsi_lower_left_xy_bounds(board_size_xy)

    sequences, split_metadata = load_sequences(
        args.sequences_json,
        num_trials=args.num_trials,
        seq_len_range=seq_len_range,
        seed=args.seed,
        sequence_mode=args.sequence_mode,
        split_name=split_name,
        split_seed=args.split_seed,
        sample_seed=args.sample_seed,
    )
    indexed_sequences, shard_metadata = shard_indexed_sequences(
        sequences,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    shard_metadata = {
        **shard_metadata,
        "is_shard": bool(args.num_shards > 1),
        "base_output_dir": str(base_output_dir),
        "shard_output_dir": str(output_dir),
    }

    env = create_env(
        render_mode="offline",
        offline_cameras=env_camera_names,
        control_freq=args.control_freq,
    )

    dataset_samples: list[dict[str, Any]] = []
    try:
        table_center_xy_world = get_table_center_xy_world(env)
        for global_sample_index, sequence in indexed_sequences:
            trial_id = f"trial_{global_sample_index:06d}"
            sample_dir = samples_dir / trial_id
            reset_dir = sample_dir / "reset"
            rollout_frames_dir = sample_dir / "rollout_frames"
            keyframes_dir = sample_dir / "rollout_keyframes"
            local_windows_dir = sample_dir / "rollout_local_windows"
            manifest_path = sample_dir / "manifest.json"

            if args.resume_existing and manifest_path.exists():
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                existing_sequence = [int(block_id) for block_id in existing_manifest.get("sequence", [])]
                if existing_sequence != [int(block_id) for block_id in sequence]:
                    raise RuntimeError(
                        f"Cannot resume {trial_id}: existing sequence {existing_sequence} "
                        f"does not match expected {list(sequence)}"
                    )
                dataset_samples.append(
                    {
                        **existing_manifest,
                        "sample_dir": str(sample_dir),
                        "manifest_path": str(manifest_path),
                    }
                )
                continue

            obs = env.reset()
            settled_obs = settled_gripper_action(env, steps=args.gripper_settle_steps)
            if settled_obs is not None:
                obs = settled_obs
            reset_paths = save_reset_frames_with_env(env, obs, camera_names, reset_dir)
            rollout_payload = rollout_sequence_with_ee_xy_metadata(
                env,
                sequence,
                video_path=sample_dir / "rollout.mp4",
                frames_dir=rollout_frames_dir,
                keyframes_dir=keyframes_dir,
                local_windows_dir=local_windows_dir,
                camera_names=camera_names,
                fps=args.fps,
                write_videos=args.write_rollout_videos,
                keep_frames=args.keep_rollout_frames,
                save_local_window=args.save_local_window,
                local_window_offsets=local_window_offsets,
                save_trajectory_metadata=save_trajectory_metadata,
                xy_norm_bounds=xy_norm_bounds,
                table_center_xy_world=table_center_xy_world,
                speed_gain=args.speed_gain,
                dwell_steps=args.dwell_steps,
                arrival_threshold=args.arrival_threshold,
                target_height=args.target_height,
                gripper_settle_steps=args.gripper_settle_steps,
                max_control_steps=args.max_control_steps,
            )

            if args.write_rollout_videos and not args.keep_rollout_videos:
                for camera_name in camera_names:
                    camera_video_path = sample_dir / f"rollout_{camera_name}.mp4"
                    if camera_video_path.exists():
                        camera_video_path.unlink()

            sample_manifest = {
                "dataset_name": dataset_name,
                "split_name": split_name,
                "trial_id": trial_id,
                "camera_names": camera_names,
                "sequence": sequence,
                "length": len(sequence),
                "reset_paths": reset_paths,
                "keyframe_paths": build_keyframe_paths(sample_dir, camera_names, sequence),
                "sample_structure": [
                    "reset",
                    "rollout_keyframes",
                    "manifest.json",
                ],
                "metadata": {
                    "fps": args.fps,
                    "control_freq": args.control_freq,
                    "speed_gain": args.speed_gain,
                    "dwell_steps": args.dwell_steps,
                    "arrival_threshold": args.arrival_threshold,
                    "target_height": args.target_height,
                    "gripper_settle_steps": args.gripper_settle_steps,
                    "seed": args.seed,
                    "ee_site_name": "gripper0_right_index_tip_site",
                    "xy_normalization": {
                        "frame": "corsi_lower_left",
                        "bounds": xy_norm_bounds,
                        "table_center_xy_world": [float(v) for v in table_center_xy_world],
                        "board_size_xy": [float(v) for v in board_size_xy],
                        "robosuite_board_size_xy": [float(v) for v in robosuite_board_size_xy],
                        "range": [-1.0, 1.0],
                    },
                    "save_local_window": bool(args.save_local_window),
                    "local_window_offsets": local_window_offsets,
                    "save_trajectory_metadata": save_trajectory_metadata,
                    "local_window_warnings": rollout_payload["local_window_warnings"],
                },
                "step_metadata": rollout_payload["step_metadata"],
            }
            if save_trajectory_metadata:
                sample_manifest["trajectory_metadata"] = rollout_payload["trajectory_metadata"]
            manifest_path.write_text(json.dumps(sample_manifest, indent=2), encoding="utf-8")

            dataset_samples.append(
                {
                    **sample_manifest,
                    "sample_dir": str(sample_dir),
                    "manifest_path": str(manifest_path),
                }
            )
    finally:
        env.close()

    dataset_manifest = dataset_manifest_payload(
        output_dir=output_dir,
        dataset_name=dataset_name,
        split_name=split_name,
        camera_names=camera_names,
        seq_len_range=seq_len_range,
        xy_norm_bounds=xy_norm_bounds,
        table_center_xy_world=table_center_xy_world,
        board_size_xy=board_size_xy,
        robosuite_board_size_xy=robosuite_board_size_xy,
        split_metadata=split_metadata,
        shard_metadata=shard_metadata,
        args=args,
        dataset_samples=dataset_samples,
    )
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")

    validation = validate_dataset(output_dir)
    if args.validate_against_dir:
        validation["sequence_overlap"] = validate_sequence_overlap(output_dir, Path(args.validate_against_dir))
        if not validation["sequence_overlap"]["passed"]:
            validation["errors"].append(
                f"Sequence overlap with {args.validate_against_dir}: "
                f"{validation['sequence_overlap']['overlap_examples']}"
            )
            validation["passed"] = False
    summary = write_preview_summary(output_dir, dataset_manifest, validation)

    if any(math.isnan(float(v)) for sample in dataset_samples for step in sample["step_metadata"] for v in step["ee_xy_norm"]):
        raise RuntimeError("NaN detected in exported EE coordinates")

    print(json.dumps(summary, indent=2))
    print(f"Dataset manifest saved to {dataset_manifest_path}")
    print(f"Preview summary saved to {output_dir / 'preview_summary.json'}")
    if not validation["passed"]:
        raise RuntimeError(f"Validation failed: {validation['errors']}")


if __name__ == "__main__":
    main()
