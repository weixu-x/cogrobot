"""Canonical data builder for Corsi memory-recall V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


RAW_SCHEMA_VERSION = "scala_corsi_motion_raw_v1"
SCHEMA_VERSION = "scala_corsi_memory_recall_v2_canonical_v1"
DEFAULT_EOS_TOKEN_ID = 9
DEFAULT_IGNORE_INDEX = -100
DEFAULT_NUM_BLOCKS = 9
REQUIRED_RAW_ARRAYS = (
    "rgb",
    "joint",
    "joint_velocity",
    "ee_pose",
    "ee_xy",
    "action",
    "qpos",
    "qvel",
    "timestamp",
    "rank",
    "block_id",
)
DEFAULT_JOINT_NAMES = [
    "robot0_joint1",
    "robot0_joint2",
    "robot0_joint3",
    "robot0_joint4",
    "robot0_joint5",
    "robot0_joint6",
    "robot0_joint7",
]
MODEL_INPUT_FIELDS = ["images", "segment_mask", "frame_mask"]
FORBIDDEN_MODEL_INPUT_FIELDS = [
    "target_tokens",
    "block_order",
    "block_id",
    "rank",
    "block_xy",
    "length",
    "joint",
    "ee_pose",
    "ee_xy",
    "qpos",
    "qvel",
    "action",
    "source_frame_index",
    "source_timestamp",
]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = root / path
    if candidate.exists():
        return candidate
    return path


def _nearest_indices(timestamps: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(timestamps, target_times, side="left")
    indices = np.clip(indices, 0, len(timestamps) - 1)
    previous = np.clip(indices - 1, 0, len(timestamps) - 1)
    choose_previous = np.abs(timestamps[previous] - target_times) <= np.abs(
        timestamps[indices] - target_times
    )
    return np.where(choose_previous, previous, indices).astype(np.int64)


def _interpolate_rows(timestamps: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"expected 2D values for interpolation, got {values.shape}")
    if len(timestamps) == 1:
        return np.repeat(values[:1], repeats=len(target_times), axis=0).astype(np.float32)
    columns = [
        np.interp(target_times, timestamps, values[:, dim]).astype(np.float32)
        for dim in range(values.shape[1])
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def _table_xy_to_norm(table_xy: Sequence[float], bounds: Mapping[str, float]) -> list[float]:
    x, y = float(table_xy[0]), float(table_xy[1])
    x_min = float(bounds["x_min"])
    x_max = float(bounds["x_max"])
    y_min = float(bounds["y_min"])
    y_max = float(bounds["y_max"])
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"invalid XY bounds: {dict(bounds)}")
    return [
        2.0 * (x - x_min) / (x_max - x_min) - 1.0,
        2.0 * (y - y_min) / (y_max - y_min) - 1.0,
    ]


def _normalize_ee_xy(ee_xy: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    xy_normalization = metadata.get("xy_normalization") or {}
    bounds = xy_normalization.get("bounds")
    if not isinstance(bounds, Mapping):
        raise ValueError("metadata.xy_normalization.bounds is required for normalized EE XY")
    rows = [_table_xy_to_norm(row.tolist(), bounds) for row in ee_xy]
    return np.asarray(rows, dtype=np.float32)


def _as_float_list(values: Sequence[float], *, expected: int, field: str) -> list[float]:
    if len(values) != expected:
        raise ValueError(f"{field}: expected {expected} values, got {len(values)}")
    return [float(value) for value in values]


def _validate_block_order(block_order: Sequence[int], *, num_blocks: int, seq_id: str) -> list[int]:
    order = [int(block_id) for block_id in block_order]
    if len(order) != len(set(order)):
        raise ValueError(f"{seq_id}: repeated block in block_order {order}")
    invalid = [block_id for block_id in order if block_id < 0 or block_id >= num_blocks]
    if invalid:
        raise ValueError(f"{seq_id}: block ids outside 0..{num_blocks - 1}: {invalid}")
    return order


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or _is_relative_to(left_resolved, right_resolved)
        or _is_relative_to(right_resolved, left_resolved)
    )


def _assert_safe_canonical_root(config: Mapping[str, Any], raw_root: Path, canonical_root: Path) -> None:
    if str(canonical_root) in {"", ".", "/"}:
        raise ValueError(f"refusing unsafe canonical_root={canonical_root}")
    if _paths_overlap(canonical_root, raw_root):
        raise ValueError(f"canonical_root must not overlap raw dataset root: {raw_root}")
    protected_roots = {
        str(config.get("protected_v1_canonical_root") or ""),
        "corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12",
    }
    for protected in protected_roots:
        if protected and _paths_overlap(canonical_root, Path(protected)):
            raise ValueError(f"canonical_root must not overlap protected V1 root: {protected}")


def canonicalize_episode(
    *,
    raw_root: Path,
    sample: Mapping[str, Any],
    k: int,
    joint_names: Sequence[str],
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    eos_token_id: int = DEFAULT_EOS_TOKEN_ID,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    seq_id = str(sample["seq_id"])
    arrays_path = _resolve(raw_root, str(sample["arrays_path"]))
    metadata_path = _resolve(raw_root, str(sample["metadata_path"]))
    segments_path = _resolve(raw_root, str(sample["segments_path"]))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    length = int(sample["length"])
    block_order = _validate_block_order(
        sample["block_order"], num_blocks=int(num_blocks), seq_id=seq_id
    )
    if len(block_order) != length:
        raise ValueError(f"{seq_id}: block_order length {len(block_order)} != length {length}")
    if len(segments) != length:
        raise ValueError(f"{seq_id}: segment count {len(segments)} != length {length}")
    if metadata.get("schema_version") != RAW_SCHEMA_VERSION:
        raise ValueError(f"{seq_id}: unexpected raw schema {metadata.get('schema_version')!r}")
    if [int(v) for v in metadata.get("block_order", [])] != block_order:
        raise ValueError(f"{seq_id}: metadata block_order disagrees with manifest")

    with np.load(arrays_path) as raw:
        missing = [key for key in REQUIRED_RAW_ARRAYS if key not in raw.files]
        if missing:
            raise ValueError(f"{seq_id}: missing raw arrays {missing}")
        rgb = np.asarray(raw["rgb"])
        joints = np.asarray(raw["joint"], dtype=np.float32)
        ee_pose = np.asarray(raw["ee_pose"], dtype=np.float32)
        ee_xy = np.asarray(raw["ee_xy"], dtype=np.float32)
        timestamps = np.asarray(raw["timestamp"], dtype=np.float64)
        rank_array = np.asarray(raw["rank"], dtype=np.int64)
        block_array = np.asarray(raw["block_id"], dtype=np.int64)

    if rgb.ndim != 4 or rgb.shape[1:] != (128, 128, 3):
        raise ValueError(f"{seq_id}: expected raw rgb [T,128,128,3], got {rgb.shape}")
    frame_count = int(rgb.shape[0])
    expected_rows = {
        "joint": joints,
        "ee_pose": ee_pose,
        "ee_xy": ee_xy,
        "timestamp": timestamps,
        "rank": rank_array,
        "block_id": block_array,
    }
    for name, values in expected_rows.items():
        if values.shape[0] != frame_count:
            raise ValueError(f"{seq_id}: {name} rows {values.shape[0]} != rgb rows {frame_count}")
    if joints.shape[1] != len(joint_names):
        raise ValueError(
            f"{seq_id}: joint width {joints.shape[1]} != configured names {len(joint_names)}"
        )
    if ee_pose.shape[1] != 7:
        raise ValueError(f"{seq_id}: expected ee_pose width 7, got {ee_pose.shape[1]}")
    if ee_xy.shape[1] != 2:
        raise ValueError(f"{seq_id}: expected ee_xy width 2, got {ee_xy.shape[1]}")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{seq_id}: timestamps must be strictly increasing")

    ee_xy_norm = _normalize_ee_xy(ee_xy, metadata)
    block_positions = metadata.get("block_positions") or {}
    tau = np.linspace(0.0, 1.0, int(k), dtype=np.float64)

    image_rows: list[np.ndarray] = []
    joint_rows: list[np.ndarray] = []
    ee_pose_rows: list[np.ndarray] = []
    ee_xy_rows: list[np.ndarray] = []
    ee_xy_norm_rows: list[np.ndarray] = []
    block_xy_rows: list[list[float]] = []
    block_xyz_rows: list[list[float]] = []
    progress_rows: list[np.ndarray] = []
    source_frame_rows: list[np.ndarray] = []
    source_time_rows: list[np.ndarray] = []
    rank_rows: list[int] = []
    block_rows: list[int] = []
    segment_start_rows: list[int] = []
    segment_end_rows: list[int] = []

    for expected_rank, segment in enumerate(segments):
        rank = int(segment["rank"])
        block_id = int(segment["block_id"])
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if rank != expected_rank:
            raise ValueError(f"{seq_id}: rank {rank} != expected {expected_rank}")
        if block_id != block_order[rank]:
            raise ValueError(f"{seq_id}: block_id {block_id} != block_order[{rank}]")
        if start < 0 or end < start or end >= frame_count:
            raise ValueError(f"{seq_id}: invalid segment bounds {start}..{end}")
        segment_slice = slice(start, end + 1)
        if np.any(rank_array[segment_slice] != rank) or np.any(block_array[segment_slice] != block_id):
            raise ValueError(f"{seq_id}: raw rank/block arrays disagree with segment {rank}")

        seg_timestamps = timestamps[segment_slice]
        target_times = seg_timestamps[0] + tau * (seg_timestamps[-1] - seg_timestamps[0])
        local_indices = _nearest_indices(seg_timestamps, target_times)
        global_indices = local_indices + start
        position = block_positions.get(str(block_id))
        if not isinstance(position, Mapping):
            raise ValueError(f"{seq_id}: missing metadata.block_positions[{block_id!r}]")
        xy_norm = _as_float_list(position["xy_norm"], expected=2, field="xy_norm")
        xyz_world = _as_float_list(position["xyz_world"], expected=3, field="xyz_world")

        image_rows.append(np.transpose(rgb[global_indices], (0, 3, 1, 2)))
        joint_rows.append(_interpolate_rows(seg_timestamps, joints[segment_slice], target_times))
        ee_pose_rows.append(_interpolate_rows(seg_timestamps, ee_pose[segment_slice], target_times))
        ee_xy_rows.append(_interpolate_rows(seg_timestamps, ee_xy[segment_slice], target_times))
        ee_xy_norm_rows.append(
            _interpolate_rows(seg_timestamps, ee_xy_norm[segment_slice], target_times)
        )
        block_xy_rows.append(xy_norm)
        block_xyz_rows.append(xyz_world)
        progress_rows.append(tau.astype(np.float32))
        source_frame_rows.append(global_indices.astype(np.int64))
        source_time_rows.append(timestamps[global_indices].astype(np.float64))
        rank_rows.append(rank)
        block_rows.append(block_id)
        segment_start_rows.append(start)
        segment_end_rows.append(end)

    images = np.stack(image_rows, axis=0).astype(np.uint8)
    target_tokens = np.asarray(block_order + [int(eos_token_id)], dtype=np.int64)
    arrays = {
        "images": images,
        "joint_targets": np.stack(joint_rows, axis=0).astype(np.float32),
        "ee_pose_targets": np.stack(ee_pose_rows, axis=0).astype(np.float32),
        "ee_xy_targets": np.stack(ee_xy_rows, axis=0).astype(np.float32),
        "ee_xy_norm_targets": np.stack(ee_xy_norm_rows, axis=0).astype(np.float32),
        "block_xy_targets": np.asarray(block_xy_rows, dtype=np.float32),
        "block_xyz_world_targets": np.asarray(block_xyz_rows, dtype=np.float32),
        "target_tokens": target_tokens,
        "target_token_mask": np.ones((length + 1,), dtype=bool),
        "segment_mask": np.ones((length,), dtype=bool),
        "frame_mask": np.ones((length, int(k)), dtype=bool),
        "rank": np.asarray(rank_rows, dtype=np.int64),
        "block_id": np.asarray(block_rows, dtype=np.int64),
        "segment_progress": np.stack(progress_rows, axis=0).astype(np.float32),
        "source_frame_index": np.stack(source_frame_rows, axis=0).astype(np.int64),
        "source_timestamp": np.stack(source_time_rows, axis=0).astype(np.float64),
        "segment_start_frame": np.asarray(segment_start_rows, dtype=np.int64),
        "segment_end_frame": np.asarray(segment_end_rows, dtype=np.int64),
    }
    episode_meta = {
        "seq_id": seq_id,
        "length": length,
        "block_order": block_order,
        "target_tokens": target_tokens.tolist(),
        "layout_id": str(sample["layout_id"]),
        "seed": int(sample["seed"]),
        "raw_frame_count": frame_count,
        "raw_arrays_path": str(arrays_path),
        "raw_metadata_path": str(metadata_path),
        "raw_segments_path": str(segments_path),
        "canonical_shape": {
            "images": list(images.shape),
            "target_tokens": list(target_tokens.shape),
        },
        "k_samples_per_segment": int(k),
        "block_xy_targets": arrays["block_xy_targets"].tolist(),
        "block_xyz_world_targets": arrays["block_xyz_world_targets"].tolist(),
    }
    return arrays, episode_meta


def deterministic_split(
    samples: list[Mapping[str, Any]],
    *,
    split_seed: int,
    train_per_length: int,
    val_per_length: int,
    test_per_length: int,
) -> dict[str, list[str]]:
    by_length: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    seen_orders: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    for sample in samples:
        length = int(sample["length"])
        order = tuple(int(v) for v in sample["block_order"])
        if order in seen_orders[length]:
            raise ValueError(f"duplicate block_order within length {length}: {order}")
        seen_orders[length].add(order)
        by_length[length].append(sample)

    required = int(train_per_length) + int(val_per_length) + int(test_per_length)
    split = {"train": [], "val": [], "test": []}
    for length, rows_for_length in sorted(by_length.items()):
        if len(rows_for_length) != required:
            raise ValueError(
                f"length {length}: expected {required} episodes, got {len(rows_for_length)}"
            )
        rows = sorted(rows_for_length, key=lambda item: str(item["seq_id"]))
        rng = random.Random(int(split_seed) + length)
        rng.shuffle(rows)
        split["train"].extend(str(item["seq_id"]) for item in rows[:train_per_length])
        split["val"].extend(
            str(item["seq_id"])
            for item in rows[train_per_length : train_per_length + val_per_length]
        )
        split["test"].extend(
            str(item["seq_id"]) for item in rows[train_per_length + val_per_length :]
        )
    return {key: sorted(values) for key, values in split.items()}


def _accumulate_rows(
    stats: dict[str, dict[str, Any]], name: str, values: np.ndarray
) -> None:
    rows = values.astype(np.float64).reshape(-1, values.shape[-1])
    if name not in stats:
        stats[name] = {
            "sum": np.zeros((rows.shape[1],), dtype=np.float64),
            "sumsq": np.zeros((rows.shape[1],), dtype=np.float64),
            "count": 0,
        }
    stats[name]["sum"] += rows.sum(axis=0)
    stats[name]["sumsq"] += (rows * rows).sum(axis=0)
    stats[name]["count"] += int(rows.shape[0])


def _fit_stats(canonical_root: Path, samples: Sequence[Mapping[str, Any]], train_ids: Iterable[str]) -> dict[str, Any]:
    sample_by_id = {str(sample["seq_id"]): sample for sample in samples}
    image_sum = np.zeros((3,), dtype=np.float64)
    image_sumsq = np.zeros((3,), dtype=np.float64)
    image_count = 0
    continuous: dict[str, dict[str, Any]] = {}
    for seq_id in train_ids:
        sample = sample_by_id[str(seq_id)]
        path = Path(str(sample["canonical_path"]))
        if not path.is_absolute() and not path.exists():
            path = canonical_root / "episodes" / path.name
        with np.load(path) as arrays:
            images = arrays["images"].astype(np.float64) / 255.0
            pixels = images.transpose(2, 0, 1, 3, 4).reshape(3, -1)
            image_sum += pixels.sum(axis=1)
            image_sumsq += (pixels * pixels).sum(axis=1)
            image_count += int(pixels.shape[1])
            _accumulate_rows(continuous, "joint", arrays["joint_targets"])
            _accumulate_rows(continuous, "ee_pose", arrays["ee_pose_targets"])
            _accumulate_rows(continuous, "ee_xy", arrays["ee_xy_targets"])
            _accumulate_rows(continuous, "ee_xy_norm", arrays["ee_xy_norm_targets"])
            _accumulate_rows(continuous, "block_xy", arrays["block_xy_targets"])
    if image_count == 0 or not continuous:
        raise ValueError("cannot fit normalization on an empty train split")

    result: dict[str, Any] = {
        "source": "train_only",
        "image_mean": (image_sum / image_count).tolist(),
        "image_std": np.sqrt(
            np.maximum(image_sumsq / image_count - (image_sum / image_count) ** 2, 1e-12)
        ).tolist(),
        "train_image_values": int(image_count),
    }
    for name, accum in sorted(continuous.items()):
        count = int(accum["count"])
        if count == 0:
            raise ValueError(f"cannot fit normalization for empty field {name}")
        mean = accum["sum"] / count
        var = np.maximum(accum["sumsq"] / count - mean * mean, 1e-12)
        result[f"{name}_mean"] = mean.tolist()
        result[f"{name}_std"] = np.sqrt(var).tolist()
        result[f"train_{name}_rows"] = count
    return result


def _length_counts(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(int(item["length"]) for item in samples)
    return {str(length): int(counts[length]) for length in sorted(counts)}


def _split_length_counts(
    samples: Sequence[Mapping[str, Any]], split: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, int]]:
    sample_by_id = {str(sample["seq_id"]): sample for sample in samples}
    result: dict[str, dict[str, int]] = {}
    for split_name, seq_ids in split.items():
        result[split_name] = _length_counts([sample_by_id[str(seq_id)] for seq_id in seq_ids])
    return result


def _protected_v1_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = config.get(
        "protected_v1_manifest_path",
        "corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12/manifest.json",
    )
    path = Path(str(manifest_path))
    return {
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path) if path.exists() else None,
    }


def build_canonical_dataset(config: Mapping[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    raw_root = Path(str(config["raw_dataset_root"]))
    canonical_root = Path(str(config["canonical_root"]))
    _assert_safe_canonical_root(config, raw_root, canonical_root)
    if canonical_root.exists() and not canonical_root.is_dir():
        raise ValueError(f"canonical_root exists and is not a directory: {canonical_root}")
    if canonical_root.exists() and any(canonical_root.iterdir()) and not overwrite:
        raise FileExistsError(f"{canonical_root} exists; pass --overwrite to rebuild")
    if overwrite and canonical_root.exists():
        shutil.rmtree(canonical_root)
    episodes_dir = canonical_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    raw_manifest_path = raw_root / "manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    if raw_manifest.get("schema_version") != RAW_SCHEMA_VERSION:
        raise ValueError(f"unexpected raw schema {raw_manifest.get('schema_version')!r}")
    joint_names = list(config.get("joint_names") or DEFAULT_JOINT_NAMES)
    if len(joint_names) != 7:
        raise ValueError("memory-recall V2 expects exactly seven Panda arm joints")
    k = int(config.get("k_samples_per_segment", 12))
    if k < 1:
        raise ValueError("k_samples_per_segment must be at least 1")
    num_blocks = int(config.get("num_blocks", DEFAULT_NUM_BLOCKS))
    eos_token_id = int(config.get("eos_token_id", DEFAULT_EOS_TOKEN_ID))
    max_sequence_length = int(
        config.get(
            "max_sequence_length",
            max(int(sample["length"]) for sample in raw_manifest.get("samples", [])),
        )
    )

    manifest_samples = []
    for sample in raw_manifest["samples"]:
        arrays, episode_meta = canonicalize_episode(
            raw_root=raw_root,
            sample=sample,
            k=k,
            joint_names=joint_names,
            num_blocks=num_blocks,
            eos_token_id=eos_token_id,
        )
        if int(episode_meta["length"]) > max_sequence_length:
            raise ValueError(
                f"{episode_meta['seq_id']}: length {episode_meta['length']} exceeds "
                f"max_sequence_length={max_sequence_length}"
            )
        episode_path = episodes_dir / f"{episode_meta['seq_id']}.npz"
        np.savez_compressed(episode_path, **arrays)
        manifest_samples.append({**episode_meta, "canonical_path": str(episode_path)})

    split = deterministic_split(
        manifest_samples,
        split_seed=int(config.get("split_seed", 20260624)),
        train_per_length=int(config.get("train_per_length", 40)),
        val_per_length=int(config.get("val_per_length", 5)),
        test_per_length=int(config.get("test_per_length", 5)),
    )
    normalization = _fit_stats(canonical_root, manifest_samples, split["train"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "raw_dataset_root": str(raw_root),
        "raw_manifest_path": str(raw_manifest_path),
        "raw_manifest_sha256": sha256_file(raw_manifest_path),
        "canonical_root": str(canonical_root),
        "k_samples_per_segment": k,
        "image_layout": "L,K,C,H,W",
        "image_shape": [3, 128, 128],
        "image_dtype": "uint8",
        "sample_shape": ["length", k, 3, 128, 128],
        "segment_sampling": "timestamp linspace per raw segment; nearest RGB frame; linear continuous targets",
        "num_blocks": num_blocks,
        "token_vocab_size": num_blocks + 1,
        "eos_token_id": eos_token_id,
        "ignore_index": int(config.get("ignore_index", DEFAULT_IGNORE_INDEX)),
        "target_token_rule": "block_order followed by EOS",
        "max_sequence_length": max_sequence_length,
        "max_target_steps": max_sequence_length + 1,
        "joint_dim": 7,
        "joint_names": joint_names,
        "target_fields": [
            "target_tokens",
            "target_xy",
            "block_xy_targets",
            "joint_targets",
            "ee_pose_targets",
            "ee_xy_targets",
            "ee_xy_norm_targets",
        ],
        "target_aliases": {
            "target_xy": "block_xy_targets",
        },
        "target_units": {
            "target_tokens": "integer block ids plus EOS",
            "target_xy": "block xy_norm in [-1, 1], not dataset-standardized",
            "block_xy_targets": "block xy_norm in [-1, 1], not dataset-standardized",
            "joint_targets": "radians, dataset-standardized by default",
            "ee_pose_targets": "world position plus quaternion, dataset-standardized by default",
            "ee_xy_targets": "raw Corsi table coordinates, dataset-standardized by default",
            "ee_xy_norm_targets": "normalized Corsi table coordinates in [-1, 1], not dataset-standardized",
        },
        "model_input_fields": list(MODEL_INPUT_FIELDS),
        "forbidden_model_input_fields": list(FORBIDDEN_MODEL_INPUT_FIELDS),
        "split_seed": int(config.get("split_seed", 20260624)),
        "split": split,
        "length_counts": _length_counts(manifest_samples),
        "split_length_counts": _split_length_counts(manifest_samples, split),
        "normalization": normalization,
        "protected_v1": _protected_v1_snapshot(config),
        "samples": sorted(manifest_samples, key=lambda item: str(item["seq_id"])),
    }
    manifest["canonical_fingerprint"] = sha256_json(
        {
            "schema_version": manifest["schema_version"],
            "raw_manifest_sha256": manifest["raw_manifest_sha256"],
            "k_samples_per_segment": manifest["k_samples_per_segment"],
            "split_seed": manifest["split_seed"],
            "split": manifest["split"],
            "target_token_rule": manifest["target_token_rule"],
            "samples": [
                {
                    "seq_id": item["seq_id"],
                    "length": item["length"],
                    "block_order": item["block_order"],
                    "target_tokens": item["target_tokens"],
                    "canonical_shape": item["canonical_shape"],
                }
                for item in manifest["samples"]
            ],
            "normalization": manifest["normalization"],
        }
    )
    manifest_path = canonical_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (canonical_root / "split_manifest.json").write_text(
        json.dumps({"split": split, "split_length_counts": manifest["split_length_counts"]}, indent=2),
        encoding="utf-8",
    )
    (canonical_root / "normalization.json").write_text(
        json.dumps(normalization, indent=2), encoding="utf-8"
    )
    return {
        "canonical_root": str(canonical_root),
        "manifest_path": str(manifest_path),
        "episode_count": len(manifest_samples),
        "length_counts": manifest["length_counts"],
        "split_counts": {key: len(value) for key, value in split.items()},
        "canonical_fingerprint": manifest["canonical_fingerprint"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build canonical Corsi memory-recall V2 data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = build_canonical_dataset(load_config(args.config), overwrite=bool(args.overwrite))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
