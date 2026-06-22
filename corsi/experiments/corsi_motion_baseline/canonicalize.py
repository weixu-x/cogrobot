"""Create the deterministic 7-joint canonical Corsi motion dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "scala_corsi_motion_canonical_7joint_v1"
DEFAULT_JOINT_NAMES = [
    "robot0_joint1",
    "robot0_joint2",
    "robot0_joint3",
    "robot0_joint4",
    "robot0_joint5",
    "robot0_joint6",
    "robot0_joint7",
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


def _resolve(raw_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = raw_root / path
    if candidate.exists():
        return candidate
    return path


def _interpolate_rows(timestamps: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    columns = [
        np.interp(target_times, timestamps, values[:, dim]).astype(np.float32)
        for dim in range(values.shape[1])
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def _nearest_indices(timestamps: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(timestamps, target_times, side="left")
    indices = np.clip(indices, 0, len(timestamps) - 1)
    previous = np.clip(indices - 1, 0, len(timestamps) - 1)
    choose_previous = np.abs(timestamps[previous] - target_times) <= np.abs(timestamps[indices] - target_times)
    return np.where(choose_previous, previous, indices).astype(np.int64)


def canonicalize_episode(
    *,
    raw_root: Path,
    sample: dict[str, Any],
    k: int,
    joint_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays_path = _resolve(raw_root, str(sample["arrays_path"]))
    metadata_path = _resolve(raw_root, str(sample["metadata_path"]))
    segments_path = _resolve(raw_root, str(sample["segments_path"]))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    block_order = [int(v) for v in sample["block_order"]]
    length = int(sample["length"])
    if len(segments) != length:
        raise ValueError(f"{sample['seq_id']}: segment count {len(segments)} != length {length}")

    with np.load(arrays_path) as raw:
        rgb = np.asarray(raw["rgb"])
        joints = np.asarray(raw["joint"], dtype=np.float32)
        timestamps = np.asarray(raw["timestamp"], dtype=np.float64)
        rank_array = np.asarray(raw["rank"], dtype=np.int64)
        block_array = np.asarray(raw["block_id"], dtype=np.int64)

    if joints.shape[1] != len(joint_names):
        raise ValueError(
            f"{sample['seq_id']}: joint width {joints.shape[1]} != configured joint names {len(joint_names)}"
        )
    if rgb.ndim != 4 or rgb.shape[1:] != (128, 128, 3):
        raise ValueError(f"{sample['seq_id']}: expected raw rgb [T,128,128,3], got {rgb.shape}")

    images: list[np.ndarray] = []
    joint_rows: list[np.ndarray] = []
    rank_rows: list[int] = []
    block_rows: list[int] = []
    block_xy_rows: list[list[float]] = []
    progress_rows: list[float] = []
    source_frame_rows: list[int] = []
    source_time_rows: list[float] = []
    boundary_rows: list[bool] = []

    tau = np.linspace(0.0, 1.0, int(k), dtype=np.float64)
    block_positions = metadata.get("block_positions", {})
    for expected_rank, segment in enumerate(segments):
        rank = int(segment["rank"])
        block_id = int(segment["block_id"])
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if rank != expected_rank:
            raise ValueError(f"{sample['seq_id']}: rank {rank} != expected {expected_rank}")
        if block_id != block_order[rank]:
            raise ValueError(f"{sample['seq_id']}: block_id {block_id} != block_order[{rank}]")
        if start < 0 or end < start or end >= len(timestamps):
            raise ValueError(f"{sample['seq_id']}: invalid segment bounds {start}..{end}")
        if end - start + 1 < 2:
            raise ValueError(f"{sample['seq_id']}: segment {rank} has fewer than two observations")
        segment_slice = slice(start, end + 1)
        seg_timestamps = timestamps[segment_slice]
        seg_joints = joints[segment_slice]
        if np.any(rank_array[segment_slice] != rank) or np.any(block_array[segment_slice] != block_id):
            raise ValueError(f"{sample['seq_id']}: raw rank/block arrays disagree with segment {rank}")
        target_times = seg_timestamps[0] + tau * (seg_timestamps[-1] - seg_timestamps[0])
        local_image_indices = _nearest_indices(seg_timestamps, target_times)
        global_image_indices = local_image_indices + start
        interpolated_joints = _interpolate_rows(seg_timestamps, seg_joints, target_times)
        xy_norm = block_positions[str(block_id)]["xy_norm"]
        for local_index, global_index in enumerate(global_image_indices.tolist()):
            images.append(np.transpose(rgb[global_index], (2, 0, 1)))
            joint_rows.append(interpolated_joints[local_index])
            rank_rows.append(rank)
            block_rows.append(block_id)
            block_xy_rows.append([float(xy_norm[0]), float(xy_norm[1])])
            progress_rows.append(float(tau[local_index]))
            source_frame_rows.append(int(global_index))
            source_time_rows.append(float(timestamps[global_index]))
            boundary_rows.append(local_index == 0 and rank > 0)

    canonical_images = np.stack(images, axis=0).astype(np.uint8)
    canonical_joints = np.stack(joint_rows, axis=0).astype(np.float32)
    ranks = np.asarray(rank_rows, dtype=np.int64)
    block_ids = np.asarray(block_rows, dtype=np.int64)
    source_frames = np.asarray(source_frame_rows, dtype=np.int64)
    source_timestamps = np.asarray(source_time_rows, dtype=np.float64)
    segment_progress = np.asarray(progress_rows, dtype=np.float32)
    block_xy = np.asarray(block_xy_rows, dtype=np.float32)
    boundary = np.asarray(boundary_rows, dtype=bool)

    total_steps = canonical_joints.shape[0]
    transition = np.zeros((total_steps,), dtype=bool)
    transition[:-1] = True
    within_segment = np.zeros((total_steps,), dtype=bool)
    segment_boundary = np.zeros((total_steps,), dtype=bool)
    within_segment[:-1] = ranks[:-1] == ranks[1:]
    segment_boundary[:-1] = ranks[:-1] != ranks[1:]

    arrays = {
        "images": canonical_images,
        "joints": canonical_joints,
        "transition_mask": transition,
        "within_segment_transition": within_segment,
        "segment_boundary_transition": segment_boundary,
        "rank": ranks,
        "block_id": block_ids,
        "block_xy": block_xy,
        "segment_progress": segment_progress,
        "source_frame_index": source_frames,
        "source_timestamp": source_timestamps,
        "boundary": boundary,
    }
    episode_meta = {
        "seq_id": str(sample["seq_id"]),
        "length": length,
        "block_order": block_order,
        "layout_id": str(sample["layout_id"]),
        "seed": int(sample["seed"]),
        "raw_arrays_path": str(arrays_path),
        "raw_metadata_path": str(metadata_path),
        "raw_segments_path": str(segments_path),
        "canonical_T": int(total_steps),
        "k_samples_per_segment": int(k),
        "joint_names": list(joint_names),
    }
    return arrays, episode_meta


def deterministic_split(
    samples: list[dict[str, Any]],
    *,
    split_seed: int,
    train_per_length: int,
    val_per_length: int,
    test_per_length: int,
) -> dict[str, list[str]]:
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_orders: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    for sample in samples:
        length = int(sample["length"])
        order = tuple(int(v) for v in sample["block_order"])
        if order in seen_orders[length]:
            raise ValueError(f"duplicate block_order within length {length}: {order}")
        seen_orders[length].add(order)
        by_length[length].append(sample)

    split = {"train": [], "val": [], "test": []}
    required = int(train_per_length) + int(val_per_length) + int(test_per_length)
    for length, rows in sorted(by_length.items()):
        if len(rows) != required:
            raise ValueError(f"length {length}: expected {required} episodes, got {len(rows)}")
        rows = sorted(rows, key=lambda item: str(item["seq_id"]))
        rng = random.Random(int(split_seed) + length)
        rng.shuffle(rows)
        train_rows = rows[:train_per_length]
        val_rows = rows[train_per_length : train_per_length + val_per_length]
        test_rows = rows[train_per_length + val_per_length :]
        split["train"].extend(str(item["seq_id"]) for item in train_rows)
        split["val"].extend(str(item["seq_id"]) for item in val_rows)
        split["test"].extend(str(item["seq_id"]) for item in test_rows)
    for key in split:
        split[key] = sorted(split[key])
    return split


def _fit_stats(canonical_root: Path, train_seq_ids: Iterable[str]) -> dict[str, Any]:
    image_sum = np.zeros((3,), dtype=np.float64)
    image_sumsq = np.zeros((3,), dtype=np.float64)
    image_count = 0
    joint_sum = None
    joint_sumsq = None
    joint_count = 0
    for seq_id in train_seq_ids:
        with np.load(canonical_root / "episodes" / f"{seq_id}.npz") as arrays:
            images = arrays["images"].astype(np.float64) / 255.0
            pixels = images.transpose(1, 0, 2, 3).reshape(3, -1)
            image_sum += pixels.sum(axis=1)
            image_sumsq += (pixels * pixels).sum(axis=1)
            image_count += pixels.shape[1]
            joints = arrays["joints"].astype(np.float64)
            if joint_sum is None:
                joint_sum = np.zeros((joints.shape[1],), dtype=np.float64)
                joint_sumsq = np.zeros((joints.shape[1],), dtype=np.float64)
            joint_sum += joints.sum(axis=0)
            joint_sumsq += (joints * joints).sum(axis=0)
            joint_count += joints.shape[0]
    if joint_sum is None or joint_sumsq is None or image_count == 0 or joint_count == 0:
        raise ValueError("cannot fit normalization statistics on empty train split")
    image_mean = image_sum / image_count
    image_var = np.maximum(image_sumsq / image_count - image_mean * image_mean, 1e-12)
    joint_mean = joint_sum / joint_count
    joint_var = np.maximum(joint_sumsq / joint_count - joint_mean * joint_mean, 1e-12)
    return {
        "image_mean": image_mean.tolist(),
        "image_std": np.sqrt(image_var).tolist(),
        "joint_mean": joint_mean.tolist(),
        "joint_std": np.sqrt(joint_var).tolist(),
        "source": "train_only",
        "train_image_values": int(image_count),
        "train_joint_rows": int(joint_count),
    }


def build_canonical_dataset(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    raw_root = Path(str(config["raw_dataset_root"]))
    canonical_root = Path(str(config["canonical_root"]))
    raw_manifest_path = raw_root / "manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    joint_names = list(config.get("joint_names") or DEFAULT_JOINT_NAMES)
    if len(joint_names) != 7:
        raise ValueError("revised baseline requires exactly seven joint names")
    k = int(config.get("k_samples_per_segment", 12))
    if k < 2:
        raise ValueError("k_samples_per_segment must be at least 2")
    if overwrite and canonical_root.exists():
        shutil.rmtree(canonical_root)
    (canonical_root / "episodes").mkdir(parents=True, exist_ok=True)

    manifest_samples = []
    for sample in raw_manifest["samples"]:
        seq_id = str(sample["seq_id"])
        arrays, episode_meta = canonicalize_episode(
            raw_root=raw_root,
            sample=sample,
            k=k,
            joint_names=joint_names,
        )
        episode_path = canonical_root / "episodes" / f"{seq_id}.npz"
        np.savez_compressed(episode_path, **arrays)
        manifest_samples.append(
            {
                **episode_meta,
                "canonical_path": str(episode_path),
            }
        )

    split = deterministic_split(
        manifest_samples,
        split_seed=int(config.get("split_seed", 20260622)),
        train_per_length=int(config.get("train_per_length", 40)),
        val_per_length=int(config.get("val_per_length", 5)),
        test_per_length=int(config.get("test_per_length", 5)),
    )
    stats = _fit_stats(canonical_root, split["train"])
    length_counts = Counter(int(item["length"]) for item in manifest_samples)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "raw_dataset_root": str(raw_root),
        "raw_manifest_path": str(raw_manifest_path),
        "raw_manifest_sha256": sha256_file(raw_manifest_path),
        "canonical_root": str(canonical_root),
        "k_samples_per_segment": k,
        "joint_dim": 7,
        "joint_names": joint_names,
        "image_layout": "CHW",
        "image_shape": [3, 128, 128],
        "image_dtype": "uint8",
        "interpolation_rule": "timestamp linear interpolation for joints; nearest timestamp for image",
        "resize_rule": "none; raw frames are already 128x128",
        "length_counts": {str(k): int(v) for k, v in sorted(length_counts.items())},
        "split_seed": int(config.get("split_seed", 20260622)),
        "split": split,
        "normalization": stats,
        "samples": sorted(manifest_samples, key=lambda item: item["seq_id"]),
    }
    manifest["canonical_fingerprint"] = sha256_json(
        {
            "schema_version": manifest["schema_version"],
            "raw_manifest_sha256": manifest["raw_manifest_sha256"],
            "k_samples_per_segment": manifest["k_samples_per_segment"],
            "joint_names": manifest["joint_names"],
            "split_seed": manifest["split_seed"],
            "split": manifest["split"],
            "samples": [
                {
                    "seq_id": item["seq_id"],
                    "length": item["length"],
                    "block_order": item["block_order"],
                    "canonical_T": item["canonical_T"],
                }
                for item in manifest["samples"]
            ],
            "normalization": manifest["normalization"],
        }
    )
    manifest_path = canonical_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    split_path = canonical_root / "split_manifest.json"
    split_path.write_text(json.dumps({"split": split}, indent=2), encoding="utf-8")
    stats_path = canonical_root / "normalization.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return {
        "canonical_root": str(canonical_root),
        "manifest_path": str(manifest_path),
        "episode_count": len(manifest_samples),
        "length_counts": manifest["length_counts"],
        "split_counts": {key: len(value) for key, value in split.items()},
        "canonical_fingerprint": manifest["canonical_fingerprint"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build canonical 7-joint Corsi motion dataset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = build_canonical_dataset(load_config(args.config), overwrite=bool(args.overwrite))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
