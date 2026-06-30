"""Validation helpers for Corsi memory-recall V2 data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from corsi.experiments.corsi_memory_recall_v2.canonicalize import (
    DEFAULT_EOS_TOKEN_ID,
    DEFAULT_NUM_BLOCKS,
    FORBIDDEN_MODEL_INPUT_FIELDS,
    MODEL_INPUT_FIELDS,
    RAW_SCHEMA_VERSION,
    REQUIRED_RAW_ARRAYS,
    SCHEMA_VERSION,
    _interpolate_rows,
    _nearest_indices,
    _normalize_ee_xy,
    load_config,
    sha256_file,
)


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(name: str, values: np.ndarray) -> None:
    if values.dtype.kind in {"f", "c"}:
        _require(np.isfinite(values).all(), f"{name}: non-finite values")


def _expected_length_counts(config: Mapping[str, Any]) -> dict[str, int] | None:
    if "expected_length_counts" in config:
        return {str(key): int(value) for key, value in config["expected_length_counts"].items()}
    split_counts = _expected_split_length_counts(config)
    if split_counts is not None:
        totals: Counter[str] = Counter()
        for counts in split_counts.values():
            for length, count in counts.items():
                totals[str(length)] += int(count)
        return {str(length): int(totals[str(length)]) for length in sorted(totals, key=int)}
    if {"length_min", "length_max", "num_trials_per_length"} <= set(config):
        per_length = int(config["num_trials_per_length"])
        return {
            str(length): per_length
            for length in range(int(config["length_min"]), int(config["length_max"]) + 1)
        }
    return None


def _expected_split_length_counts(config: Mapping[str, Any]) -> dict[str, dict[str, int]] | None:
    if "expected_split_length_counts" in config:
        return {
            str(split): {str(length): int(count) for length, count in dict(counts).items()}
            for split, counts in dict(config["expected_split_length_counts"]).items()
        }
    if "split_length_counts" in config:
        return {
            str(split): {str(length): int(count) for length, count in dict(counts).items()}
            for split, counts in dict(config["split_length_counts"]).items()
        }
    required_keys = {
        "length_min",
        "length_max",
        "train_per_length",
        "val_per_length",
        "test_per_length",
    }
    if not required_keys <= set(config):
        return None
    lengths = range(int(config["length_min"]), int(config["length_max"]) + 1)
    return {
        "train": {str(length): int(config["train_per_length"]) for length in lengths},
        "val": {str(length): int(config["val_per_length"]) for length in lengths},
        "test": {str(length): int(config["test_per_length"]) for length in lengths},
    }


def validate_raw_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    raw_root = Path(str(config["raw_dataset_root"]))
    manifest_path = raw_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("schema_version") == RAW_SCHEMA_VERSION,
        f"unexpected raw schema {manifest.get('schema_version')!r}",
    )
    failed = list(manifest.get("failed_episodes") or [])
    skipped = list(manifest.get("skipped_episodes") or [])
    _require(not failed, f"raw manifest has failed episodes: {failed[:3]}")
    _require(not skipped, f"raw manifest has skipped episodes: {skipped[:3]}")
    samples = list(manifest.get("samples") or [])
    _require(samples, "raw manifest has no samples")
    expected = _expected_length_counts(config)
    if expected is not None:
        _require(
            {str(k): int(v) for k, v in manifest.get("length_counts", {}).items()} == expected,
            f"raw length_counts {manifest.get('length_counts')} != expected {expected}",
        )

    length_counter: Counter[int] = Counter()
    split_length_counter: dict[str, Counter[int]] = {"train": Counter(), "val": Counter(), "test": Counter()}
    for sample in samples:
        seq_id = str(sample["seq_id"])
        length = int(sample["length"])
        block_order = [int(value) for value in sample["block_order"]]
        num_blocks = int(config.get("num_blocks", DEFAULT_NUM_BLOCKS))
        _require(len(block_order) == length, f"{seq_id}: block_order length mismatch")
        _require(len(block_order) == len(set(block_order)), f"{seq_id}: repeated block in order")
        _require(
            all(0 <= block_id < num_blocks for block_id in block_order),
            f"{seq_id}: block id outside 0..{num_blocks - 1}",
        )
        arrays_path = _resolve(raw_root, str(sample["arrays_path"]))
        metadata_path = _resolve(raw_root, str(sample["metadata_path"]))
        segments_path = _resolve(raw_root, str(sample["segments_path"]))
        _require(arrays_path.exists(), f"{seq_id}: missing arrays file {arrays_path}")
        _require(metadata_path.exists(), f"{seq_id}: missing metadata file {metadata_path}")
        _require(segments_path.exists(), f"{seq_id}: missing segments file {segments_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        segments = json.loads(segments_path.read_text(encoding="utf-8"))
        _require(metadata.get("schema_version") == RAW_SCHEMA_VERSION, f"{seq_id}: bad metadata schema")
        _require(int(metadata["length"]) == length, f"{seq_id}: metadata length mismatch")
        _require([int(v) for v in metadata["block_order"]] == block_order, f"{seq_id}: metadata order mismatch")
        _require(len(segments) == length, f"{seq_id}: expected {length} segments, got {len(segments)}")
        _require(isinstance(metadata.get("block_positions"), dict), f"{seq_id}: missing block_positions")
        _require(
            isinstance((metadata.get("xy_normalization") or {}).get("bounds"), dict),
            f"{seq_id}: missing xy_normalization.bounds",
        )

        with np.load(arrays_path) as arrays:
            missing = [key for key in REQUIRED_RAW_ARRAYS if key not in arrays.files]
            _require(not missing, f"{seq_id}: missing arrays {missing}")
            rgb = arrays["rgb"]
            timestamps = arrays["timestamp"]
            rank_array = arrays["rank"]
            block_array = arrays["block_id"]
            frame_count = int(rgb.shape[0])
            _require(rgb.dtype == np.uint8, f"{seq_id}: rgb dtype {rgb.dtype} != uint8")
            _require(rgb.shape[1:] == (128, 128, 3), f"{seq_id}: rgb shape {rgb.shape}")
            _require(timestamps.shape == (frame_count,), f"{seq_id}: timestamp shape mismatch")
            _finite(f"{seq_id}:timestamp", timestamps)
            _require(rank_array.shape == (frame_count,), f"{seq_id}: rank shape mismatch")
            _require(block_array.shape == (frame_count,), f"{seq_id}: block_id shape mismatch")
            _require(np.all(np.diff(timestamps) > 0.0), f"{seq_id}: timestamps not strictly increasing")
            expected_shapes = {
                "joint": (frame_count, 7),
                "joint_velocity": (frame_count, 7),
                "ee_pose": (frame_count, 7),
                "ee_xy": (frame_count, 2),
                "action": (frame_count, 12),
                "qpos": (frame_count, 19),
                "qvel": (frame_count, 19),
            }
            for key, shape in expected_shapes.items():
                _require(arrays[key].shape == shape, f"{seq_id}: {key} shape {arrays[key].shape} != {shape}")
                _finite(f"{seq_id}:{key}", arrays[key])

            for expected_rank, segment in enumerate(segments):
                rank = int(segment["rank"])
                block_id = int(segment["block_id"])
                start = int(segment["start_frame"])
                end = int(segment["end_frame"])
                _require(rank == expected_rank, f"{seq_id}: segment rank {rank} != {expected_rank}")
                _require(block_id == block_order[rank], f"{seq_id}: segment block mismatch")
                _require(0 <= start <= end < frame_count, f"{seq_id}: invalid segment bounds")
                segment_slice = slice(start, end + 1)
                _require(
                    np.all(rank_array[segment_slice] == rank),
                    f"{seq_id}: rank array disagrees with segment {rank}",
                )
                _require(
                    np.all(block_array[segment_slice] == block_id),
                    f"{seq_id}: block array disagrees with segment {rank}",
                )
        length_counter[length] += 1
        split_name = sample.get("split")
        if split_name is not None:
            split_name = str(split_name)
            _require(split_name in split_length_counter, f"{seq_id}: unexpected split {split_name!r}")
            split_length_counter[split_name][length] += 1

    computed_length_counts = {str(k): int(v) for k, v in sorted(length_counter.items())}
    if expected is not None:
        _require(
            computed_length_counts == expected,
            f"computed raw length counts {computed_length_counts} != expected {expected}",
        )
        _require(
            len(samples) == sum(expected.values()),
            f"raw sample count {len(samples)} != expected {sum(expected.values())}",
        )
    expected_split_counts = (
        _expected_split_length_counts(config)
        if "expected_split_length_counts" in config or "split_length_counts" in config
        else None
    )
    computed_split_length_counts = {
        split: {str(length): int(count) for length, count in sorted(counter.items())}
        for split, counter in split_length_counter.items()
        if counter
    }
    if expected_split_counts is not None:
        _require(
            computed_split_length_counts == expected_split_counts,
            f"raw split_length_counts {computed_split_length_counts} != expected {expected_split_counts}",
        )

    result = {
        "check": "raw",
        "manifest_path": str(manifest_path),
        "episode_count": len(samples),
        "length_counts": computed_length_counts,
        "status": "ok",
    }
    if computed_split_length_counts:
        result["split_length_counts"] = computed_split_length_counts
    return result


def _canonical_manifest_path(config: Mapping[str, Any]) -> Path:
    if "canonical_manifest_path" in config:
        return Path(str(config["canonical_manifest_path"]))
    return Path(str(config["canonical_root"])) / "manifest.json"


def _split_sets(split: Mapping[str, Sequence[str]]) -> dict[str, set[str]]:
    return {key: set(str(value) for value in values) for key, values in split.items()}


def _assert_allclose(name: str, actual: np.ndarray, expected: np.ndarray, *, atol: float = 1e-5) -> None:
    _require(actual.shape == expected.shape, f"{name}: shape {actual.shape} != {expected.shape}")
    _require(np.allclose(actual, expected, atol=atol, rtol=0.0), f"{name}: values do not match raw source")


def _validate_canonical_raw_alignment(
    *,
    raw_root: Path,
    seq_id: str,
    sample: Mapping[str, Any],
    arrays,
    k: int,
) -> None:
    arrays_path = _resolve(raw_root, str(sample["raw_arrays_path"]))
    metadata_path = _resolve(raw_root, str(sample["raw_metadata_path"]))
    segments_path = _resolve(raw_root, str(sample["raw_segments_path"]))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    with np.load(arrays_path) as raw:
        raw_rgb = np.asarray(raw["rgb"])
        raw_joint = np.asarray(raw["joint"], dtype=np.float32)
        raw_ee_pose = np.asarray(raw["ee_pose"], dtype=np.float32)
        raw_ee_xy = np.asarray(raw["ee_xy"], dtype=np.float32)
        raw_timestamp = np.asarray(raw["timestamp"], dtype=np.float64)
        raw_rank = np.asarray(raw["rank"], dtype=np.int64)
        raw_block = np.asarray(raw["block_id"], dtype=np.int64)

    raw_ee_xy_norm = _normalize_ee_xy(raw_ee_xy, metadata)
    tau = np.linspace(0.0, 1.0, int(k), dtype=np.float64)
    block_order = [int(value) for value in sample["block_order"]]
    for expected_rank, segment in enumerate(segments):
        rank = int(segment["rank"])
        block_id = int(segment["block_id"])
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        _require(rank == expected_rank, f"{seq_id}: raw segment rank {rank} != {expected_rank}")
        _require(block_id == block_order[rank], f"{seq_id}: raw segment block mismatch at rank {rank}")
        source_indices = np.asarray(arrays["source_frame_index"][rank], dtype=np.int64)
        segment_slice = slice(start, end + 1)
        _require(np.all(raw_rank[segment_slice] == rank), f"{seq_id}: raw rank mismatch at segment {rank}")
        _require(np.all(raw_block[segment_slice] == block_id), f"{seq_id}: raw block mismatch at segment {rank}")
        seg_timestamps = raw_timestamp[segment_slice]
        target_times = seg_timestamps[0] + tau * (seg_timestamps[-1] - seg_timestamps[0])
        expected_indices = _nearest_indices(seg_timestamps, target_times) + start
        _require(
            source_indices.tolist() == expected_indices.tolist(),
            f"{seq_id}: source_frame_index mismatch at rank {rank}",
        )
        _assert_allclose(
            f"{seq_id}:source_timestamp[{rank}]",
            np.asarray(arrays["source_timestamp"][rank], dtype=np.float64),
            raw_timestamp[expected_indices].astype(np.float64),
            atol=1e-9,
        )
        _require(
            np.array_equal(arrays["images"][rank], np.transpose(raw_rgb[expected_indices], (0, 3, 1, 2))),
            f"{seq_id}: images do not match raw source at rank {rank}",
        )
        _assert_allclose(
            f"{seq_id}:joint_targets[{rank}]",
            arrays["joint_targets"][rank],
            _interpolate_rows(seg_timestamps, raw_joint[segment_slice], target_times),
        )
        _assert_allclose(
            f"{seq_id}:ee_pose_targets[{rank}]",
            arrays["ee_pose_targets"][rank],
            _interpolate_rows(seg_timestamps, raw_ee_pose[segment_slice], target_times),
        )
        _assert_allclose(
            f"{seq_id}:ee_xy_targets[{rank}]",
            arrays["ee_xy_targets"][rank],
            _interpolate_rows(seg_timestamps, raw_ee_xy[segment_slice], target_times),
        )
        _assert_allclose(
            f"{seq_id}:ee_xy_norm_targets[{rank}]",
            arrays["ee_xy_norm_targets"][rank],
            _interpolate_rows(seg_timestamps, raw_ee_xy_norm[segment_slice], target_times),
        )
        position = metadata["block_positions"][str(block_id)]
        _assert_allclose(
            f"{seq_id}:block_xy_targets[{rank}]",
            arrays["block_xy_targets"][rank],
            np.asarray(position["xy_norm"], dtype=np.float32),
        )
        _assert_allclose(
            f"{seq_id}:block_xyz_world_targets[{rank}]",
            arrays["block_xyz_world_targets"][rank],
            np.asarray(position["xyz_world"], dtype=np.float32),
        )


def validate_canonical_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = _canonical_manifest_path(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION,
        f"unexpected canonical schema {manifest.get('schema_version')!r}",
    )
    canonical_root = Path(str(manifest["canonical_root"]))
    k = int(manifest["k_samples_per_segment"])
    eos = int(manifest.get("eos_token_id", DEFAULT_EOS_TOKEN_ID))
    split = _split_sets(manifest["split"])
    _require(set(split) == {"train", "val", "test"}, f"unexpected split keys {sorted(split)}")
    _require(split["train"].isdisjoint(split["val"]), "train/val overlap")
    _require(split["train"].isdisjoint(split["test"]), "train/test overlap")
    _require(split["val"].isdisjoint(split["test"]), "val/test overlap")
    all_split_ids = set().union(*split.values())
    sample_ids = {str(sample["seq_id"]) for sample in manifest["samples"]}
    _require(all_split_ids == sample_ids, "split ids do not match sample ids")
    sample_by_id = {str(sample["seq_id"]): sample for sample in manifest["samples"]}
    computed_split_length_counts: dict[str, dict[str, int]] = {}
    for split_name, ids in split.items():
        counter = Counter(int(sample_by_id[seq_id]["length"]) for seq_id in ids)
        computed_split_length_counts[split_name] = {
            str(length): int(count) for length, count in sorted(counter.items())
        }
    computed_length_counts = {
        str(length): int(count)
        for length, count in sorted(Counter(int(sample["length"]) for sample in manifest["samples"]).items())
    }
    expected_lengths = _expected_length_counts(config)
    if expected_lengths is not None:
        _require(
            computed_length_counts == expected_lengths,
            f"computed canonical length counts {computed_length_counts} != expected {expected_lengths}",
        )
        _require(
            len(manifest["samples"]) == sum(expected_lengths.values()),
            f"canonical sample count {len(manifest['samples'])} != expected {sum(expected_lengths.values())}",
        )
    _require(
        computed_length_counts == manifest.get("length_counts"),
        "manifest length_counts do not match samples",
    )
    _require(
        computed_split_length_counts == manifest.get("split_length_counts"),
        "split_length_counts do not match manifest samples",
    )
    expected_split_counts = _expected_split_length_counts(config)
    if expected_split_counts is not None:
        _require(
            computed_split_length_counts == expected_split_counts,
            f"split_length_counts {computed_split_length_counts} != expected {expected_split_counts}",
        )
    _require(manifest["normalization"].get("source") == "train_only", "normalization is not train_only")
    _require(manifest.get("model_input_fields") == MODEL_INPUT_FIELDS, "unexpected model_input_fields")

    raw_manifest = Path(str(manifest["raw_manifest_path"]))
    if raw_manifest.exists():
        _require(
            sha256_file(raw_manifest) == manifest["raw_manifest_sha256"],
            "raw manifest sha256 changed since canonicalization",
        )
    protected_v1 = manifest.get("protected_v1") or {}
    protected_path = Path(str(protected_v1.get("manifest_path", "")))
    protected_sha = protected_v1.get("manifest_sha256")
    if protected_sha and protected_path.exists():
        _require(
            sha256_file(protected_path) == protected_sha,
            "protected V1 manifest sha256 changed since canonicalization",
        )

    raw_root = Path(str(manifest["raw_dataset_root"]))
    for sample in manifest["samples"]:
        seq_id = str(sample["seq_id"])
        length = int(sample["length"])
        block_order = [int(value) for value in sample["block_order"]]
        path = Path(str(sample["canonical_path"]))
        if not path.is_absolute() and not path.exists():
            path = canonical_root / "episodes" / path.name
        _require(path.exists(), f"{seq_id}: missing canonical file {path}")
        with np.load(path) as arrays:
            expected_keys = {
                "images",
                "joint_targets",
                "ee_pose_targets",
                "ee_xy_targets",
                "ee_xy_norm_targets",
                "block_xy_targets",
                "block_xyz_world_targets",
                "target_tokens",
                "target_token_mask",
                "segment_mask",
                "frame_mask",
                "rank",
                "block_id",
                "segment_progress",
                "source_frame_index",
                "source_timestamp",
                "segment_start_frame",
                "segment_end_frame",
            }
            missing = sorted(expected_keys - set(arrays.files))
            _require(not missing, f"{seq_id}: missing canonical arrays {missing}")
            _require(arrays["images"].shape == (length, k, 3, 128, 128), f"{seq_id}: image shape")
            _require(arrays["images"].dtype == np.uint8, f"{seq_id}: images not uint8")
            _require(arrays["joint_targets"].shape == (length, k, 7), f"{seq_id}: joint shape")
            _require(arrays["ee_pose_targets"].shape == (length, k, 7), f"{seq_id}: ee_pose shape")
            _require(arrays["ee_xy_targets"].shape == (length, k, 2), f"{seq_id}: ee_xy shape")
            _require(arrays["ee_xy_norm_targets"].shape == (length, k, 2), f"{seq_id}: ee_xy_norm shape")
            _require(arrays["block_xy_targets"].shape == (length, 2), f"{seq_id}: block_xy shape")
            _require(arrays["block_xyz_world_targets"].shape == (length, 3), f"{seq_id}: block_xyz shape")
            _require(arrays["target_tokens"].tolist() == block_order + [eos], f"{seq_id}: target tokens")
            _require(arrays["target_token_mask"].shape == (length + 1,), f"{seq_id}: token mask shape")
            _require(arrays["target_token_mask"].all(), f"{seq_id}: token mask contains false")
            _require(arrays["segment_mask"].shape == (length,), f"{seq_id}: segment mask shape")
            _require(arrays["segment_mask"].all(), f"{seq_id}: segment mask contains false")
            _require(arrays["frame_mask"].shape == (length, k), f"{seq_id}: frame mask shape")
            _require(arrays["frame_mask"].all(), f"{seq_id}: frame mask contains false")
            _require(arrays["rank"].tolist() == list(range(length)), f"{seq_id}: rank mismatch")
            _require(arrays["block_id"].tolist() == block_order, f"{seq_id}: block_id mismatch")
            for key in [
                "joint_targets",
                "ee_pose_targets",
                "ee_xy_targets",
                "ee_xy_norm_targets",
                "block_xy_targets",
                "block_xyz_world_targets",
                "segment_progress",
                "source_timestamp",
            ]:
                _finite(f"{seq_id}:{key}", arrays[key])
            _validate_canonical_raw_alignment(
                raw_root=raw_root,
                seq_id=seq_id,
                sample=sample,
                arrays=arrays,
                k=k,
            )

    return {
        "check": "canonical",
        "manifest_path": str(manifest_path),
        "episode_count": len(manifest["samples"]),
        "split_counts": {key: len(values) for key, values in manifest["split"].items()},
        "status": "ok",
    }


def validate_leakage(config: Mapping[str, Any]) -> dict[str, Any]:
    from corsi.experiments.corsi_memory_recall_v2.dataset import (
        CorsiMemoryRecallV2Dataset,
        collate_memory_recall_v2_batch,
    )

    manifest_path = _canonical_manifest_path(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("model_input_fields") == MODEL_INPUT_FIELDS, "manifest input fields changed")
    forbidden = set(FORBIDDEN_MODEL_INPUT_FIELDS)
    _require(
        not forbidden.intersection(manifest["model_input_fields"]),
        f"forbidden manifest input fields: {forbidden.intersection(manifest['model_input_fields'])}",
    )
    split = "train" if manifest.get("split", {}).get("train") else None
    dataset = CorsiMemoryRecallV2Dataset(manifest_path, split=split)
    _require(len(dataset) > 0, "cannot leakage-check empty dataset")
    batch = collate_memory_recall_v2_batch([dataset[index] for index in range(min(2, len(dataset)))])
    _require(set(batch) >= {"model_inputs", "targets", "metadata"}, "collate sections missing")
    input_keys = set(batch["model_inputs"].keys())
    _require(input_keys == set(MODEL_INPUT_FIELDS), f"unexpected model input keys {sorted(input_keys)}")
    _require(
        not forbidden.intersection(input_keys),
        f"forbidden model input keys: {sorted(forbidden.intersection(input_keys))}",
    )
    target_keys = set(batch["targets"].keys())
    _require(
        {"tokens", "target_xy", "block_xy"} <= target_keys,
        "required targets missing",
    )
    _require("block_id" in batch["metadata"] and "rank" in batch["metadata"], "metadata labels missing")
    return {
        "check": "leakage",
        "manifest_path": str(manifest_path),
        "model_input_keys": sorted(input_keys),
        "target_keys": sorted(target_keys),
        "status": "ok",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Corsi memory-recall V2 data.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--check",
        choices=["raw", "canonical", "leakage", "all"],
        default="all",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    results: dict[str, Any] = {}
    if args.check in {"raw", "all"}:
        results["raw"] = validate_raw_dataset(config)
    if args.check in {"canonical", "all"}:
        results["canonical"] = validate_canonical_dataset(config)
    if args.check in {"leakage", "all"}:
        results["leakage"] = validate_leakage(config)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
