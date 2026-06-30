import json
from pathlib import Path

import numpy as np
import pytest
import torch

from corsi.experiments.corsi_memory_recall_v2.canonicalize import (
    DEFAULT_JOINT_NAMES,
    SCHEMA_VERSION,
    build_canonical_dataset,
)
from corsi.experiments.corsi_memory_recall_v2.dataset import (
    CorsiMemoryRecallV2Dataset,
    collate_memory_recall_v2_batch,
)
from corsi.experiments.corsi_memory_recall_v2.validate import (
    validate_canonical_dataset,
    validate_leakage,
    validate_raw_dataset,
)


def _xy_norm(x: float, y: float) -> list[float]:
    return [2.0 * x / 255.0 - 1.0, 2.0 * y / 205.0 - 1.0]


def _write_synthetic_raw(root: Path) -> None:
    samples = []
    for length in [2, 3]:
        for trial in range(3):
            seq_id = f"len{length:02d}_trial{trial:03d}"
            episode_dir = root / "episodes" / seq_id
            episode_dir.mkdir(parents=True)
            frames_per_segment = 4
            total = length * frames_per_segment
            timestamps = np.arange(total, dtype=np.float64) * 0.1 + trial
            joints = np.stack(
                [timestamps.astype(np.float32) * 10.0 + dim for dim in range(7)],
                axis=1,
            ).astype(np.float32)
            joint_velocity = np.full((total, 7), 0.25 + trial, dtype=np.float32)
            ee_pose = np.stack(
                [timestamps.astype(np.float32) * 2.0 + dim * 0.01 for dim in range(7)],
                axis=1,
            ).astype(np.float32)
            ee_xy = np.stack(
                [50.0 + timestamps.astype(np.float32), 40.0 + timestamps.astype(np.float32) * 2.0],
                axis=1,
            ).astype(np.float32)
            action = np.full((total, 12), trial, dtype=np.float32)
            qpos = np.full((total, 19), length, dtype=np.float32)
            qvel = np.zeros((total, 19), dtype=np.float32)
            rgb = np.zeros((total, 128, 128, 3), dtype=np.uint8)
            for frame_index in range(total):
                rgb[frame_index, :, :, :] = frame_index + trial
            ranks = np.repeat(np.arange(length, dtype=np.int64), frames_per_segment)
            block_order = list(range(trial, trial + length))
            block_ids = np.repeat(np.asarray(block_order, dtype=np.int64), frames_per_segment)
            arrays_path = episode_dir / "arrays.npz"
            np.savez_compressed(
                arrays_path,
                rgb=rgb,
                joint=joints,
                joint_velocity=joint_velocity,
                ee_pose=ee_pose,
                ee_xy=ee_xy,
                action=action,
                qpos=qpos,
                qvel=qvel,
                timestamp=timestamps,
                rank=ranks,
                block_id=block_ids,
            )
            segments = [
                {
                    "segment_id": rank,
                    "rank": rank,
                    "block_id": block_order[rank],
                    "start_frame": rank * frames_per_segment,
                    "end_frame": rank * frames_per_segment + frames_per_segment - 1,
                }
                for rank in range(length)
            ]
            block_positions = {}
            for block_id in range(9):
                x = 10.0 + block_id * 20.0
                y = 15.0 + block_id * 10.0
                block_positions[str(block_id)] = {
                    "body_name": f"corsi_block_{block_id}",
                    "xyz_world": [float(block_id), float(block_id) + 0.5, 0.8],
                    "xy_table": [x, y],
                    "xy_norm": _xy_norm(x, y),
                }
            metadata = {
                "schema_version": "scala_corsi_motion_raw_v1",
                "seq_id": seq_id,
                "length": length,
                "block_order": block_order,
                "layout_id": "fixed_9block_v1",
                "seed": 2000 + length * 10 + trial,
                "frame_count": total,
                "camera_name": "freecam",
                "image_shape": [128, 128, 3],
                "array_keys": [
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
                ],
                "block_positions": block_positions,
                "xy_normalization": {
                    "frame": "corsi_lower_left",
                    "bounds": {"x_min": 0.0, "x_max": 255.0, "y_min": 0.0, "y_max": 205.0},
                    "range": [-1.0, 1.0],
                },
                "segments": segments,
            }
            metadata_path = episode_dir / "metadata.json"
            segments_path = episode_dir / "segments.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            segments_path.write_text(json.dumps(segments), encoding="utf-8")
            samples.append(
                {
                    "seq_id": seq_id,
                    "length": length,
                    "block_order": block_order,
                    "layout_id": "fixed_9block_v1",
                    "seed": metadata["seed"],
                    "frame_count": total,
                    "arrays_path": str(arrays_path),
                    "metadata_path": str(metadata_path),
                    "segments_path": str(segments_path),
                    "status": "existing",
                }
            )
    manifest = {
        "schema_version": "scala_corsi_motion_raw_v1",
        "dataset_name": "synthetic_memory_recall_v2",
        "dataset_root": str(root),
        "num_episodes": len(samples),
        "expected_episodes": len(samples),
        "length_counts": {"2": 3, "3": 3},
        "camera_name": "freecam",
        "array_keys": [
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
        ],
        "failed_episodes": [],
        "skipped_episodes": [],
        "samples": samples,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _tiny_config(tmp_path: Path) -> dict:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _write_synthetic_raw(raw_root)
    return {
        "raw_dataset_root": str(raw_root),
        "canonical_root": str(tmp_path / "canonical"),
        "joint_names": DEFAULT_JOINT_NAMES,
        "k_samples_per_segment": 3,
        "num_blocks": 9,
        "eos_token_id": 9,
        "ignore_index": -100,
        "max_sequence_length": 3,
        "length_min": 2,
        "length_max": 3,
        "num_trials_per_length": 3,
        "split_seed": 13,
        "train_per_length": 1,
        "val_per_length": 1,
        "test_per_length": 1,
    }


def _build_tiny_canonical(tmp_path: Path) -> tuple[dict, Path]:
    config = _tiny_config(tmp_path)
    result = build_canonical_dataset(config, overwrite=True)
    assert result["split_counts"] == {"train": 2, "val": 2, "test": 2}
    return config, Path(result["manifest_path"])


def test_canonical_sampling_schema_and_targets(tmp_path):
    _, manifest_path = _build_tiny_canonical(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["model_input_fields"] == ["images", "segment_mask", "frame_mask"]
    sample = next(item for item in manifest["samples"] if item["seq_id"] == "len02_trial000")
    with np.load(sample["canonical_path"]) as arrays:
        assert arrays["images"].shape == (2, 3, 3, 128, 128)
        assert arrays["joint_targets"].shape == (2, 3, 7)
        assert arrays["ee_pose_targets"].shape == (2, 3, 7)
        assert arrays["ee_xy_targets"].shape == (2, 3, 2)
        assert arrays["ee_xy_norm_targets"].shape == (2, 3, 2)
        assert arrays["target_tokens"].tolist() == [0, 1, 9]
        assert arrays["target_token_mask"].tolist() == [True, True, True]
        assert arrays["segment_mask"].tolist() == [True, True]
        assert arrays["frame_mask"].tolist() == [[True, True, True], [True, True, True]]
        np.testing.assert_allclose(arrays["joint_targets"][0, :, 0], [0.0, 1.5, 3.0], atol=1e-6)
        assert arrays["source_frame_index"][0].tolist() == [0, 2, 3]
        np.testing.assert_allclose(arrays["block_xy_targets"][0], _xy_norm(10.0, 15.0))


def test_episode_split_and_train_only_normalization(tmp_path):
    _, manifest_path = _build_tiny_canonical(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["split_length_counts"] == {
        "train": {"2": 1, "3": 1},
        "val": {"2": 1, "3": 1},
        "test": {"2": 1, "3": 1},
    }
    assert set(manifest["split"]["train"]).isdisjoint(manifest["split"]["val"])
    train_ids = set(manifest["split"]["train"])
    train_joint_rows = []
    for sample in manifest["samples"]:
        if sample["seq_id"] in train_ids:
            with np.load(sample["canonical_path"]) as arrays:
                train_joint_rows.append(arrays["joint_targets"].reshape(-1, 7))
    train_joints = np.concatenate(train_joint_rows, axis=0)
    np.testing.assert_allclose(manifest["normalization"]["joint_mean"], train_joints.mean(axis=0), atol=1e-6)
    assert manifest["normalization"]["source"] == "train_only"


def test_canonical_preserves_explicit_raw_split_labels(tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _write_synthetic_raw(raw_root)
    manifest_path = raw_root / "manifest.json"
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_by_trial = {"000": "train", "001": "val", "002": "test"}
    for sample in raw_manifest["samples"]:
        split = split_by_trial[str(sample["seq_id"]).rsplit("trial", 1)[1]]
        sample["split"] = split
    manifest_path.write_text(json.dumps(raw_manifest), encoding="utf-8")

    config = {
        "raw_dataset_root": str(raw_root),
        "canonical_root": str(tmp_path / "canonical"),
        "joint_names": DEFAULT_JOINT_NAMES,
        "k_samples_per_segment": 3,
        "num_blocks": 9,
        "eos_token_id": 9,
        "ignore_index": -100,
        "max_sequence_length": 3,
        "split_length_counts": {
            "train": {"2": 1, "3": 1},
            "val": {"2": 1, "3": 1},
            "test": {"2": 1, "3": 1},
        },
    }
    result = build_canonical_dataset(config, overwrite=True)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    assert manifest["split_length_counts"] == config["split_length_counts"]
    assert all(str(seq_id).endswith("trial000") for seq_id in manifest["split"]["train"])
    assert all(str(seq_id).endswith("trial001") for seq_id in manifest["split"]["val"])
    assert all(str(seq_id).endswith("trial002") for seq_id in manifest["split"]["test"])


def test_dataset_and_collate_separate_inputs_from_targets(tmp_path):
    _, manifest_path = _build_tiny_canonical(tmp_path)
    dataset = CorsiMemoryRecallV2Dataset(manifest_path, split="train")
    batch = collate_memory_recall_v2_batch([dataset[0], dataset[1]])
    assert set(batch["model_inputs"]) == {"images", "segment_mask", "frame_mask"}
    for forbidden in ["tokens", "block_id", "rank", "length", "joint", "ee_pose", "ee_xy", "block_xy"]:
        assert forbidden not in batch["model_inputs"]
    assert batch["model_inputs"]["images"].shape == (2, 3, 3, 3, 128, 128)
    assert batch["targets"]["tokens"].shape == (2, 4)
    for batch_index, length in enumerate(batch["metadata"]["length"].tolist()):
        order = batch["metadata"]["block_order"][batch_index]
        expected = torch.tensor(order + [9], dtype=torch.long)
        torch.testing.assert_close(batch["targets"]["tokens"][batch_index, : length + 1], expected)
        assert batch["targets"]["token_mask"][batch_index, : length + 1].all()
        assert batch["model_inputs"]["segment_mask"][batch_index, :length].all()
        if length + 1 < batch["targets"]["tokens"].shape[1]:
            assert batch["targets"]["tokens"][batch_index, length + 1].item() == -100
            assert not batch["targets"]["token_mask"][batch_index, length + 1]


def test_dataset_can_skip_images(tmp_path):
    _, manifest_path = _build_tiny_canonical(tmp_path)
    dataset = CorsiMemoryRecallV2Dataset(manifest_path, split="train", load_images=False)
    sample = dataset[0]
    assert sample["model_inputs"]["images"] is None
    batch = collate_memory_recall_v2_batch([sample])
    assert batch["model_inputs"]["images"] is None
    assert batch["targets"]["joint"].shape[-1] == 7


def test_validators_accept_synthetic_dataset(tmp_path):
    config, manifest_path = _build_tiny_canonical(tmp_path)
    config = {**config, "canonical_manifest_path": str(manifest_path)}
    raw = validate_raw_dataset(config)
    canonical = validate_canonical_dataset(config)
    leakage = validate_leakage(config)
    assert raw["status"] == "ok"
    assert canonical["status"] == "ok"
    assert leakage["status"] == "ok"


def test_overwrite_guard_rejects_parent_of_protected_v1_root(tmp_path):
    config = _tiny_config(tmp_path)
    protected_parent = tmp_path / "protected_parent"
    protected_v1 = protected_parent / "corsi_motion_7joint_k12"
    protected_v1.mkdir(parents=True)
    unsafe = {
        **config,
        "canonical_root": str(protected_parent),
        "protected_v1_canonical_root": str(protected_v1),
    }
    with pytest.raises(ValueError, match="protected V1 root"):
        build_canonical_dataset(unsafe, overwrite=True)


def test_raw_validator_rejects_incomplete_manifest_even_if_manifest_counts_claim_complete(tmp_path):
    config = _tiny_config(tmp_path)
    manifest_path = Path(config["raw_dataset_root"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"] = manifest["samples"][:-1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="computed raw length counts|raw sample count"):
        validate_raw_dataset(config)
