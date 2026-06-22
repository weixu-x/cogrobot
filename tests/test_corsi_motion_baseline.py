import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from corsi.experiments.corsi_motion_baseline.canonicalize import (
    DEFAULT_JOINT_NAMES,
    build_canonical_dataset,
)
from corsi.experiments.corsi_motion_baseline.dataset import (
    CorsiMotionCanonicalDataset,
    collate_motion_prediction_batch,
)
from corsi.experiments.corsi_motion_baseline.model import (
    InstrumentedLSTMCell,
    build_model,
    persistence_prediction,
)
from corsi.experiments.corsi_motion_baseline.train import train_one_run


def _write_synthetic_raw(root: Path) -> None:
    samples = []
    for length in [2, 3]:
        for trial in range(3):
            seq_id = f"len{length:02d}_trial{trial:03d}"
            episode_dir = root / "episodes" / seq_id
            episode_dir.mkdir(parents=True)
            frames_per_segment = 3
            total = length * frames_per_segment
            timestamps = []
            for rank in range(length):
                base = float(rank)
                timestamps.extend([base, base + 0.1, base + 0.4])
            timestamps = np.asarray(timestamps, dtype=np.float64)
            joints = np.stack(
                [timestamps.astype(np.float32) * 10.0 + dim + trial for dim in range(7)],
                axis=1,
            ).astype(np.float32)
            rgb = np.zeros((total, 128, 128, 3), dtype=np.uint8)
            for index in range(total):
                rgb[index, :, :, :] = index
            ranks = np.repeat(np.arange(length, dtype=np.int64), frames_per_segment)
            block_order = list(range(trial, trial + length))
            block_ids = np.repeat(np.asarray(block_order, dtype=np.int64), frames_per_segment)
            arrays_path = episode_dir / "arrays.npz"
            np.savez_compressed(
                arrays_path,
                rgb=rgb,
                joint=joints,
                joint_velocity=np.zeros_like(joints),
                ee_pose=np.zeros((total, 7), dtype=np.float32),
                ee_xy=np.zeros((total, 2), dtype=np.float32),
                action=np.zeros((total, 12), dtype=np.float32),
                qpos=np.zeros((total, 19), dtype=np.float32),
                qvel=np.zeros((total, 19), dtype=np.float32),
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
            block_positions = {
                str(block): {"xy_norm": [float(block) / 10.0, -float(block) / 10.0]}
                for block in block_order
            }
            metadata = {
                "schema_version": "scala_corsi_motion_raw_v1",
                "seq_id": seq_id,
                "length": length,
                "block_order": block_order,
                "layout_id": "fixed_9block_v1",
                "seed": 1000 + length * 10 + trial,
                "frame_count": total,
                "camera_name": "freecam",
                "image_shape": [128, 128, 3],
                "array_keys": ["rgb", "joint", "timestamp", "rank", "block_id"],
                "block_positions": block_positions,
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
                    "status": "generated",
                }
            )
    manifest = {
        "schema_version": "scala_corsi_motion_raw_v1",
        "dataset_root": str(root),
        "num_episodes": len(samples),
        "length_counts": {"2": 3, "3": 3},
        "samples": samples,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _build_tiny_canonical(tmp_path: Path) -> Path:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _write_synthetic_raw(raw_root)
    canonical_root = tmp_path / "canonical"
    config = {
        "raw_dataset_root": str(raw_root),
        "canonical_root": str(canonical_root),
        "joint_names": DEFAULT_JOINT_NAMES,
        "k_samples_per_segment": 3,
        "split_seed": 5,
        "train_per_length": 1,
        "val_per_length": 1,
        "test_per_length": 1,
    }
    result = build_canonical_dataset(config, overwrite=True)
    assert result["split_counts"] == {"train": 2, "val": 2, "test": 2}
    return canonical_root / "manifest.json"


def test_timestamp_based_k_sampling_and_expected_t(tmp_path):
    manifest_path = _build_tiny_canonical(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample = next(item for item in manifest["samples"] if item["seq_id"] == "len02_trial000")
    with np.load(sample["canonical_path"]) as arrays:
        assert arrays["images"].shape == (6, 3, 128, 128)
        assert arrays["joints"].shape == (6, 7)
        assert arrays["source_frame_index"].tolist()[:3] == [0, 1, 2]
        # First segment target times are [0.0, 0.2, 0.4]; joint dim 0 is timestamp * 10.
        np.testing.assert_allclose(arrays["joints"][:3, 0], [0.0, 2.0, 4.0], atol=1e-5)
        assert arrays["transition_mask"].tolist() == [True, True, True, True, True, False]
        assert arrays["within_segment_transition"].tolist() == [True, True, False, True, True, False]
        assert arrays["segment_boundary_transition"].tolist() == [False, False, True, False, False, False]


def test_normalization_uses_train_only(tmp_path):
    manifest_path = _build_tiny_canonical(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_ids = set(manifest["split"]["train"])
    rows = []
    for sample in manifest["samples"]:
        if sample["seq_id"] in train_ids:
            with np.load(sample["canonical_path"]) as arrays:
                rows.append(arrays["joints"])
    train_joints = np.concatenate(rows, axis=0)
    np.testing.assert_allclose(manifest["normalization"]["joint_mean"], train_joints.mean(axis=0), atol=1e-6)


def test_collate_masks_and_shift_alignment(tmp_path):
    manifest_path = _build_tiny_canonical(tmp_path)
    dataset = CorsiMotionCanonicalDataset(manifest_path, split="train")
    batch = collate_motion_prediction_batch([dataset[0], dataset[1]])
    assert batch["images"].shape[0] == 2
    for item_index, length in enumerate(batch["lengths"].tolist()):
        assert batch["valid_mask"][item_index, :length].all()
        assert not batch["loss_mask"][item_index, length - 1]
        torch.testing.assert_close(
            batch["targets_next"][item_index, : length - 1],
            batch["joints"][item_index, 1:length],
        )


def test_joint_only_dataset_can_skip_images(tmp_path):
    manifest_path = _build_tiny_canonical(tmp_path)
    dataset = CorsiMotionCanonicalDataset(manifest_path, split="train", load_images=False)
    sample = dataset[0]
    assert sample["images"] is None
    batch = collate_motion_prediction_batch([sample])
    assert batch["images"] is None
    assert batch["joints"].shape[-1] == 7


def test_resume_restores_checkpoint_rng_states(tmp_path):
    manifest_path = _build_tiny_canonical(tmp_path)
    run_dir = tmp_path / "run"
    config = {
        "canonical_root": str(manifest_path.parent),
        "device": "cpu",
        "batch_size": 2,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "amp": False,
        "min_epochs": 1,
        "early_stopping_patience": 10,
    }
    summary = train_one_run(config, model_type="joint_only", seed=11, run_dir=run_dir, max_epochs_override=1)
    assert summary["epochs_completed"] == 1
    payload = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)

    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    resumed = train_one_run(config, model_type="joint_only", seed=11, run_dir=run_dir, max_epochs_override=1, resume=True)

    assert resumed["epochs_completed"] == 1
    torch.testing.assert_close(torch.get_rng_state(), payload["torch_rng_state"])
    numpy_state = np.random.get_state()
    assert numpy_state[0] == payload["numpy_rng_state"][0]
    np.testing.assert_array_equal(numpy_state[1], payload["numpy_rng_state"][1])
    assert random.getstate() == payload["python_random_state"]


def test_instrumented_cell_matches_torch_lstm_cell():
    torch.manual_seed(3)
    ours = InstrumentedLSTMCell(5, 7)
    reference = torch.nn.LSTMCell(5, 7)
    with torch.no_grad():
        reference.weight_ih.copy_(ours.weight_ih)
        reference.weight_hh.copy_(ours.weight_hh)
        reference.bias_ih.copy_(ours.bias_ih)
        reference.bias_hh.copy_(ours.bias_hh)
    x = torch.randn(4, 5)
    h = torch.randn(4, 7)
    c = torch.randn(4, 7)
    h_ours, c_ours, gates = ours(x, (h, c))
    h_ref, c_ref = reference(x, (h, c))
    torch.testing.assert_close(h_ours, h_ref)
    torch.testing.assert_close(c_ours, c_ref)
    assert gates["input_gate"].min() >= 0
    assert gates["input_gate"].max() <= 1
    assert gates["forget_gate"].shape == (4, 7)


def test_padding_does_not_update_recurrent_state():
    torch.manual_seed(4)
    model = build_model("joint_only", joint_dim=7)
    joints = torch.randn(2, 5, 7)
    valid = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    out = model(images=None, joints=joints, valid_mask=valid, return_traces=True)
    traces = out["traces"]
    torch.testing.assert_close(traces["h_t"][1, 3], traces["h_t"][1, 2])
    torch.testing.assert_close(traces["c_t"][1, 4], traces["c_t"][1, 2])


def test_segment_boundary_does_not_reset_recurrent_state():
    torch.manual_seed(5)
    model = build_model("joint_only", joint_dim=7)
    joints = torch.randn(1, 6, 7)
    valid = torch.ones(1, 6, dtype=torch.bool)
    boundary_out = model(images=None, joints=joints, valid_mask=valid, return_traces=True)
    # The model receives no rank/block/boundary input and keeps recurrence continuous.
    altered = joints.clone()
    altered[:, 3:] += 0.1
    altered_out = model(images=None, joints=altered, valid_mask=valid, return_traces=True)
    torch.testing.assert_close(boundary_out["traces"]["h_t"][:, :3], altered_out["traces"]["h_t"][:, :3])
    with pytest.raises(AssertionError):
        torch.testing.assert_close(boundary_out["traces"]["h_t"][:, 3:], altered_out["traces"]["h_t"][:, 3:])


def test_persistence_and_cpu_gpu_forward():
    images = torch.randn(2, 4, 3, 128, 128)
    joints = torch.randn(2, 4, 7)
    valid = torch.ones(2, 4, dtype=torch.bool)
    model = build_model("visual_joint", joint_dim=7)
    out = model(images=images, joints=joints, valid_mask=valid, return_traces=True)
    assert out["pred_joints_next"].shape == (2, 4, 7)
    assert out["visual_feature"].shape == (2, 4, 128)
    torch.testing.assert_close(persistence_prediction(joints), joints)
    if torch.cuda.is_available():
        model = model.cuda()
        cuda_out = model(images=images.cuda(), joints=joints.cuda(), valid_mask=valid.cuda())
        assert cuda_out["pred_joints_next"].is_cuda
