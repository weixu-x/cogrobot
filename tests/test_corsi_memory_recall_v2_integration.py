import torch

from corsi.experiments.corsi_memory_recall_v2.analysis import run_causal_sanity_checks
from corsi.experiments.corsi_memory_recall_v2.train import (
    _build_model,
    _call_model,
    compute_stage1_loss,
    compute_stage2_loss,
    evaluate_loader,
)


def _model_config():
    return {
        "raw_dataset_root": "ignored_by_model_builder",
        "joint_names": [f"joint_{index}" for index in range(7)],
        "k_samples_per_segment": 2,
        "max_sequence_length": 3,
        "num_blocks": 9,
        "eos_token_id": 9,
        "ignore_index": -100,
        "cnn_out": 8,
        "visual_hidden": 8,
        "motor_hidden": 8,
        "item_dim": 8,
        "D_mem": 4,
        "recall_hidden": 8,
        "recall_token_dim": 4,
        "memory_noise_std": 0.0,
    }


def _manifest():
    return {
        "schema_version": "scala_corsi_memory_recall_v2_canonical_v1",
        "k_samples_per_segment": 2,
        "max_sequence_length": 3,
        "num_blocks": 9,
        "eos_token_id": 9,
        "ignore_index": -100,
        "canonical_fingerprint": "synthetic",
        "samples": [{"seq_id": "a"}, {"seq_id": "b"}],
        "split": {"train": ["a"], "val": ["b"], "test": []},
    }


def _batch():
    torch.manual_seed(23)
    batch_size = 2
    segments = 3
    frames = 2
    images = torch.randn(batch_size, segments, frames, 3, 128, 128)
    segment_mask = torch.tensor([[True, True, True], [True, True, False]])
    frame_mask = segment_mask.unsqueeze(-1).expand(batch_size, segments, frames).clone()
    tokens = torch.tensor(
        [
            [0, 1, 2, 9],
            [3, 4, 9, -100],
        ],
        dtype=torch.long,
    )
    token_mask = tokens.ne(-100)
    target_xy = torch.zeros(batch_size, segments, 2)
    target_xy[0] = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    target_xy[1, :2] = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])
    joint = torch.randn(batch_size, segments, frames, 7)
    ee_pose = torch.randn(batch_size, segments, frames, 7)
    ee_xy = torch.randn(batch_size, segments, frames, 2)
    return {
        "model_inputs": {
            "images": images,
            "segment_mask": segment_mask,
            "frame_mask": frame_mask,
        },
        "targets": {
            "tokens": tokens,
            "token_mask": token_mask,
            "target_xy": target_xy,
            "block_xy": target_xy,
            "joint": joint,
            "ee_pose": ee_pose,
            "ee_xy": ee_xy,
        },
        "metadata": {
            "length": torch.tensor([3, 2], dtype=torch.long),
            "block_order": [[0, 1, 2], [3, 4]],
            "ignore_index": torch.tensor([-100, -100], dtype=torch.long),
            "eos_token_id": torch.tensor([9, 9], dtype=torch.long),
        },
        "ignore_index": -100,
    }


def test_lane_b_c_d_contract_runs_forward_losses_and_evaluation():
    model = _build_model(_model_config(), stage=2, manifest=_manifest())
    assert model.config.k_samples_per_segment == 2
    assert model.config.max_sequence_length == 3
    assert model.config.memory_dim == 4
    assert model.config.cnn_dim == 8

    batch = _batch()
    outputs = _call_model(model, batch, stage=2)
    assert outputs["logits"].shape == (*batch["targets"]["tokens"].shape, 10)
    assert outputs["pred_joint"].shape == batch["targets"]["joint"].shape

    stage1_loss = compute_stage1_loss(outputs, batch)
    stage2_loss = compute_stage2_loss(outputs, batch)
    assert torch.isfinite(stage1_loss)
    assert torch.isfinite(stage2_loss)
    (stage1_loss + stage2_loss).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    metrics = evaluate_loader(model, [batch], device=torch.device("cpu"), stage=2)
    assert metrics["sequence_count"] == 2
    assert "full_sequence_accuracy" in metrics

    with torch.no_grad():
        checks = run_causal_sanity_checks(model, batch, eos_token_id=9, ignore_index=-100)
    assert set(checks) == {"normal", "memory_zero", "memory_shuffle", "presentation_order_shuffle"}
    assert checks["memory_zero"]["operation"] == "memory_zero"
    assert checks["memory_shuffle"]["operation"] == "memory_shuffle"
    assert isinstance(checks["presentation_order_shuffle"]["decoded_order_changed"], bool)
