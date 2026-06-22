import json
from pathlib import Path

import numpy as np
import pytest
import torch

from corsi.experiments.corsi_motion_baseline.posthoc_suite import (
    _autoregressive_prediction,
    _copy_warm_start_inputs,
    _step_accuracy_summary_for_predictions,
    _teacher_forced_prediction,
    FKEvaluator,
    RunRef,
    autoregressive_growth_metrics,
    convergence_stats,
    endpoint_transition_indices,
    exact_resume_possible,
    full_accuracy_summary,
    full_trial_rows_from_token_rows,
    nearest_block_id,
    physical_hit,
    token_rows_from_predictions,
    token_accuracy_summary,
)


def _curves(values, train_start=1.0, train_step=-0.001):
    return [
        {
            "epoch": index + 1,
            "train_loss": train_start + train_step * index,
            "val_normalized_rmse": float(value),
            "val_normalized_mse": float(value) ** 2,
            "test_metric_that_must_be_ignored": 999.0 - index,
        }
        for index, value in enumerate(values)
    ]


def test_convergence_classifies_flat_improving_and_overfitting():
    flat = _curves([1.0 + 0.00001 * ((index % 2) - 0.5) for index in range(60)])
    improving = _curves(np.linspace(1.0, 0.8, 60))
    overfit = _curves(np.linspace(1.0, 1.02, 60), train_start=1.0, train_step=-0.002)
    assert convergence_stats(flat, 50)["classification"] == "CONVERGED"
    assert convergence_stats(improving, 50)["classification"] == "STILL_IMPROVING"
    assert convergence_stats(overfit, 50)["classification"] == "OVERFITTING"


def test_convergence_threshold_edges_are_exact():
    near_converged = _curves([1.0] * 50 + [0.995] * 10)
    improving = _curves([1.0] * 50 + [0.99] * 10)
    assert convergence_stats(near_converged, 50)["classification"] == "UNCERTAIN"
    assert convergence_stats(improving, 50)["classification"] == "STILL_IMPROVING"


def test_convergence_ignores_test_metrics_for_decisions():
    rows = _curves(np.linspace(1.0, 0.9, 60))
    baseline = convergence_stats(rows, 50)
    for row in rows:
        row["test_metric_that_must_be_ignored"] = -1e9
    changed_test = convergence_stats(rows, 50)
    assert baseline["classification"] == changed_test["classification"]
    assert baseline["relative_change"] == changed_test["relative_change"]


def test_exact_resume_requires_scaler_and_rng_states():
    minimal = {
        "exists": True,
        "has_model_state": True,
        "has_optimizer_state": True,
        "has_scaler_state": False,
        "has_scheduler_state": True,
        "has_torch_rng_state": True,
        "has_numpy_rng_state": True,
        "has_python_random_state": True,
    }
    assert not exact_resume_possible(minimal)
    minimal["has_scaler_state"] = True
    minimal["has_scheduler_state"] = False
    assert not exact_resume_possible(minimal)
    minimal["has_scheduler_state"] = True
    assert exact_resume_possible(minimal)


def test_warm_start_metadata_when_exact_resume_is_missing(tmp_path):
    original = tmp_path / "joint_only_seed0_full"
    extension = tmp_path / "joint_only_seed0_extended_continuation_from_best"
    original.mkdir()
    torch.save({"epoch": 7, "model_state_dict": {"x": torch.tensor([1.0])}}, original / "best.pt")
    curves = [{"epoch": epoch, "val_normalized_rmse": 1.0 / epoch, "train_loss": 1.0} for epoch in range(1, 10)]
    (original / "curves.json").write_text(json.dumps(curves), encoding="utf-8")

    metadata = _copy_warm_start_inputs(
        RunRef("joint_only", 0, "joint_only_seed0_full", original),
        extension,
    )

    assert metadata["continuation_type"] == "warm_start_from_best"
    assert metadata["exact_resume"] is False
    assert "lack AMP scaler and RNG states" in metadata["reason_exact_resume_unavailable"]
    assert torch.load(extension / "latest.pt", map_location="cpu", weights_only=False)["epoch"] == 7
    warm_curves = json.loads((extension / "curves.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in warm_curves] == list(range(1, 8))


def _synthetic_prediction():
    joints = np.zeros((6, 7), dtype=np.float32)
    for step in range(6):
        joints[step] = step * 0.01
    pred = joints.copy()
    pred[:-1] = joints[1:]
    pred[1, 0] += np.deg2rad(0.25)
    return {
        "condition": "synthetic",
        "model_type": "joint_only",
        "mode": "normal",
        "seed": 0,
        "seq_id": "s0",
        "length": 2,
        "block_order": [0, 1],
        "joints": joints.copy(),
        "physical_joints": joints.copy(),
        "joints_norm": joints,
        "joints_phys": joints,
        "pred_norm": pred,
        "pred_phys": pred,
        "transition_mask": np.array([True, True, True, True, True, False]),
        "loss_mask": np.array([True, True, True, True, True, False]),
        "within_segment_transition": np.array([True, True, False, True, True, False]),
        "segment_boundary_transition": np.array([False, False, True, False, False, False]),
        "rank": np.array([0, 0, 0, 1, 1, 1]),
        "block_id": np.array([0, 0, 0, 1, 1, 1]),
        "source_frame_index": np.arange(6),
        "segment_progress": np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0]),
    }


def test_step_tolerance_accuracy_and_threshold_edge():
    summary = _step_accuracy_summary_for_predictions(
        [_synthetic_prediction()],
        thresholds=[0.1, 0.25],
        condition_meta={"condition": "synthetic", "model_type": "joint_only", "mode": "normal", "seed": 0},
    )
    assert summary["thresholds"]["0.1"]["all_joint_step_accuracy"] == 4 / 5
    assert summary["thresholds"]["0.25"]["all_joint_step_accuracy"] == 1.0
    assert summary["thresholds"]["0.1"]["boundary_transition_step_accuracy"] == 1.0


def test_segment_endpoint_indexing_excludes_boundary_prediction():
    prediction = _synthetic_prediction()
    assert endpoint_transition_indices(prediction, 3) == [(0, 1), (1, 4)]
    prediction["within_segment_transition"][1] = False
    try:
        endpoint_transition_indices(prediction, 3)
    except ValueError as exc:
        assert "not within-segment" in str(exc)
    else:
        raise AssertionError("expected endpoint boundary guard to fail")


def test_nearest_block_and_physical_hit_logic():
    geometry = {
        "0": {"center_world": [0.0, 0.0, 0.8], "footprint_half_size_xy": [0.02, 0.02]},
        "1": {"center_world": [0.1, 0.0, 0.8], "footprint_half_size_xy": [0.02, 0.02]},
    }
    assert nearest_block_id(np.array([0.08, 0.0]), geometry) == 1
    assert physical_hit(np.array([0.024, 0.0, 0.84]), 0, 0.84, geometry, margin_m=0.005)
    assert not physical_hit(np.array([0.026, 0.0, 0.84]), 0, 0.84, geometry, margin_m=0.005)
    assert not physical_hit(np.array([0.0, 0.0, 0.87]), 0, 0.84, geometry, margin_m=0.005)


class _FakeFK:
    block_geometry = {
        "0": {"center_world": [0.0, 0.0, 0.8], "footprint_half_size_xy": [0.02, 0.02]},
        "1": {"center_world": [0.1, 0.0, 0.8], "footprint_half_size_xy": [0.02, 0.02]},
    }

    def joint_limits(self):
        return np.asarray([[-1.0, 1.0]] * 7, dtype=np.float64)

    def fk(self, q_arm):
        q_arm = np.asarray(q_arm, dtype=np.float64)
        return np.asarray([q_arm[0], 0.0, 0.84], dtype=np.float64)


def test_joint_limit_violation_fails_autoregressive_physical_hit(tmp_path):
    raw_path = tmp_path / "arrays.npz"
    np.savez(raw_path, ee_pose=np.tile(np.asarray([[0.0, 0.0, 0.84]], dtype=np.float64), (6, 1)))
    prediction = _synthetic_prediction()
    prediction["pred_phys"][1] = 0.0
    prediction["pred_phys"][1, 0] = 2.0
    rows = token_rows_from_predictions(
        [prediction],
        condition_meta={"condition": "c", "model_type": "joint_only", "mode": "normal", "seed": 0},
        manifest_samples={"s0": {"raw_arrays_path": str(raw_path)}},
        fk=_FakeFK(),
        k=3,
        autoregressive=True,
    )
    first = rows[0]
    assert first["joint_limit_violation"]
    assert not first["physical_hit_margin_5mm"]


def test_full_trial_all_token_aggregation_by_length():
    token_rows = [
        {
            "condition": "c",
            "model_type": "joint_only",
            "mode": "normal",
            "seed": 0,
            "seq_id": "a",
            "length": 2,
            "rank": 0,
            "nearest_block_correct": True,
            "physical_hit_margin_0mm": True,
            "physical_hit_margin_5mm": True,
            "physical_hit_margin_10mm": True,
            "endpoint_ee_error_cm": 0.1,
            "endpoint_xy_error_cm": 0.1,
            "endpoint_z_error_cm": 0.0,
            "mode_type": "teacher_forced",
        },
        {
            "condition": "c",
            "model_type": "joint_only",
            "mode": "normal",
            "seed": 0,
            "seq_id": "a",
            "length": 2,
            "rank": 1,
            "nearest_block_correct": False,
            "physical_hit_margin_0mm": True,
            "physical_hit_margin_5mm": True,
            "physical_hit_margin_10mm": True,
            "endpoint_ee_error_cm": 0.2,
            "endpoint_xy_error_cm": 0.2,
            "endpoint_z_error_cm": 0.0,
            "mode_type": "teacher_forced",
        },
    ]
    full_rows = full_trial_rows_from_token_rows(token_rows, prefix="tf")
    assert not full_rows[0]["tf_full_nearest_block_correct"]
    assert full_rows[0]["tf_full_physical_hit_correct"]
    summary = full_accuracy_summary(full_rows, prefix="tf")
    assert summary["per_condition"][0]["by_length"]["2"]["tf_full_physical_hit_acc"] == 1.0
    token_summary = token_accuracy_summary(token_rows, prefix="tf")
    assert token_summary["per_condition"][0]["tf_token_nearest_block_acc"] == 0.5


class _SpyModel:
    def __init__(self):
        self.image_sum = None

    def __call__(self, *, images, joints, valid_mask):
        self.image_sum = float(images.abs().sum().item())
        return {"pred_joints_next": joints.clone()}


def test_zero_vision_is_applied_to_post_normalization_tensor():
    item = _synthetic_prediction()
    item["images"] = np.ones((6, 3, 128, 128), dtype=np.float32)
    manifest = {"normalization": {"joint_mean": [0] * 7, "joint_std": [1] * 7}, "joint_dim": 7}
    model = _SpyModel()
    _teacher_forced_prediction(
        item=item,
        model=model,
        model_type="visual_joint",
        mode="zero_vision",
        manifest=manifest,
        device=torch.device("cpu"),
    )
    assert model.image_sum == 0.0


def test_autoregressive_feeds_predicted_joint_feedback():
    torch.manual_seed(0)
    from corsi.experiments.corsi_motion_baseline.model import build_model

    model = build_model("joint_only", joint_dim=7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.output.bias.fill_(0.123)
    item = _synthetic_prediction()
    item["joints"][1:] = 10.0
    manifest = {"normalization": {"joint_mean": [0] * 7, "joint_std": [1] * 7}, "joint_dim": 7}
    pred = _autoregressive_prediction(
        item=item,
        model=model,
        model_type="joint_only",
        mode="normal",
        manifest=manifest,
        device=torch.device("cpu"),
    )
    np.testing.assert_allclose(pred["pred_norm"][0], np.full((7,), 0.123, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(pred["pred_norm"][1], np.full((7,), 0.123, dtype=np.float32), atol=1e-6)


def test_step_accuracy_summary_is_deterministic():
    prediction = _synthetic_prediction()
    meta = {"condition": "synthetic", "model_type": "joint_only", "mode": "normal", "seed": 3}
    first = _step_accuracy_summary_for_predictions([prediction], thresholds=[0.1, 1.0], condition_meta=meta)
    second = _step_accuracy_summary_for_predictions([prediction], thresholds=[0.1, 1.0], condition_meta=meta)
    assert first == second


def test_ground_truth_fk_reconstruction_gate_on_local_artifact():
    manifest_path = Path("corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12/manifest.json")
    if not manifest_path.exists():
        pytest.skip("local canonical Corsi motion artifact is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fk = FKEvaluator(manifest, split="test")
    try:
        validation = fk.validate()
    finally:
        fk.close()
    assert validation["site_name"] == "gripper0_right_index_tip_site"
    assert validation["median_position_error_mm"] <= 2.0
    assert validation["max_position_error_mm"] <= 5.0
