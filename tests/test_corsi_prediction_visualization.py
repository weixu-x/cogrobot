from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
import torch

from corsi.experiments.corsi_motion_baseline.canonicalize import load_config
from corsi.experiments.corsi_motion_baseline.visualize_predictions import (
    StaticPoseRenderer,
    attach_fk_ee,
    episode_metrics,
    generate_comparison_video,
    load_episode,
    predict_sequence,
    resolve_checkpoints,
    resolve_joint_limit_metadata,
    select_episodes,
    select_representative_seed,
    validate_fk_for_predictions,
    write_prediction_export,
    write_video_from_frames,
)


CONFIG_PATH = Path("corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json")
SEQ_ID = "len02_trial002"


def _context():
    if not CONFIG_PATH.exists():
        pytest.skip("Corsi motion baseline config is not available")
    config = load_config(CONFIG_PATH)
    manifest_path = Path(str(config["canonical_root"])) / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("canonical motion manifest is not available")
    manifest = json.loads(manifest_path.read_text())
    checkpoints = resolve_checkpoints(config)
    episode = load_episode(manifest, SEQ_ID)
    return config, manifest, checkpoints, episode


def test_checkpoint_selection_uses_validation_metadata_and_median_seed():
    config, _manifest, checkpoints, _episode = _context()
    assert config["output_root"].endswith("corsi_motion_7joint_k12")
    assert select_representative_seed(checkpoints) == 15
    visual = [checkpoints[("visual_joint", seed)] for seed in [0, 5, 10, 15, 20]]
    assert all(ref.checkpoint_path.name == "best.pt" for ref in visual)
    assert {ref.source for ref in visual} <= {"original", "extension"}
    # The median is computed from validation RMSE only, not test metrics.
    assert sorted((ref.best_val_rmse, ref.seed) for ref in visual)[2][1] == 15


def test_teacher_forced_alignment_and_final_timestep_exclusion():
    _config, manifest, checkpoints, episode = _context()
    prediction = predict_sequence(
        model_family="joint_only",
        mode="teacher_forced",
        checkpoint=checkpoints[("joint_only", 15)],
        episode=episode,
        manifest=manifest,
        device=torch.device("cpu"),
    )
    joints = episode["canonical"]["joints"]
    assert len(prediction["timestep"]) == joints.shape[0] - 1
    np.testing.assert_allclose(prediction["q_input"][0], joints[0], atol=1e-6)
    np.testing.assert_allclose(prediction["q_target"][0], joints[1], atol=1e-6)
    assert int(prediction["timestep"][-1]) == joints.shape[0] - 2


def test_open_loop_feedback_uses_previous_prediction_after_first_step():
    _config, manifest, checkpoints, episode = _context()
    prediction = predict_sequence(
        model_family="joint_only",
        mode="open_loop_joint_feedback_with_exogenous_images",
        checkpoint=checkpoints[("joint_only", 15)],
        episode=episode,
        manifest=manifest,
        device=torch.device("cpu"),
    )
    np.testing.assert_allclose(prediction["q_input"][0], episode["canonical"]["joints"][0], atol=1e-6)
    np.testing.assert_allclose(prediction["q_input"][1:], prediction["q_prediction"][:-1], atol=1e-5)


def test_segment_boundary_does_not_force_ground_truth_feedback():
    _config, manifest, checkpoints, episode = _context()
    prediction = predict_sequence(
        model_family="joint_only",
        mode="open_loop_joint_feedback_with_exogenous_images",
        checkpoint=checkpoints[("joint_only", 15)],
        episode=episode,
        manifest=manifest,
        device=torch.device("cpu"),
    )
    boundary_indices = np.where(prediction["segment_boundary"])[0]
    assert boundary_indices.size > 0
    for index in boundary_indices:
        if index > 0:
            np.testing.assert_allclose(prediction["q_input"][index], prediction["q_prediction"][index - 1], atol=1e-5)


def test_episode_selection_is_deterministic_and_includes_required_categories():
    visual = {}
    joint = {}
    for length in range(2, 10):
        for trial in range(3):
            seq_id = f"len{length:02d}_trial{trial:03d}"
            visual[seq_id] = {
                "seq_id": seq_id,
                "length": length,
                "normalized_rmse": float(length + trial) / 100.0,
                "mean_joint_mae_deg": float(length + trial),
                "boundary_mean_joint_mae_deg": float(length * trial),
                "endpoint_mean_joint_mae_deg": float(length + 2 * trial),
            }
            joint[seq_id] = {
                "seq_id": seq_id,
                "length": length,
                "mean_joint_mae_deg": float(length + trial + (1 if trial == 0 else -1)),
            }
    first = select_episodes(visual_metrics=visual, joint_metrics=joint)
    second = select_episodes(visual_metrics=visual, joint_metrics=joint)
    assert first == second
    categories = {row["category"] for row in first}
    assert "global_lowest_rmse" in categories
    assert "global_highest_rmse" in categories
    assert "largest_segment_boundary_error" in categories
    assert "largest_endpoint_error" in categories
    assert "visual_best_gain_over_joint_only" in categories
    assert "visual_worst_gap_vs_joint_only" in categories
    assert all(f"median_rmse_length_{length}" in categories for length in range(2, 10))


def test_static_pose_renderer_qpos_mapping_fixed_hand_and_fk_validation():
    _config, manifest, checkpoints, episode = _context()
    prediction = predict_sequence(
        model_family="visual_joint",
        mode="teacher_forced",
        checkpoint=checkpoints[("visual_joint", 15)],
        episode=episode,
        manifest=manifest,
        device=torch.device("cpu"),
    )
    renderer = StaticPoseRenderer(episode, offscreen=False)
    try:
        assert renderer.arm_qpos_indexes.tolist() == list(range(7))
        assert renderer.gripper_qpos_indexes.tolist() == list(range(7, 19))
        mapping = renderer.mapping_metadata()
        assert np.asarray(mapping["joint_limits"]).shape == (7, 2)
        assert mapping["source"] == "robosuite PandaDexRH MuJoCo model jnt_range"
        fixed = renderer.fixed_hand_qpos.copy()
        renderer.set_arm(prediction["q_target"][0])
        np.testing.assert_allclose(renderer.env.sim.data.qpos[renderer.gripper_qpos_indexes], fixed, atol=1e-7)
        renderer.set_arm(prediction["q_target"][-1])
        np.testing.assert_allclose(renderer.env.sim.data.qpos[renderer.gripper_qpos_indexes], fixed, atol=1e-7)
    finally:
        renderer.close()
    joint_metadata = resolve_joint_limit_metadata(episode, manifest)
    assert np.asarray(joint_metadata["joint_limits"]).shape == (7, 2)
    validation = validate_fk_for_predictions([episode], [prediction])
    assert validation["tier_b_available"]
    assert validation["median_error_m"] <= 0.002
    assert validation["max_error_m"] <= 0.005


def test_prediction_export_contains_fk_ee_fields(tmp_path):
    _config, manifest, checkpoints, episode = _context()
    prediction = predict_sequence(
        model_family="visual_joint",
        mode="teacher_forced",
        checkpoint=checkpoints[("visual_joint", 15)],
        episode=episode,
        manifest=manifest,
        device=torch.device("cpu"),
    )
    prediction = attach_fk_ee(prediction, episode)
    result = write_prediction_export(tmp_path, prediction, manifest, extra_metadata={"tier_b_available": True})
    assert result["status"] == "written"
    assert result["schema_version"] == "corsi_prediction_export_v3"
    npz = np.load(result["paths"]["npz"])
    assert npz["predicted_ee_pose"].shape[0] == len(prediction["timestep"])
    assert np.isfinite(npz["predicted_ee_pose"]).all()
    assert np.isfinite(npz["ee_position_error_cm"]).all()
    assert npz["target_ee_xy_norm"].shape == prediction["target_ee_xy_norm"].shape
    assert np.isfinite(npz["target_ee_xy_norm"]).all()


def _synthetic_prediction(model_family: str) -> dict:
    steps = 2
    q_target = np.zeros((steps, 7), dtype=np.float32)
    q_prediction = q_target + (0.01 if model_family == "visual_joint" else 0.02)
    return {
        "seq_id": "synthetic_len02",
        "model_family": model_family,
        "seed": 15 if model_family != "persistence" else -1,
        "prediction_mode": "teacher_forced",
        "images_uint8": np.zeros((steps, 3, 32, 32), dtype=np.uint8),
        "timestep": np.arange(steps, dtype=np.int64),
        "target_ee_pose": np.zeros((steps, 7), dtype=np.float32),
        "block_xy": np.zeros((steps, 2), dtype=np.float32),
        "q_target": q_target,
        "q_prediction": q_prediction,
        "mean_joint_mae_deg": np.full((steps,), 0.25, dtype=np.float32),
        "segment_boundary": np.asarray([False, True]),
        "length": np.full((steps,), 2, dtype=np.int64),
        "rank": np.zeros((steps,), dtype=np.int64),
        "block_id": np.zeros((steps,), dtype=np.int64),
        "joint_limits": np.asarray(
            [
                [-2.0, 2.0],
                [-2.0, 2.0],
                [-2.0, 2.0],
                [-3.0, 0.0],
                [-2.0, 2.0],
                [0.0, 3.0],
                [-2.0, 2.0],
            ],
            dtype=np.float32,
        ),
    }


def test_comparison_video_outputs_decodable_mp4_and_cleans_frames(tmp_path):
    result = generate_comparison_video(
        root=tmp_path,
        predictions_by_model={
            "visual_joint": _synthetic_prediction("visual_joint"),
            "joint_only": _synthetic_prediction("joint_only"),
            "persistence": _synthetic_prediction("persistence"),
        },
        mode="teacher_forced",
        seq_id="synthetic_len02",
        resume=False,
        fps=12,
        keep_frames=False,
    )
    assert result["status"] == "written"
    assert Path(result["path"]).exists()
    assert not Path(result["frames_dir"]).exists()
    reader = imageio.get_reader(result["path"])
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    assert frame.shape[:2] == (1080, 1920)


def test_video_writer_outputs_decodable_mp4_with_expected_frame_count(tmp_path):
    frames = []
    for index in range(3):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :, index % 3] = 50 + index
        path = tmp_path / f"frame_{index:05d}.png"
        imageio.imwrite(path, frame)
        frames.append(path)
    video_path = tmp_path / "smoke.mp4"
    write_video_from_frames(frames, video_path, fps=12)
    assert video_path.exists()
    reader = imageio.get_reader(video_path)
    try:
        decoded = [reader.get_data(index) for index in range(3)]
    finally:
        reader.close()
    assert len(decoded) == 3
    assert decoded[0].shape[:2] == (1080, 1920)
