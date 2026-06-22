"""Prediction export and visualization for the Corsi 7-joint motion baseline.

This module is intentionally post-hoc: it reads existing canonical data and
checkpoints, exports frame-aligned predictions, and generates visual summaries.
It does not train, mutate checkpoints, or change evaluation metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from corsi.experiments.corsi_motion_baseline.canonicalize import load_config
from corsi.experiments.corsi_motion_baseline.model import build_model
from corsi.experiments.corsi_motion_baseline.train import resolve_device

if "MPLCONFIGDIR" not in os.environ:
    _MPLCONFIGDIR = Path("/tmp/corsi_prediction_mplconfig")
    _MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_MPLCONFIGDIR)

SEEDS = [0, 5, 10, 15, 20]
MODEL_FAMILIES = ["visual_joint", "joint_only", "persistence"]
PREDICTION_MODES = ["teacher_forced", "open_loop_joint_feedback_with_exogenous_images"]
SHORT_MODE = {
    "teacher_forced": "teacher_forced",
    "open_loop_joint_feedback_with_exogenous_images": "autoregressive",
}
INTERPRETATION_STATEMENT = (
    "These visualizations show predictions from a causal sensorimotor motion model. "
    "Teacher-forced videos provide ground-truth current joint angles at each step. "
    "Autoregressive videos feed predicted joints back into the model but continue to "
    "use prerecorded exogenous images. Neither visualization is a closed-loop robot "
    "execution or evidence of Corsi working-memory recall."
)
DEFAULT_PANDA_ARM_JOINT_LIMITS = np.asarray(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=np.float32,
)
DEFAULT_PANDA_ARM_JOINT_LIMIT_SOURCE = "fallback Panda arm joint limits from robosuite Franka defaults"


@dataclass(frozen=True)
class CheckpointRef:
    model_family: str
    seed: int
    run_name: str
    checkpoint_path: Path
    checkpoint_hash: str
    best_val_rmse: float
    source: str


@dataclass(frozen=True)
class PredictionKey:
    model_family: str
    seed: int
    mode: str
    seq_id: str


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ensure_output_tree(root: Path) -> None:
    for rel in [
        "audit",
        "predictions",
        "videos/teacher_forced",
        "videos/autoregressive",
        "videos/comparisons",
        "videos/overlays",
        "montages/whole_episode",
        "montages/endpoints",
        "montages/segment_progress",
        "montages/trajectories",
        "frames",
        "manifests",
        "reports",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def fallback_joint_limit_metadata(joint_names: Sequence[str]) -> dict[str, Any]:
    return {
        "source": DEFAULT_PANDA_ARM_JOINT_LIMIT_SOURCE,
        "joint_names": list(joint_names),
        "joint_limits": DEFAULT_PANDA_ARM_JOINT_LIMITS.astype(float).tolist(),
    }


def joint_limits_from_metadata(metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    limits = np.asarray(metadata["joint_limits"], dtype=np.float32)
    if limits.shape != (7, 2) or not np.isfinite(limits).all():
        limits = DEFAULT_PANDA_ARM_JOINT_LIMITS.copy()
    return limits[:, 0].copy(), limits[:, 1].copy()


def attach_joint_limit_metadata(prediction: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    limits = np.asarray(metadata["joint_limits"], dtype=np.float32)
    if limits.shape != (7, 2):
        limits = DEFAULT_PANDA_ARM_JOINT_LIMITS.copy()
    prediction["joint_limits"] = limits.astype(np.float32)
    prediction["joint_limit_source"] = str(metadata.get("source", DEFAULT_PANDA_ARM_JOINT_LIMIT_SOURCE))
    return prediction


def _posthoc_selection_path() -> Path:
    return Path(
        "corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/convergence/final_checkpoint_selection.json"
    )


def resolve_checkpoints(config: dict[str, Any]) -> dict[tuple[str, int], CheckpointRef]:
    """Resolve final checkpoints strictly from validation-selected metadata."""

    refs: dict[tuple[str, int], CheckpointRef] = {}
    selection_path = _posthoc_selection_path()
    if selection_path.exists():
        selection = read_json(selection_path)
        for row in selection.get("selections", []):
            model_type = str(row["model_type"])
            seed = int(row["seed"])
            checkpoint = Path(str(row["final_checkpoint"]))
            if not checkpoint.exists():
                raise FileNotFoundError(f"selected checkpoint does not exist: {checkpoint}")
            refs[(model_type, seed)] = CheckpointRef(
                model_family=model_type,
                seed=seed,
                run_name=str(row["final_run_name"]),
                checkpoint_path=checkpoint,
                checkpoint_hash=sha256_file(checkpoint),
                best_val_rmse=float(row["final_best_val_rmse"]),
                source=str(row.get("final_source", "posthoc_selection")),
            )
    else:
        run_root = Path(str(config["output_root"]))
        for model_type in ["visual_joint", "joint_only"]:
            for seed in SEEDS:
                run_name = f"{model_type}_seed{seed}_full"
                summary_path = run_root / run_name / "summary.json"
                if not summary_path.exists():
                    continue
                summary = read_json(summary_path)
                checkpoint = Path(str(summary["best_checkpoint"]))
                refs[(model_type, seed)] = CheckpointRef(
                    model_family=model_type,
                    seed=seed,
                    run_name=run_name,
                    checkpoint_path=checkpoint,
                    checkpoint_hash=sha256_file(checkpoint),
                    best_val_rmse=float(summary["best_val_normalized_rmse"]),
                    source="summary_best",
                )
    return refs


def select_representative_seed(checkpoints: dict[tuple[str, int], CheckpointRef]) -> int:
    visual = [checkpoints[("visual_joint", seed)] for seed in SEEDS if ("visual_joint", seed) in checkpoints]
    if not visual:
        raise ValueError("no visual_joint checkpoints available")
    ranked = sorted(visual, key=lambda ref: (ref.best_val_rmse, ref.seed))
    return int(ranked[len(ranked) // 2].seed)


def sample_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(sample["seq_id"]): dict(sample) for sample in manifest["samples"]}


def split_of_seq(manifest: dict[str, Any], seq_id: str) -> str:
    for split, ids in manifest["split"].items():
        if seq_id in set(ids):
            return str(split)
    return ""


def canonical_path(manifest: dict[str, Any], sample: dict[str, Any]) -> Path:
    path = Path(str(sample["canonical_path"]))
    if not path.is_absolute() and not path.exists():
        path = Path(str(manifest["canonical_root"])) / "episodes" / path.name
    return path


def load_episode(manifest: dict[str, Any], seq_id: str) -> dict[str, Any]:
    samples = sample_by_id(manifest)
    if seq_id not in samples:
        raise KeyError(f"unknown seq_id: {seq_id}")
    sample = samples[seq_id]
    with np.load(canonical_path(manifest, sample)) as arrays:
        canonical = {key: arrays[key].copy() for key in arrays.files}
    raw_arrays_path = Path(str(sample["raw_arrays_path"]))
    raw_metadata_path = Path(str(sample["raw_metadata_path"]))
    raw_segments_path = Path(str(sample["raw_segments_path"]))
    with np.load(raw_arrays_path) as arrays:
        raw = {key: arrays[key].copy() for key in arrays.files}
    metadata = read_json(raw_metadata_path)
    segments = read_json(raw_segments_path)
    return {
        "sample": sample,
        "split": split_of_seq(manifest, str(sample["seq_id"])),
        "canonical": canonical,
        "raw": raw,
        "metadata": metadata,
        "segments": segments,
    }


def normalize_images(images_uint8: np.ndarray, manifest: dict[str, Any]) -> np.ndarray:
    stats = manifest["normalization"]
    mean = np.asarray(stats["image_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(stats["image_std"], dtype=np.float32).reshape(1, 3, 1, 1)
    return (images_uint8.astype(np.float32) / 255.0 - mean) / std


def normalize_joints(joints: np.ndarray, manifest: dict[str, Any]) -> np.ndarray:
    stats = manifest["normalization"]
    mean = np.asarray(stats["joint_mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(stats["joint_std"], dtype=np.float32).reshape(1, -1)
    return (joints.astype(np.float32) - mean) / std


def denormalize_joints(joints: np.ndarray, manifest: dict[str, Any]) -> np.ndarray:
    stats = manifest["normalization"]
    mean = np.asarray(stats["joint_mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(stats["joint_std"], dtype=np.float32).reshape(1, -1)
    return joints.astype(np.float32) * std + mean


def interpolate_columns(timestamps: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    values = np.asarray(values)
    if values.ndim == 1:
        return np.interp(query, timestamps, values).astype(np.float32)
    columns = [np.interp(query, timestamps, values[:, col]) for col in range(values.shape[1])]
    return np.stack(columns, axis=1).astype(np.float32)


def normalize_table_xy_rows(table_xy: np.ndarray, bounds: dict[str, Any]) -> np.ndarray:
    table_xy = np.asarray(table_xy, dtype=np.float32)
    x_min = float(bounds["x_min"])
    x_max = float(bounds["x_max"])
    y_min = float(bounds["y_min"])
    y_max = float(bounds["y_max"])
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"invalid XY normalization bounds: {bounds}")
    out = np.empty_like(table_xy, dtype=np.float32)
    out[:, 0] = 2.0 * (table_xy[:, 0] - x_min) / (x_max - x_min) - 1.0
    out[:, 1] = 2.0 * (table_xy[:, 1] - y_min) / (y_max - y_min) - 1.0
    return out


def segment_id_for_rank(segments: Sequence[dict[str, Any]], rank: np.ndarray) -> np.ndarray:
    mapping = {int(row["rank"]): int(row["segment_id"]) for row in segments}
    return np.asarray([mapping.get(int(value), int(value)) for value in rank], dtype=np.int64)


def load_model_for_ref(ref: CheckpointRef, manifest: dict[str, Any], device: torch.device):
    payload = torch.load(ref.checkpoint_path, map_location=device, weights_only=False)
    model = build_model(ref.model_family, joint_dim=int(manifest["joint_dim"]), hidden_dim=128).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_sequence(
    *,
    model_family: str,
    mode: str,
    checkpoint: CheckpointRef | None,
    episode: dict[str, Any],
    manifest: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if mode not in PREDICTION_MODES:
        raise ValueError(f"unknown prediction mode: {mode}")
    canonical = episode["canonical"]
    images_uint8 = canonical["images"]
    joints_physical = canonical["joints"].astype(np.float32)
    transition_mask = canonical["transition_mask"].astype(bool)
    steps = int(joints_physical.shape[0])
    if steps < 2:
        raise ValueError("episode must have at least two canonical steps")
    images_norm = normalize_images(images_uint8, manifest)
    joints_norm = normalize_joints(joints_physical, manifest)
    valid_steps = steps - 1

    if model_family == "persistence":
        model = None
    else:
        if checkpoint is None:
            raise ValueError(f"checkpoint required for {model_family}")
        model = load_model_for_ref(checkpoint, manifest, device)

    pred_norm_rows: list[np.ndarray] = []
    q_input_norm_rows: list[np.ndarray] = []
    traces: dict[str, list[np.ndarray]] = {
        "h_t": [],
        "c_t": [],
        "input_gate": [],
        "forget_gate": [],
        "candidate": [],
        "output_gate": [],
    }

    if mode == "teacher_forced":
        if model is None:
            pred_norm = joints_norm.copy()
            for step in range(valid_steps):
                pred_norm_rows.append(pred_norm[step].copy())
                q_input_norm_rows.append(joints_norm[step].copy())
        else:
            images_tensor = torch.as_tensor(images_norm[None], dtype=torch.float32, device=device)
            joints_tensor = torch.as_tensor(joints_norm[None], dtype=torch.float32, device=device)
            valid_mask = torch.ones((1, steps), dtype=torch.bool, device=device)
            outputs = model(
                images=images_tensor if model_family == "visual_joint" else None,
                joints=joints_tensor,
                valid_mask=valid_mask,
                return_traces=True,
            )
            pred_all = outputs["pred_joints_next"][0].detach().cpu().numpy()  # type: ignore[index]
            trace_all = outputs["traces"]  # type: ignore[index]
            for step in range(valid_steps):
                pred_norm_rows.append(pred_all[step].astype(np.float32).copy())
                q_input_norm_rows.append(joints_norm[step].copy())
                for key in traces:
                    traces[key].append(trace_all[key][0, step].detach().cpu().numpy().astype(np.float32))
    else:
        if model is None:
            q_in = joints_norm[0].copy()
            for step in range(valid_steps):
                pred = q_in.copy()
                pred_norm_rows.append(pred)
                q_input_norm_rows.append(q_in.copy())
                q_in = pred.copy()
        else:
            hidden_dim = int(model.config.hidden_dim)
            h = torch.zeros((1, hidden_dim), dtype=torch.float32, device=device)
            c = torch.zeros_like(h)
            q_in = joints_norm[0].copy()
            for step in range(valid_steps):
                q_tensor = torch.as_tensor(q_in.reshape(1, -1), dtype=torch.float32, device=device)
                joint_feature = model.joint_encoder(q_tensor)
                if model_family == "visual_joint":
                    image_tensor = torch.as_tensor(images_norm[step : step + 1], dtype=torch.float32, device=device)
                    visual_feature = model.visual_encoder(image_tensor)  # type: ignore[operator]
                    fused_input = torch.cat([visual_feature, joint_feature], dim=-1)
                else:
                    fused_input = joint_feature
                fused = model.fusion(fused_input)
                h, c, gates = model.recurrent(fused, (h, c))
                pred_tensor = model.output(h)
                pred = pred_tensor[0].detach().cpu().numpy().astype(np.float32)
                pred_norm_rows.append(pred.copy())
                q_input_norm_rows.append(q_in.copy())
                traces["h_t"].append(h[0].detach().cpu().numpy().astype(np.float32))
                traces["c_t"].append(c[0].detach().cpu().numpy().astype(np.float32))
                for key, value in gates.items():
                    traces[key].append(value[0].detach().cpu().numpy().astype(np.float32))
                q_in = pred.copy()

    pred_norm_arr = np.asarray(pred_norm_rows, dtype=np.float32)
    q_input_norm_arr = np.asarray(q_input_norm_rows, dtype=np.float32)
    q_prediction = denormalize_joints(pred_norm_arr, manifest)
    q_input = denormalize_joints(q_input_norm_arr, manifest)
    q_target = joints_physical[1:steps].astype(np.float32)
    display_steps = np.arange(valid_steps, dtype=np.int64)
    source_frame_index = canonical["source_frame_index"][:valid_steps].astype(np.int64)
    source_timestamp = canonical["source_timestamp"][:valid_steps].astype(np.float64)
    target_timestamp = canonical["source_timestamp"][1:steps].astype(np.float64)
    target_source_frame_index = canonical["source_frame_index"][1:steps].astype(np.int64)
    segments = episode["segments"]
    segment_ids = segment_id_for_rank(segments, canonical["rank"][:valid_steps])
    segment_boundary = canonical["segment_boundary_transition"][:valid_steps].astype(bool)
    abs_err_rad = np.abs(q_prediction - q_target)
    abs_err_deg = abs_err_rad * (180.0 / math.pi)

    raw = episode["raw"]
    raw_time = raw["timestamp"]
    target_ee_pose = interpolate_columns(raw_time, raw["ee_pose"], target_timestamp)
    input_ee_pose = interpolate_columns(raw_time, raw["ee_pose"], source_timestamp)
    target_ee_xy = interpolate_columns(raw_time, raw["ee_xy"], target_timestamp)
    input_ee_xy = interpolate_columns(raw_time, raw["ee_xy"], source_timestamp)
    xy_bounds = episode["metadata"]["xy_normalization"]["bounds"]
    target_ee_xy_norm = normalize_table_xy_rows(target_ee_xy, xy_bounds)
    input_ee_xy_norm = normalize_table_xy_rows(input_ee_xy, xy_bounds)
    target_qpos = interpolate_columns(raw_time, raw["qpos"], target_timestamp)
    input_qpos = interpolate_columns(raw_time, raw["qpos"], source_timestamp)

    result = {
        "seq_id": str(episode["sample"]["seq_id"]),
        "split": str(episode["split"]),
        "model_family": model_family,
        "seed": -1 if checkpoint is None else int(checkpoint.seed),
        "checkpoint_path": "" if checkpoint is None else str(checkpoint.checkpoint_path),
        "checkpoint_hash": "persistence" if checkpoint is None else checkpoint.checkpoint_hash,
        "prediction_mode": mode,
        "timestep": display_steps,
        "source_frame_index": source_frame_index,
        "target_source_frame_index": target_source_frame_index,
        "source_timestamp": source_timestamp,
        "target_timestamp": target_timestamp,
        "length": np.full(valid_steps, int(episode["sample"]["length"]), dtype=np.int64),
        "rank": canonical["rank"][:valid_steps].astype(np.int64),
        "block_id": canonical["block_id"][:valid_steps].astype(np.int64),
        "segment_id": segment_ids,
        "segment_progress": canonical["segment_progress"][:valid_steps].astype(np.float32),
        "segment_boundary": segment_boundary,
        "block_xy": canonical["block_xy"][:valid_steps].astype(np.float32),
        "q_input": q_input.astype(np.float32),
        "q_target": q_target.astype(np.float32),
        "q_prediction": q_prediction.astype(np.float32),
        "q_input_normalized": q_input_norm_arr.astype(np.float32),
        "q_prediction_normalized": pred_norm_arr.astype(np.float32),
        "q_target_normalized": joints_norm[1:steps].astype(np.float32),
        "per_joint_abs_error_deg": abs_err_deg.astype(np.float32),
        "mean_joint_mae_deg": abs_err_deg.mean(axis=1).astype(np.float32),
        "max_joint_error_deg": abs_err_deg.max(axis=1).astype(np.float32),
        "target_ee_pose": target_ee_pose.astype(np.float32),
        "input_ee_pose": input_ee_pose.astype(np.float32),
        "target_ee_xy": target_ee_xy.astype(np.float32),
        "input_ee_xy": input_ee_xy.astype(np.float32),
        "target_ee_xy_norm": target_ee_xy_norm.astype(np.float32),
        "input_ee_xy_norm": input_ee_xy_norm.astype(np.float32),
        "target_qpos": target_qpos.astype(np.float32),
        "input_qpos": input_qpos.astype(np.float32),
        "transition_mask": transition_mask[:valid_steps],
        "block_order": np.asarray(episode["sample"]["block_order"], dtype=np.int64),
        "images_uint8": images_uint8[:valid_steps].astype(np.uint8),
    }
    for key, values in traces.items():
        if values:
            result[key] = np.asarray(values, dtype=np.float32)
    return result


def prediction_output_paths(root: Path, key: PredictionKey) -> dict[str, Path]:
    base = root / "predictions" / key.model_family / str(key.seed) / key.mode
    return {
        "parquet": base / f"{key.seq_id}.parquet",
        "npz": base / f"{key.seq_id}.npz",
        "json": base / f"{key.seq_id}.json",
    }


def prediction_metadata(prediction: dict[str, Any], manifest: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "corsi_prediction_export_v3",
        "seq_id": prediction["seq_id"],
        "split": prediction["split"],
        "model_family": prediction["model_family"],
        "seed": int(prediction["seed"]),
        "checkpoint_path": prediction["checkpoint_path"],
        "checkpoint_hash": prediction["checkpoint_hash"],
        "prediction_mode": prediction["prediction_mode"],
        "row_count": int(len(prediction["timestep"])),
        "final_timestep_excluded": True,
        "joint_names": manifest["joint_names"],
        "canonical_fingerprint": manifest["canonical_fingerprint"],
        "mean_joint_mae_deg": float(np.mean(prediction["mean_joint_mae_deg"])),
        "max_joint_error_deg": float(np.max(prediction["max_joint_error_deg"])),
        **extra,
    }


def _prediction_hash_inputs(prediction: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    predicted_ee = prediction.get("predicted_ee_pose")
    ee_error = prediction.get("ee_position_error_cm")
    return {
        "metadata": metadata,
        "q_prediction_sha256": hashlib.sha256(prediction["q_prediction"].tobytes()).hexdigest(),
        "q_target_sha256": hashlib.sha256(prediction["q_target"].tobytes()).hexdigest(),
        "predicted_ee_pose_sha256": ""
        if predicted_ee is None
        else hashlib.sha256(np.asarray(predicted_ee, dtype=np.float32).tobytes()).hexdigest(),
        "ee_position_error_cm_sha256": ""
        if ee_error is None
        else hashlib.sha256(np.asarray(ee_error, dtype=np.float32).tobytes()).hexdigest(),
        "mode": prediction["prediction_mode"],
    }


def write_prediction_export(
    root: Path,
    prediction: dict[str, Any],
    manifest: dict[str, Any],
    *,
    resume: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import pandas as pd

    key = PredictionKey(
        model_family=str(prediction["model_family"]),
        seed=int(prediction["seed"]),
        mode=str(prediction["prediction_mode"]),
        seq_id=str(prediction["seq_id"]),
    )
    paths = prediction_output_paths(root, key)
    metadata = prediction_metadata(prediction, manifest, extra_metadata or {})
    content_hash = stable_hash(_prediction_hash_inputs(prediction, metadata))
    metadata["content_hash"] = content_hash
    if resume and paths["json"].exists():
        old = read_json(paths["json"])
        if old.get("content_hash") == content_hash and paths["npz"].exists() and paths["parquet"].exists():
            return {"status": "skipped", **metadata, "paths": {k: str(v) for k, v in paths.items()}}

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    row_count = int(len(prediction["timestep"]))
    predicted_ee_pose = prediction.get("predicted_ee_pose")
    if predicted_ee_pose is None:
        predicted_ee_pose = np.full((row_count, 7), np.nan, dtype=np.float32)
    fk_target_ee_pose = prediction.get("fk_target_ee_pose")
    if fk_target_ee_pose is None:
        fk_target_ee_pose = np.full((row_count, 7), np.nan, dtype=np.float32)
    ee_position_error_cm = prediction.get("ee_position_error_cm")
    if ee_position_error_cm is None:
        ee_position_error_cm = np.full((row_count,), np.nan, dtype=np.float32)

    np.savez_compressed(
        paths["npz"],
        timestep=prediction["timestep"],
        source_frame_index=prediction["source_frame_index"],
        target_source_frame_index=prediction["target_source_frame_index"],
        source_timestamp=prediction["source_timestamp"],
        target_timestamp=prediction["target_timestamp"],
        rank=prediction["rank"],
        block_id=prediction["block_id"],
        segment_id=prediction["segment_id"],
        segment_progress=prediction["segment_progress"],
        segment_boundary=prediction["segment_boundary"],
        block_xy=prediction["block_xy"],
        q_input=prediction["q_input"],
        q_target=prediction["q_target"],
        q_prediction=prediction["q_prediction"],
        per_joint_abs_error_deg=prediction["per_joint_abs_error_deg"],
        mean_joint_mae_deg=prediction["mean_joint_mae_deg"],
        max_joint_error_deg=prediction["max_joint_error_deg"],
        target_ee_pose=prediction["target_ee_pose"],
        input_ee_pose=prediction["input_ee_pose"],
        predicted_ee_pose=np.asarray(predicted_ee_pose, dtype=np.float32),
        fk_target_ee_pose=np.asarray(fk_target_ee_pose, dtype=np.float32),
        ee_position_error_cm=np.asarray(ee_position_error_cm, dtype=np.float32),
        target_ee_xy=prediction["target_ee_xy"],
        input_ee_xy=prediction["input_ee_xy"],
        target_ee_xy_norm=prediction["target_ee_xy_norm"],
        input_ee_xy_norm=prediction["input_ee_xy_norm"],
        target_qpos=prediction["target_qpos"],
        input_qpos=prediction["input_qpos"],
    )
    rows = []
    for index in range(int(len(prediction["timestep"]))):
        rows.append(
            {
                "seq_id": prediction["seq_id"],
                "split": prediction["split"],
                "model_family": prediction["model_family"],
                "seed": int(prediction["seed"]),
                "checkpoint_path": prediction["checkpoint_path"],
                "checkpoint_hash": prediction["checkpoint_hash"],
                "prediction_mode": prediction["prediction_mode"],
                "timestep": int(prediction["timestep"][index]),
                "source_frame_index": int(prediction["source_frame_index"][index]),
                "source_timestamp": float(prediction["source_timestamp"][index]),
                "length": int(prediction["length"][index]),
                "rank": int(prediction["rank"][index]),
                "block_id": int(prediction["block_id"][index]),
                "segment_id": int(prediction["segment_id"][index]),
                "segment_progress": float(prediction["segment_progress"][index]),
                "segment_boundary": bool(prediction["segment_boundary"][index]),
                "q_input": prediction["q_input"][index].astype(float).tolist(),
                "q_target": prediction["q_target"][index].astype(float).tolist(),
                "q_prediction": prediction["q_prediction"][index].astype(float).tolist(),
                "per_joint_abs_error_deg": prediction["per_joint_abs_error_deg"][index].astype(float).tolist(),
                "mean_joint_mae_deg": float(prediction["mean_joint_mae_deg"][index]),
                "max_joint_error_deg": float(prediction["max_joint_error_deg"][index]),
                "target_ee_pose": prediction["target_ee_pose"][index].astype(float).tolist(),
                "target_ee_xy_norm": prediction["target_ee_xy_norm"][index].astype(float).tolist(),
                "predicted_ee_pose": np.asarray(predicted_ee_pose[index], dtype=float).tolist(),
                "fk_target_ee_pose": np.asarray(fk_target_ee_pose[index], dtype=float).tolist(),
                "ee_position_error_cm": float(np.asarray(ee_position_error_cm)[index]),
            }
        )
    pd.DataFrame(rows).to_parquet(paths["parquet"], index=False)
    write_json(paths["json"], {**metadata, "paths": {k: str(v) for k, v in paths.items()}})
    return {"status": "written", **metadata, "paths": {k: str(v) for k, v in paths.items()}}


def episode_metrics(prediction: dict[str, Any]) -> dict[str, Any]:
    sq = (prediction["q_prediction_normalized"] - prediction["q_target_normalized"]) ** 2
    boundary = prediction["segment_boundary"].astype(bool)
    endpoint_mask = prediction["segment_progress"] >= 0.999
    return {
        "seq_id": prediction["seq_id"],
        "length": int(prediction["length"][0]),
        "normalized_rmse": float(math.sqrt(float(np.mean(sq)))),
        "mean_joint_mae_deg": float(np.mean(prediction["mean_joint_mae_deg"])),
        "max_joint_error_deg": float(np.max(prediction["max_joint_error_deg"])),
        "boundary_mean_joint_mae_deg": float(np.mean(prediction["mean_joint_mae_deg"][boundary]))
        if np.any(boundary)
        else 0.0,
        "endpoint_mean_joint_mae_deg": float(np.mean(prediction["mean_joint_mae_deg"][endpoint_mask]))
        if np.any(endpoint_mask)
        else float(np.mean(prediction["mean_joint_mae_deg"])),
        "endpoint_max_joint_error_deg": float(np.max(prediction["max_joint_error_deg"][endpoint_mask]))
        if np.any(endpoint_mask)
        else float(np.max(prediction["max_joint_error_deg"])),
    }


def deterministic_choice(rows: Sequence[dict[str, Any]], key_fn, *, reverse: bool = False) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select from empty rows")
    return sorted(rows, key=lambda row: (key_fn(row), str(row["seq_id"])), reverse=reverse)[0]


def select_episodes(
    *,
    visual_metrics: dict[str, dict[str, Any]],
    joint_metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = list(visual_metrics.values())
    selected: list[dict[str, Any]] = []
    for length in range(2, 10):
        length_rows = sorted([row for row in rows if int(row["length"]) == length], key=lambda item: item["normalized_rmse"])
        if not length_rows:
            continue
        median_value = length_rows[len(length_rows) // 2]["normalized_rmse"]
        row = sorted(
            length_rows,
            key=lambda item: (abs(float(item["normalized_rmse"]) - float(median_value)), str(item["seq_id"])),
        )[0]
        selected.append(
            {
                "category": f"median_rmse_length_{length}",
                "seq_id": row["seq_id"],
                "length": int(length),
                "visual_normalized_rmse": row["normalized_rmse"],
                "tie_break": "closest to per-length median visual_joint RMSE, then lexicographic seq_id",
            }
        )
    lowest = sorted(rows, key=lambda row: (float(row["normalized_rmse"]), str(row["seq_id"])))[0]
    highest = sorted(rows, key=lambda row: (-float(row["normalized_rmse"]), str(row["seq_id"])))[0]
    boundary = sorted(rows, key=lambda row: (-float(row["boundary_mean_joint_mae_deg"]), str(row["seq_id"])))[0]
    endpoint = sorted(rows, key=lambda row: (-float(row["endpoint_mean_joint_mae_deg"]), str(row["seq_id"])))[0]
    selected.extend(
        [
            {
                "category": "global_lowest_rmse",
                "seq_id": lowest["seq_id"],
                "length": int(lowest["length"]),
                "visual_normalized_rmse": lowest["normalized_rmse"],
                "tie_break": "lowest visual_joint RMSE, then lexicographic seq_id",
            },
            {
                "category": "global_highest_rmse",
                "seq_id": highest["seq_id"],
                "length": int(highest["length"]),
                "visual_normalized_rmse": highest["normalized_rmse"],
                "tie_break": "highest visual_joint RMSE, then lexicographic seq_id",
            },
            {
                "category": "largest_segment_boundary_error",
                "seq_id": boundary["seq_id"],
                "length": int(boundary["length"]),
                "boundary_mean_joint_mae_deg": boundary["boundary_mean_joint_mae_deg"],
                "tie_break": "highest visual_joint boundary mean MAE, then lexicographic seq_id",
            },
            {
                "category": "largest_endpoint_error",
                "seq_id": endpoint["seq_id"],
                "length": int(endpoint["length"]),
                "endpoint_mean_joint_mae_deg": endpoint["endpoint_mean_joint_mae_deg"],
                "tie_break": "highest endpoint mean joint MAE before FK endpoint EE is available, then lexicographic seq_id",
            },
        ]
    )
    gains = []
    for seq_id, visual in visual_metrics.items():
        joint = joint_metrics.get(seq_id)
        if joint is None:
            continue
        gains.append(
            {
                "seq_id": seq_id,
                "length": int(visual["length"]),
                "visual_mean_joint_mae_deg": float(visual["mean_joint_mae_deg"]),
                "joint_mean_joint_mae_deg": float(joint["mean_joint_mae_deg"]),
                "visual_gain_over_joint_deg": float(joint["mean_joint_mae_deg"] - visual["mean_joint_mae_deg"]),
            }
        )
    best_gain = sorted(gains, key=lambda row: (-row["visual_gain_over_joint_deg"], str(row["seq_id"])))[0]
    worst_gap = sorted(gains, key=lambda row: (row["visual_gain_over_joint_deg"], str(row["seq_id"])))[0]
    selected.extend(
        [
            {
                "category": "visual_best_gain_over_joint_only",
                **best_gain,
                "tie_break": "largest joint_only MAE minus visual_joint MAE, then lexicographic seq_id",
            },
            {
                "category": "visual_worst_gap_vs_joint_only",
                **worst_gap,
                "tie_break": "smallest joint_only MAE minus visual_joint MAE, then lexicographic seq_id",
            },
        ]
    )
    seen: set[tuple[str, str]] = set()
    unique = []
    for row in selected:
        key = (str(row["category"]), str(row["seq_id"]))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def unique_selected_seq_ids(selection: Sequence[dict[str, Any]]) -> list[str]:
    return sorted({str(row["seq_id"]) for row in selection})


class StaticPoseRenderer:
    """Deterministic qpos setter and optional free-camera renderer."""

    def __init__(self, episode: dict[str, Any], *, offscreen: bool, width: int = 512, height: int = 512):
        import robosuite as suite
        from corsi.envs.robosuite_corsi import standard_corsi_robosuite_board_size
        from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

        metadata = episode["metadata"]
        block_positions = metadata["block_positions"]
        ordered = [
            tuple(float(v) for v in block_positions[str(index)]["xyz_world"][:2])
            for index in sorted(int(key) for key in block_positions.keys())
        ]
        self.env = suite.make(
            env_name="CorsiSceneDemo",
            robots="PandaDexRH",
            controller_configs=load_composite_controller_config(controller="BASIC"),
            has_renderer=False,
            has_offscreen_renderer=bool(offscreen),
            use_camera_obs=False,
            ignore_done=True,
            block_xy_positions=ordered,
            corsi_board_size_xy=standard_corsi_robosuite_board_size(),
            seed=int(episode["sample"].get("seed", 0)),
        )
        self.width = int(width)
        self.height = int(height)
        robot = self.env.robots[0]
        self.robot = robot
        self.arm_qpos_indexes = np.asarray(robot._ref_arm_joint_pos_indexes, dtype=np.int64)
        gripper_indexes = robot._ref_gripper_joint_pos_indexes
        if isinstance(gripper_indexes, dict):
            arm = robot.arms[0] if getattr(robot, "arms", None) else next(iter(gripper_indexes))
            gripper_indexes = gripper_indexes[arm]
        self.gripper_qpos_indexes = np.asarray(gripper_indexes, dtype=np.int64)
        raw_qpos = np.asarray(episode["raw"]["qpos"], dtype=np.float32)
        self.fixed_hand_qpos = raw_qpos[0, self.gripper_qpos_indexes].astype(np.float32)
        self.site_name = "gripper0_right_index_tip_site"
        self.site_id = int(self.env.sim.model.site_name2id(self.site_name))

    def close(self) -> None:
        if getattr(self, "env", None) is not None:
            self.env.close()

    def set_arm(self, q_arm: np.ndarray) -> None:
        import mujoco

        self.env.sim.data.qpos[self.arm_qpos_indexes] = np.asarray(q_arm, dtype=np.float64)
        self.env.sim.data.qpos[self.gripper_qpos_indexes] = self.fixed_hand_qpos.astype(np.float64)
        self.env.sim.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.sim.model._model, self.env.sim.data._data)

    def ee_pose(self, q_arm: np.ndarray) -> np.ndarray:
        self.set_arm(q_arm)
        pos = np.asarray(self.env.sim.data.site_xpos[self.site_id], dtype=np.float32)
        mat = np.asarray(self.env.sim.data.site_xmat[self.site_id], dtype=np.float32).reshape(3, 3)
        quat = mat_to_quat_wxyz(mat)
        return np.concatenate([pos, quat.astype(np.float32)], axis=0)

    def render(self, q_arm: np.ndarray) -> np.ndarray:
        from corsi.envs.robosuite_corsi import render_tuned_free_camera_frame

        self.set_arm(q_arm)
        return render_tuned_free_camera_frame(self.env, width=self.width, height=self.height)

    def joint_limits(self) -> np.ndarray:
        ranges = []
        for joint_id in getattr(self.robot, "_ref_arm_joint_indexes", []):
            ranges.append(np.asarray(self.env.sim.model.jnt_range[int(joint_id)], dtype=np.float32))
        limits = np.asarray(ranges, dtype=np.float32)
        if limits.shape != (7, 2):
            return DEFAULT_PANDA_ARM_JOINT_LIMITS.copy()
        return limits

    def mapping_metadata(self) -> dict[str, Any]:
        joint_names = []
        for joint_id in getattr(self.robot, "_ref_arm_joint_indexes", []):
            joint_names.append(str(self.env.sim.model.joint_id2name(int(joint_id))))
        return {
            "robot": self.robot.name,
            "site_name": self.site_name,
            "site_id": self.site_id,
            "arm_joint_names": joint_names,
            "arm_qpos_indexes": self.arm_qpos_indexes.tolist(),
            "fixed_hand_qpos_indexes": self.gripper_qpos_indexes.tolist(),
            "fixed_hand_qpos": self.fixed_hand_qpos.astype(float).tolist(),
            "joint_limits": self.joint_limits().astype(float).tolist(),
            "source": "robosuite PandaDexRH MuJoCo model jnt_range",
            "fk_qpos_policy": "set seven Panda arm qpos, preserve first-frame hand qpos, call mj_forward, read gripper0_right_index_tip_site",
        }


def resolve_joint_limit_metadata(episode: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    renderer = None
    try:
        renderer = StaticPoseRenderer(episode, offscreen=False)
        metadata = renderer.mapping_metadata()
        limits = np.asarray(metadata["joint_limits"], dtype=np.float32)
        if limits.shape == (7, 2) and np.isfinite(limits).all():
            return metadata
    except Exception as exc:  # pragma: no cover - depends on MuJoCo runtime
        fallback = fallback_joint_limit_metadata(manifest["joint_names"])
        fallback["resolution_error"] = str(exc)
        fallback["traceback"] = traceback.format_exc()
        return fallback
    finally:
        if renderer is not None:
            renderer.close()
    return fallback_joint_limit_metadata(manifest["joint_names"])


def mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to wxyz quaternion."""

    m = np.asarray(mat, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 0 else quat


def validate_fk_for_predictions(
    episodes: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    *,
    max_rows_per_episode: int = 256,
) -> dict[str, Any]:
    errors = []
    examples = []
    for episode, prediction in zip(episodes, predictions):
        renderer = None
        try:
            renderer = StaticPoseRenderer(episode, offscreen=False)
            count = int(len(prediction["q_target"]))
            if count > max_rows_per_episode:
                indices = np.linspace(0, count - 1, max_rows_per_episode).round().astype(np.int64)
            else:
                indices = np.arange(count, dtype=np.int64)
            for index in indices:
                fk_pose = renderer.ee_pose(prediction["q_target"][index])
                target = prediction["target_ee_pose"][index]
                error_m = float(np.linalg.norm(fk_pose[:3] - target[:3]))
                errors.append(error_m)
                if len(examples) < 5:
                    examples.append(
                        {
                            "seq_id": prediction["seq_id"],
                            "timestep": int(prediction["timestep"][index]),
                            "fk_pos": fk_pose[:3].tolist(),
                            "target_pos": target[:3].tolist(),
                            "error_m": error_m,
                        }
                    )
        except Exception as exc:  # pragma: no cover - depends on MuJoCo runtime
            return {
                "tier_b_available": False,
                "reason": f"FK validation failed to run: {exc}",
                "traceback": traceback.format_exc(),
                "median_error_m": None,
                "max_error_m": None,
                "examples": examples,
            }
        finally:
            if renderer is not None:
                renderer.close()
    if not errors:
        return {
            "tier_b_available": False,
            "reason": "no FK validation rows",
            "median_error_m": None,
            "max_error_m": None,
            "examples": examples,
        }
    median_error = float(np.median(np.asarray(errors)))
    max_error = float(np.max(np.asarray(errors)))
    passed = median_error <= 0.002 and max_error <= 0.005
    return {
        "tier_b_available": bool(passed),
        "reason": "passed" if passed else "FK error exceeded threshold",
        "median_error_m": median_error,
        "max_error_m": max_error,
        "thresholds": {"median_m": 0.002, "max_m": 0.005},
        "rows_checked": int(len(errors)),
        "examples": examples,
    }


def attach_fk_ee(prediction: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    renderer = StaticPoseRenderer(episode, offscreen=False)
    try:
        pred_pose = np.asarray([renderer.ee_pose(q) for q in prediction["q_prediction"]], dtype=np.float32)
        target_pose = np.asarray([renderer.ee_pose(q) for q in prediction["q_target"]], dtype=np.float32)
    finally:
        renderer.close()
    prediction["predicted_ee_pose"] = pred_pose
    prediction["fk_target_ee_pose"] = target_pose
    prediction["ee_position_error_cm"] = (np.linalg.norm(pred_pose[:, :3] - target_pose[:, :3], axis=1) * 100.0).astype(
        np.float32
    )
    return prediction


def image_chw_to_hwc(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] == 3:
        return np.transpose(image, (1, 2, 0))
    return image


def _pil_font(size: int):
    from PIL import ImageFont

    for name in ["DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_multiline(draw, xy: tuple[int, int], text: str, *, font, fill=(20, 24, 28), spacing: int = 5) -> None:
    x, y = xy
    for line in text.split("\n"):
        draw.text((x, y), line, font=font, fill=fill)
        y += int(getattr(font, "size", 12)) + spacing


def _line_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[tuple[int, int]]:
    left, top, right, bottom = rect
    width = max(1, right - left)
    height = max(1, bottom - top)
    x_den = max(1e-6, float(x_max - x_min))
    y_den = max(1e-6, float(y_max - y_min))
    points = []
    for x_value, y_value in zip(x_values, y_values):
        x = left + int(round((float(x_value) - x_min) / x_den * width))
        y = bottom - int(round((float(y_value) - y_min) / y_den * height))
        points.append((x, y))
    return points


def _draw_axes(draw, rect: tuple[int, int, int, int], *, title: str, font, small_font) -> None:
    left, top, right, bottom = rect
    draw.rectangle(rect, outline=(80, 86, 94), width=2)
    for fraction in [0.25, 0.5, 0.75]:
        y = int(round(top + (bottom - top) * fraction))
        draw.line([(left, y), (right, y)], fill=(226, 230, 235), width=1)
    draw.text((left, top - 28), title, font=font, fill=(20, 24, 28))
    draw.text((left + 6, bottom + 6), "canonical timestep", font=small_font, fill=(80, 86, 94))


def _draw_joint_trajectory_plot(
    draw,
    rect: tuple[int, int, int, int],
    prediction: dict[str, Any],
    step_index: int,
    *,
    title: str,
    font,
    small_font,
) -> None:
    timesteps = prediction["timestep"]
    current_t = int(timesteps[step_index])
    lo, hi = _joint_limits_from_prediction(prediction)
    y_min = float(lo.min())
    y_max = float(hi.max())
    _draw_axes(draw, rect, title=title, font=font, small_font=small_font)
    left, top, right, bottom = rect
    colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
    ]
    for boundary_t in timesteps[prediction["segment_boundary"].astype(bool)]:
        x = _line_points(
            np.asarray([boundary_t]),
            np.asarray([y_min]),
            rect,
            x_min=float(timesteps[0]),
            x_max=float(timesteps[-1]),
            y_min=y_min,
            y_max=y_max,
        )[0][0]
        draw.line([(x, top), (x, bottom)], fill=(150, 154, 160), width=1)
    for joint_index, color in enumerate(colors):
        target_points = _line_points(
            timesteps,
            prediction["q_target"][:, joint_index],
            rect,
            x_min=float(timesteps[0]),
            x_max=float(timesteps[-1]),
            y_min=y_min,
            y_max=y_max,
        )
        pred_points = _line_points(
            timesteps,
            prediction["q_prediction"][:, joint_index],
            rect,
            x_min=float(timesteps[0]),
            x_max=float(timesteps[-1]),
            y_min=y_min,
            y_max=y_max,
        )
        if len(target_points) >= 2:
            draw.line(target_points, fill=color, width=2)
            draw.line(pred_points, fill=tuple(max(0, c - 55) for c in color), width=1)
    current_x = _line_points(
        np.asarray([current_t]),
        np.asarray([y_min]),
        rect,
        x_min=float(timesteps[0]),
        x_max=float(timesteps[-1]),
        y_min=y_min,
        y_max=y_max,
    )[0][0]
    draw.line([(current_x, top), (current_x, bottom)], fill=(0, 0, 0), width=3)
    draw.text((left + 8, top + 8), "target solid, prediction darker", font=small_font, fill=(60, 64, 70))
    draw.text((left + 8, bottom - 24), "fixed Panda joint-limit envelope", font=small_font, fill=(60, 64, 70))


def _draw_error_timeline(
    draw,
    rect: tuple[int, int, int, int],
    prediction: dict[str, Any],
    step_index: int,
    *,
    title: str,
    font,
    small_font,
) -> None:
    timesteps = prediction["timestep"]
    values = prediction["mean_joint_mae_deg"]
    y_max = max(1e-6, float(np.max(values)) * 1.15)
    _draw_axes(draw, rect, title=title, font=font, small_font=small_font)
    points = _line_points(
        timesteps,
        values,
        rect,
        x_min=float(timesteps[0]),
        x_max=float(timesteps[-1]),
        y_min=0.0,
        y_max=y_max,
    )
    if len(points) >= 2:
        draw.line(points, fill=(31, 119, 180), width=3)
    current_t = int(timesteps[step_index])
    current_x = _line_points(
        np.asarray([current_t]),
        np.asarray([0.0]),
        rect,
        x_min=float(timesteps[0]),
        x_max=float(timesteps[-1]),
        y_min=0.0,
        y_max=y_max,
    )[0][0]
    draw.line([(current_x, rect[1]), (current_x, rect[3])], fill=(0, 0, 0), width=3)
    draw.text((rect[0] + 8, rect[1] + 8), "mean joint abs error (deg)", font=small_font, fill=(60, 64, 70))


def _draw_current_joint_bars(
    draw,
    rect: tuple[int, int, int, int],
    prediction: dict[str, Any],
    step_index: int,
    *,
    title: str,
    font,
    small_font,
) -> None:
    left, top, right, bottom = rect
    draw.rectangle(rect, outline=(80, 86, 94), width=2)
    draw.text((left, top - 28), title, font=font, fill=(20, 24, 28))
    lo, hi = _joint_limits_from_prediction(prediction)
    y_min = float(lo.min())
    y_max = float(hi.max())
    zero_y = bottom - int(round((0.0 - y_min) / max(1e-6, y_max - y_min) * (bottom - top)))
    draw.line([(left, zero_y), (right, zero_y)], fill=(170, 175, 182), width=1)
    bar_group = (right - left) / 7.0
    for index in range(7):
        target = float(prediction["q_target"][step_index, index])
        pred = float(prediction["q_prediction"][step_index, index])
        base_x = left + int(round(index * bar_group))
        center = base_x + int(round(bar_group * 0.5))
        for offset, value, color in [(-8, target, (31, 119, 180)), (8, pred, (214, 39, 40))]:
            y = bottom - int(round((value - y_min) / max(1e-6, y_max - y_min) * (bottom - top)))
            draw.rectangle((center + offset - 5, min(zero_y, y), center + offset + 5, max(zero_y, y)), fill=color)
        draw.text((center - 6, bottom + 4), str(index + 1), font=small_font, fill=(80, 86, 94))
    draw.text((left + 8, top + 8), "blue target, red prediction", font=small_font, fill=(60, 64, 70))


def _draw_ee_plot(
    draw,
    rect: tuple[int, int, int, int],
    prediction: dict[str, Any],
    step_index: int,
    *,
    title: str,
    font,
    small_font,
) -> None:
    left, top, right, bottom = rect
    _draw_axes(draw, rect, title=title, font=font, small_font=small_font)
    target_xy = np.asarray(prediction.get("target_ee_xy_norm", prediction["target_ee_pose"][:, :2]), dtype=np.float32)
    block_xy = np.asarray(prediction["block_xy"], dtype=np.float32)
    pred_norm = prediction.get("predicted_ee_xy_norm")
    pred_xy = None if pred_norm is None else np.asarray(pred_norm, dtype=np.float32)
    values = [target_xy, block_xy]
    if pred_xy is not None and np.isfinite(pred_xy).any():
        values.append(pred_xy[np.isfinite(pred_xy).all(axis=1)])
    all_xy = np.concatenate([value for value in values if len(value)], axis=0)
    xy_min = all_xy.min(axis=0)
    xy_max = all_xy.max(axis=0)
    margin = np.maximum((xy_max - xy_min) * 0.15, 0.03)
    x_min, y_min = (xy_min - margin).astype(float)
    x_max, y_max = (xy_max + margin).astype(float)

    def point(xy: np.ndarray) -> tuple[int, int]:
        x = left + int(round((float(xy[0]) - x_min) / max(1e-6, x_max - x_min) * (right - left)))
        y = bottom - int(round((float(xy[1]) - y_min) / max(1e-6, y_max - y_min) * (bottom - top)))
        return x, y

    target_points = [point(xy) for xy in target_xy]
    if len(target_points) >= 2:
        draw.line(target_points, fill=(31, 119, 180), width=3)
    if pred_xy is not None and np.isfinite(pred_xy).all():
        pred_points = [point(xy) for xy in pred_xy]
        if len(pred_points) >= 2:
            draw.line(pred_points, fill=(214, 39, 40), width=2)
    for xy in block_xy:
        x, y = point(xy)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(20, 24, 28))
    x, y = point(block_xy[step_index])
    draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=(214, 39, 40), width=3)
    draw.text((left + 8, bottom - 24), "target block marker is analysis-only", font=small_font, fill=(60, 64, 70))


def render_fast_tier_a_frame(
    *,
    prediction: dict[str, Any],
    step_index: int,
    output_path: Path,
    title: str,
) -> None:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (1920, 1080), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _pil_font(24)
    font = _pil_font(18)
    small_font = _pil_font(13)
    image = Image.fromarray(image_chw_to_hwc(prediction["images_uint8"][step_index]).astype(np.uint8)).convert("RGB")
    image.thumbnail((520, 390))
    canvas.paste(image, (40, 90))
    draw.text((40, 58), "Input RGB I_t", font=font, fill=(20, 24, 28))
    _draw_current_joint_bars(
        draw,
        (610, 90, 1160, 420),
        prediction,
        step_index,
        title="Current q_{t+1}: target vs prediction",
        font=font,
        small_font=small_font,
    )
    _draw_error_timeline(
        draw,
        (1260, 90, 1870, 420),
        prediction,
        step_index,
        title="Per-frame prediction error",
        font=font,
        small_font=small_font,
    )
    _draw_joint_trajectory_plot(
        draw,
        (40, 560, 1230, 1010),
        prediction,
        step_index,
        title="7 joint trajectories",
        font=font,
        small_font=small_font,
    )
    _draw_ee_plot(
        draw,
        (1320, 560, 1870, 1010),
        prediction,
        step_index,
        title="EE top-down trajectory",
        font=font,
        small_font=small_font,
    )
    ee_error = prediction.get("ee_position_error_cm")
    ee_text = "n/a" if ee_error is None else f"{float(ee_error[step_index]):.2f} cm"
    header = (
        f"{title} | seq={prediction['seq_id']} model={prediction['model_family']} seed={prediction['seed']} "
        f"mode={prediction['prediction_mode']} t={int(prediction['timestep'][step_index])}"
    )
    draw.text((40, 18), header[:180], font=title_font, fill=(20, 24, 28))
    summary = (
        f"length={int(prediction['length'][step_index])} rank/block="
        f"{int(prediction['rank'][step_index])}/{int(prediction['block_id'][step_index])} "
        f"progress={float(prediction['segment_progress'][step_index]):.2f} "
        f"mean MAE={float(prediction['mean_joint_mae_deg'][step_index]):.3f} deg "
        f"max err={float(prediction['max_joint_error_deg'][step_index]):.3f} deg EE err={ee_text}"
    )
    draw.text((40, 1030), summary[:200], font=font, fill=(20, 24, 28))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def render_fast_comparison_frame(
    *,
    predictions_by_model: dict[str, dict[str, Any]],
    step_index: int,
    output_path: Path,
    title: str,
) -> None:
    from PIL import Image, ImageDraw

    ordered = [model for model in MODEL_FAMILIES if model in predictions_by_model]
    reference = predictions_by_model[ordered[0]]
    canvas = Image.new("RGB", (1920, 1080), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _pil_font(24)
    font = _pil_font(17)
    small_font = _pil_font(12)
    draw.text(
        (34, 18),
        f"{title} | seq={reference['seq_id']} mode={reference['prediction_mode']} t={int(reference['timestep'][step_index])}",
        font=title_font,
        fill=(20, 24, 28),
    )
    image = Image.fromarray(image_chw_to_hwc(reference["images_uint8"][step_index]).astype(np.uint8)).convert("RGB")
    image.thumbnail((420, 315))
    canvas.paste(image, (34, 90))
    draw.text((34, 58), "Input RGB I_t", font=font, fill=(20, 24, 28))
    _draw_ee_plot(
        draw,
        (34, 520, 454, 1000),
        reference,
        step_index,
        title="Target EE path",
        font=font,
        small_font=small_font,
    )
    column_rects = [(510, 90, 940, 430), (1010, 90, 1440, 430), (1490, 90, 1888, 430)]
    timeline_rects = [(510, 560, 940, 1000), (1010, 560, 1440, 1000), (1490, 560, 1888, 1000)]
    for model_family, bar_rect, timeline_rect in zip(ordered, column_rects, timeline_rects):
        prediction = predictions_by_model[model_family]
        _draw_current_joint_bars(
            draw,
            bar_rect,
            prediction,
            step_index,
            title=f"{model_family}: q_target vs q_hat",
            font=font,
            small_font=small_font,
        )
        _draw_error_timeline(
            draw,
            timeline_rect,
            prediction,
            step_index,
            title=f"{model_family}: mean MAE",
            font=font,
            small_font=small_font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _joint_limits_from_prediction(prediction: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    limits = prediction.get("joint_limits")
    if limits is not None:
        limits = np.asarray(limits, dtype=np.float32)
        if limits.shape == (7, 2) and np.isfinite(limits).all():
            return limits[:, 0].copy(), limits[:, 1].copy()
    return DEFAULT_PANDA_ARM_JOINT_LIMITS[:, 0].copy(), DEFAULT_PANDA_ARM_JOINT_LIMITS[:, 1].copy()


def render_plot_frame(
    *,
    prediction: dict[str, Any],
    step_index: int,
    gt_render: np.ndarray | None,
    pred_render: np.ndarray | None,
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = image_chw_to_hwc(prediction["images_uint8"][step_index])
    timesteps = prediction["timestep"]
    current_t = int(timesteps[step_index])
    lo, hi = _joint_limits_from_prediction(prediction)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])
    ax0 = fig.add_subplot(grid[0, 0])
    ax0.imshow(image)
    ax0.set_title("Input RGB I_t")
    ax0.axis("off")
    for ax, frame, name in [
        (fig.add_subplot(grid[0, 1]), gt_render, "Ground truth q_{t+1}"),
        (fig.add_subplot(grid[0, 2]), pred_render, "Prediction q_hat_{t+1}"),
    ]:
        if frame is not None:
            ax.imshow(frame)
        else:
            ax.text(0.5, 0.5, "Tier-B render unavailable", ha="center", va="center", fontsize=16)
        ax.set_title(name)
        ax.axis("off")
    ax_joint = fig.add_subplot(grid[1, :2])
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    for joint_index in range(7):
        ax_joint.plot(timesteps, prediction["q_target"][:, joint_index], color=colors[joint_index], lw=1.4)
        ax_joint.plot(timesteps, prediction["q_prediction"][:, joint_index], color=colors[joint_index], ls="--", lw=1.2)
    for boundary_t in timesteps[prediction["segment_boundary"].astype(bool)]:
        ax_joint.axvline(int(boundary_t), color="k", lw=0.5, alpha=0.25)
    ax_joint.axvline(current_t, color="black", lw=1.5)
    ax_joint.set_ylim(float(lo.min()), float(hi.max()))
    ax_joint.set_title("7 joint trajectories: target solid, prediction dashed")
    ax_joint.set_xlabel("canonical timestep")
    ax_joint.set_ylabel("joint angle (rad)")
    ax_joint.text(0.01, 0.02, "fixed Panda joint-limit envelope", transform=ax_joint.transAxes, fontsize=8)
    ax_joint.grid(alpha=0.2)
    ax_ee = fig.add_subplot(grid[1, 2])
    target_xy = prediction.get("target_ee_xy_norm", prediction["target_ee_pose"][:, :2])
    pred_xy = prediction.get("predicted_ee_xy_norm")
    ax_ee.plot(target_xy[:, 0], target_xy[:, 1], label="target EE", lw=1.5)
    if pred_xy is not None and np.isfinite(pred_xy).any():
        ax_ee.plot(pred_xy[:, 0], pred_xy[:, 1], label="pred EE", lw=1.5, ls="--")
    block_xy = prediction["block_xy"]
    ax_ee.scatter(block_xy[:, 0], block_xy[:, 1], s=18, c="black", alpha=0.4, label="block centers")
    ax_ee.scatter(block_xy[step_index, 0], block_xy[step_index, 1], s=80, facecolors="none", edgecolors="red")
    ax_ee.text(
        0.01,
        0.01,
        "Target annotation is analysis-only; not model input.",
        transform=ax_ee.transAxes,
        fontsize=8,
        va="bottom",
    )
    ax_ee.set_title("EE top-down trajectory (normalized board XY)")
    ax_ee.set_aspect("equal", adjustable="box")
    ax_ee.legend(fontsize=8, loc="upper right")
    ax_ee.grid(alpha=0.2)
    ee_error = prediction.get("ee_position_error_cm")
    ee_text = "n/a" if ee_error is None else f"{float(ee_error[step_index]):.2f} cm"
    fig.suptitle(
        f"{title}\nseq={prediction['seq_id']} model={prediction['model_family']} seed={prediction['seed']} "
        f"mode={prediction['prediction_mode']} length={int(prediction['length'][step_index])} "
        f"rank/block={int(prediction['rank'][step_index])}/{int(prediction['block_id'][step_index])} "
        f"progress={float(prediction['segment_progress'][step_index]):.2f} "
        f"mean MAE={float(prediction['mean_joint_mae_deg'][step_index]):.3f} deg "
        f"max err={float(prediction['max_joint_error_deg'][step_index]):.3f} deg EE err={ee_text}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_video_from_frames(frame_paths: Sequence[Path], output_path: Path, *, fps: int = 12) -> None:
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=int(fps),
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    try:
        for path in frame_paths:
            writer.append_data(imageio.imread(path))
    finally:
        writer.close()


def cleanup_frame_paths(frame_paths: Sequence[Path]) -> None:
    touched_dirs: set[Path] = set()
    for path in frame_paths:
        touched_dirs.add(path.parent)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    for directory in sorted(touched_dirs, key=lambda item: len(item.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def generate_video(
    *,
    root: Path,
    prediction: dict[str, Any],
    episode: dict[str, Any],
    tier_b_available: bool,
    resume: bool,
    fps: int = 12,
    keep_frames: bool = False,
) -> dict[str, Any]:
    mode_dir = root / "videos" / SHORT_MODE[str(prediction["prediction_mode"])]
    video_path = mode_dir / str(prediction["model_family"]) / str(prediction["seed"]) / f"{prediction['seq_id']}.mp4"
    frames_dir = (
        root
        / "frames"
        / str(prediction["model_family"])
        / str(prediction["seed"])
        / str(prediction["prediction_mode"])
        / str(prediction["seq_id"])
    )
    expected_count = int(len(prediction["timestep"]))
    output_meta = {
        "path": str(video_path),
        "frames_dir": str(frames_dir),
        "frame_count": expected_count,
        "fps": int(fps),
        "tier_b_used": bool(tier_b_available),
        "frames_kept": bool(keep_frames),
    }
    meta_path = video_path.with_suffix(".json")
    if resume and video_path.exists() and meta_path.exists():
        old = read_json(meta_path)
        if int(old.get("frame_count", -1)) == expected_count:
            return {"status": "skipped", **old}

    renderer = None
    if tier_b_available:
        renderer = StaticPoseRenderer(episode, offscreen=True, width=512, height=512)
    frame_paths = []
    try:
        for index in range(expected_count):
            gt_render = renderer.render(prediction["q_target"][index]) if renderer is not None else None
            pred_render = renderer.render(prediction["q_prediction"][index]) if renderer is not None else None
            frame_path = frames_dir / f"frame_{index:05d}.png"
            title = (
                "Teacher-forced one-step prediction"
                if prediction["prediction_mode"] == "teacher_forced"
                else "Open-loop joint feedback; exogenous recorded images"
            )
            if renderer is None:
                render_fast_tier_a_frame(
                    prediction=prediction,
                    step_index=index,
                    output_path=frame_path,
                    title=title,
                )
            else:
                render_plot_frame(
                    prediction=prediction,
                    step_index=index,
                    gt_render=gt_render,
                    pred_render=pred_render,
                    output_path=frame_path,
                    title=title,
                )
            frame_paths.append(frame_path)
    finally:
        if renderer is not None:
            renderer.close()
    write_video_from_frames(frame_paths, video_path, fps=fps)
    if not keep_frames:
        cleanup_frame_paths(frame_paths)
    write_json(meta_path, output_meta)
    return {"status": "written", **output_meta}


def render_comparison_frame(
    *,
    predictions_by_model: dict[str, dict[str, Any]],
    step_index: int,
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = [model for model in MODEL_FAMILIES if model in predictions_by_model]
    reference = predictions_by_model[ordered[0]]
    image = image_chw_to_hwc(reference["images_uint8"][step_index])
    timesteps = reference["timestep"]
    current_t = int(timesteps[step_index])
    lo, hi = _joint_limits_from_prediction(reference)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], width_ratios=[1.05, 1.0, 1.0, 1.0])

    ax_img = fig.add_subplot(grid[0, 0])
    ax_img.imshow(image)
    ax_img.set_title("Input RGB I_t")
    ax_img.axis("off")

    ax_ee = fig.add_subplot(grid[1, 0])
    target_xy = reference.get("target_ee_xy_norm", reference["target_ee_pose"][:, :2])
    ax_ee.plot(target_xy[:, 0], target_xy[:, 1], lw=1.5, label="target EE")
    block_xy = reference["block_xy"]
    ax_ee.scatter(block_xy[:, 0], block_xy[:, 1], s=20, c="black", alpha=0.45, label="block centers")
    ax_ee.scatter(block_xy[step_index, 0], block_xy[step_index, 1], s=90, facecolors="none", edgecolors="red")
    ax_ee.text(
        0.01,
        0.01,
        "Target annotation is analysis-only; not model input.",
        transform=ax_ee.transAxes,
        fontsize=8,
        va="bottom",
    )
    ax_ee.set_title("Target EE path and block layout (normalized board XY)")
    ax_ee.set_aspect("equal", adjustable="box")
    ax_ee.grid(alpha=0.2)

    x = np.arange(1, 8)
    width = 0.34
    for col, model_family in enumerate(ordered, start=1):
        prediction = predictions_by_model[model_family]
        ax_joint = fig.add_subplot(grid[0, col])
        ax_joint.bar(x - width / 2, prediction["q_target"][step_index], width=width, label="target")
        ax_joint.bar(x + width / 2, prediction["q_prediction"][step_index], width=width, label="prediction")
        ax_joint.set_ylim(float(lo.min()), float(hi.max()))
        ax_joint.set_xticks(x)
        ax_joint.set_title(
            f"{model_family}\nMAE {float(prediction['mean_joint_mae_deg'][step_index]):.3f} deg"
        )
        ax_joint.set_xlabel("joint")
        ax_joint.set_ylabel("rad")
        ax_joint.grid(axis="y", alpha=0.2)
        if col == 1:
            ax_joint.legend(fontsize=8, loc="upper right")

        ax_err = fig.add_subplot(grid[1, col])
        ax_err.plot(prediction["timestep"], prediction["mean_joint_mae_deg"], lw=1.4)
        for boundary_t in prediction["timestep"][prediction["segment_boundary"].astype(bool)]:
            ax_err.axvline(int(boundary_t), color="k", lw=0.5, alpha=0.25)
        ax_err.axvline(current_t, color="black", lw=1.4)
        ax_err.set_title("Mean joint absolute error")
        ax_err.set_xlabel("canonical timestep")
        ax_err.set_ylabel("deg")
        ax_err.set_ylim(0.0, max(1e-6, float(np.max(prediction["mean_joint_mae_deg"])) * 1.15))
        ax_err.grid(alpha=0.2)

    fig.suptitle(
        f"{title}\nseq={reference['seq_id']} mode={reference['prediction_mode']} "
        f"length={int(reference['length'][step_index])} rank/block="
        f"{int(reference['rank'][step_index])}/{int(reference['block_id'][step_index])} t={current_t}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def generate_comparison_video(
    *,
    root: Path,
    predictions_by_model: dict[str, dict[str, Any]],
    mode: str,
    seq_id: str,
    resume: bool,
    fps: int = 12,
    keep_frames: bool = False,
) -> dict[str, Any]:
    ordered = [model for model in MODEL_FAMILIES if model in predictions_by_model]
    if len(ordered) < 2:
        return {"status": "skipped", "reason": "need at least two model predictions for comparison"}
    expected_count = min(int(len(predictions_by_model[model]["timestep"])) for model in ordered)
    video_path = root / "videos" / "comparisons" / SHORT_MODE[str(mode)] / f"{seq_id}.mp4"
    frames_dir = root / "frames" / "comparisons" / str(mode) / str(seq_id)
    output_meta = {
        "path": str(video_path),
        "frames_dir": str(frames_dir),
        "frame_count": expected_count,
        "fps": int(fps),
        "models": ordered,
        "mode": mode,
        "seq_id": seq_id,
        "frames_kept": bool(keep_frames),
    }
    meta_path = video_path.with_suffix(".json")
    if resume and video_path.exists() and meta_path.exists():
        old = read_json(meta_path)
        if int(old.get("frame_count", -1)) == expected_count and list(old.get("models", [])) == ordered:
            return {"status": "skipped", **old}

    frame_paths = []
    for index in range(expected_count):
        frame_path = frames_dir / f"frame_{index:05d}.png"
        render_fast_comparison_frame(
            predictions_by_model=predictions_by_model,
            step_index=index,
            output_path=frame_path,
            title=(
                "Synchronized teacher-forced one-step comparison"
                if mode == "teacher_forced"
                else "Synchronized autoregressive comparison with prerecorded images"
            ),
        )
        frame_paths.append(frame_path)
    write_video_from_frames(frame_paths, video_path, fps=fps)
    if not keep_frames:
        cleanup_frame_paths(frame_paths)
    write_json(meta_path, output_meta)
    return {"status": "written", **output_meta}


def generate_trajectory_montage(root: Path, prediction: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = (
        root
        / "montages"
        / "trajectories"
        / str(prediction["model_family"])
        / str(prediction["seed"])
        / str(prediction["prediction_mode"])
        / f"{prediction['seq_id']}.png"
    )
    out_pdf = out_png.with_suffix(".pdf")
    if resume and out_png.exists() and out_pdf.exists():
        return {"png": str(out_png), "pdf": str(out_pdf), "status": "skipped"}
    timesteps = prediction["timestep"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=140)
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    lo, hi = _joint_limits_from_prediction(prediction)
    for joint_index in range(7):
        axes[0].plot(timesteps, prediction["q_target"][:, joint_index], color=colors[joint_index], lw=1.4)
        axes[0].plot(timesteps, prediction["q_prediction"][:, joint_index], color=colors[joint_index], ls="--", lw=1.2)
    for boundary_t in timesteps[prediction["segment_boundary"].astype(bool)]:
        axes[0].axvline(int(boundary_t), color="k", lw=0.5, alpha=0.25)
    axes[0].set_title("Joint trajectories: target solid, prediction dashed")
    axes[0].set_xlabel("canonical timestep")
    axes[0].set_ylabel("joint angle (rad)")
    axes[0].set_ylim(float(lo.min()), float(hi.max()))
    axes[0].text(0.01, 0.02, "fixed Panda joint-limit envelope", transform=axes[0].transAxes, fontsize=8)
    axes[0].grid(alpha=0.2)
    target_xy = prediction.get("target_ee_xy_norm", prediction["target_ee_pose"][:, :2])
    axes[1].plot(target_xy[:, 0], target_xy[:, 1], lw=1.5, label="target EE")
    pred_xy = prediction.get("predicted_ee_xy_norm")
    if pred_xy is not None:
        axes[1].plot(pred_xy[:, 0], pred_xy[:, 1], lw=1.5, ls="--", label="pred EE")
    block_xy = prediction["block_xy"]
    axes[1].scatter(block_xy[:, 0], block_xy[:, 1], s=24, c="black", label="block centers")
    for idx, (x, y) in enumerate(block_xy):
        if idx == 0 or prediction["segment_boundary"][idx]:
            axes[1].text(float(x), float(y), str(int(prediction["rank"][idx])), fontsize=8)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title("Top-down EE path and block layout (normalized board XY)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle(
        f"{prediction['seq_id']} {prediction['model_family']} seed={prediction['seed']} {prediction['prediction_mode']}"
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    return {"png": str(out_png), "pdf": str(out_pdf)}


def generate_simple_contact_sheet(
    root: Path,
    prediction: dict[str, Any],
    *,
    name: str,
    indices: Sequence[int],
    resume: bool = False,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = [int(i) for i in indices if 0 <= int(i) < len(prediction["timestep"])]
    if not indices:
        indices = [0]
    out_png = (
        root
        / "montages"
        / name
        / str(prediction["model_family"])
        / str(prediction["seed"])
        / str(prediction["prediction_mode"])
        / f"{prediction['seq_id']}.png"
    )
    out_pdf = out_png.with_suffix(".pdf")
    if resume and out_png.exists() and out_pdf.exists():
        return {"png": str(out_png), "pdf": str(out_pdf), "status": "skipped"}
    fig, axes = plt.subplots(2, len(indices), figsize=(2.6 * len(indices), 5.2), dpi=140)
    if len(indices) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col, index in enumerate(indices):
        axes[0, col].imshow(image_chw_to_hwc(prediction["images_uint8"][index]))
        axes[0, col].set_title(f"t={int(prediction['timestep'][index])}")
        axes[0, col].axis("off")
        err = prediction["per_joint_abs_error_deg"][index]
        axes[1, col].bar(np.arange(1, 8), err)
        axes[1, col].set_ylim(0, max(1e-6, float(np.max(prediction["per_joint_abs_error_deg"])) * 1.1))
        axes[1, col].set_title(f"MAE {float(prediction['mean_joint_mae_deg'][index]):.2f} deg")
    fig.suptitle(f"{name}: {prediction['seq_id']} {prediction['model_family']} {prediction['prediction_mode']}")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    return {"png": str(out_png), "pdf": str(out_pdf)}


def generate_montages(root: Path, prediction: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    count = int(len(prediction["timestep"]))
    whole_indices = np.linspace(0, count - 1, min(10, count)).round().astype(int)
    endpoint_indices = []
    for segment_id in sorted(set(int(v) for v in prediction["segment_id"])):
        matches = np.where(prediction["segment_id"] == segment_id)[0]
        if matches.size:
            endpoint_indices.append(int(matches[-1]))
    progress_indices = []
    for segment_id in sorted(set(int(v) for v in prediction["segment_id"])):
        matches = np.where(prediction["segment_id"] == segment_id)[0]
        if matches.size:
            for fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
                progress_indices.append(int(matches[round((matches.size - 1) * fraction)]))
    return {
        "whole_episode": generate_simple_contact_sheet(
            root, prediction, name="whole_episode", indices=whole_indices, resume=resume
        ),
        "endpoints": generate_simple_contact_sheet(
            root, prediction, name="endpoints", indices=endpoint_indices, resume=resume
        ),
        "segment_progress": generate_simple_contact_sheet(
            root, prediction, name="segment_progress", indices=progress_indices, resume=resume
        ),
        "trajectories": generate_trajectory_montage(root, prediction, resume=resume),
    }


def write_audit(
    *,
    output_root: Path,
    report_path: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    checkpoints: dict[tuple[str, int], CheckpointRef],
    representative_seed: int,
) -> dict[str, Any]:
    raw_root = Path(str(config["raw_dataset_root"]))
    raw_manifest = read_json(raw_root / "manifest.json")
    first_sample = manifest["samples"][0]
    raw_meta = read_json(first_sample["raw_metadata_path"])
    raw_arrays = np.load(first_sample["raw_arrays_path"])
    joint_limit_metadata = resolve_joint_limit_metadata(load_episode(manifest, str(first_sample["seq_id"])), manifest)
    camera_settings = {
        "raw_camera_name": raw_meta.get("camera_name"),
        "raw_image_shape": raw_meta.get("image_shape"),
        "free_camera_config": {
            "lookat": [0.0, 0.0, 0.9],
            "distance": 0.489545,
            "azimuth": -179.858241,
            "elevation": -63.442729,
        },
        "render_resolution": [512, 512],
        "video_resolution": [1920, 1080],
    }
    arm_indexes = list(joint_limit_metadata.get("arm_qpos_indexes", list(range(7))))
    gripper_indexes = list(joint_limit_metadata.get("fixed_hand_qpos_indexes", list(range(7, 19))))
    if "fixed_hand_qpos" in joint_limit_metadata:
        hand_values = list(joint_limit_metadata["fixed_hand_qpos"])
    else:
        hand_values = raw_arrays["qpos"][0, gripper_indexes].astype(float).tolist()
    block_positions = raw_meta.get("block_positions", {})
    block_xyz = {key: value.get("xyz_world") for key, value in block_positions.items()}
    checkpoint_rows = [
        {
            "model_family": ref.model_family,
            "seed": ref.seed,
            "run_name": ref.run_name,
            "checkpoint_path": str(ref.checkpoint_path),
            "checkpoint_hash": ref.checkpoint_hash,
            "best_val_rmse": ref.best_val_rmse,
            "source": ref.source,
        }
        for ref in sorted(checkpoints.values(), key=lambda item: (item.model_family, item.seed))
    ]
    audit = {
        "schema_version": "corsi_prediction_visualization_audit_v1",
        "created_at_unix": time.time(),
        "config_path": "corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json",
        "canonical_root": manifest["canonical_root"],
        "raw_root": str(raw_root),
        "raw_manifest_schema": raw_manifest.get("schema_version"),
        "canonical_fingerprint": manifest["canonical_fingerprint"],
        "representative_visual_seed": representative_seed,
        "checkpoint_selection_rule": "posthoc final_checkpoint_selection.json if present; otherwise run summary best checkpoint; validation RMSE only",
        "checkpoints": checkpoint_rows,
        "joint_names": manifest["joint_names"],
        "arm_qpos_indexes": arm_indexes,
        "fixed_hand_qpos_indexes": gripper_indexes,
        "fixed_hand_qpos_values_from_first_raw_frame": hand_values,
        "joint_limit_source": joint_limit_metadata.get("source"),
        "joint_limits": joint_limit_metadata.get("joint_limits"),
        "renderer_mapping_metadata": joint_limit_metadata,
        "input_image_source": "canonical images copied from raw freecam RGB; source_frame_index/source_timestamp preserved per timestep",
        "camera_settings": camera_settings,
        "block_layout": block_xyz,
        "ee_plot_coordinate_policy": "FK validation uses raw world-frame target_ee_pose; videos and montages plot target_ee_xy_norm against block_xy in normalized board coordinates.",
        "block_geom_dimensions": {
            "half_size_xyz": [0.025, 0.025, 0.015],
            "source": "robosuite/models/arenas/corsi_table_arena.py CorsiTableArena.block_half_size",
        },
        "simulator_rerendering_reproducibility": "pending renderer_validation.json; pixel-perfect equality is not required",
    }
    write_json(output_root / "audit" / "visualization_audit.json", audit)
    lines = [
        "# Corsi Prediction Visualization Audit",
        "",
        f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Scope",
        "",
        "Post-hoc visualization of already-trained causal 7-joint motion prediction baselines.",
        "No retraining, dataset regeneration, checkpoint mutation, or metric changes are performed.",
        "",
        "## Checkpoint Selection",
        "",
        "Rule: use `posthoc_convergence_accuracy_v1/convergence/final_checkpoint_selection.json` when present; otherwise use each run summary best checkpoint. The selection key is validation RMSE only.",
        "",
        "| Model | Seed | Source | Best val RMSE | Checkpoint SHA256 | Path |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    for row in checkpoint_rows:
        lines.append(
            f"| {row['model_family']} | {row['seed']} | {row['source']} | {row['best_val_rmse']:.6f} | "
            f"`{row['checkpoint_hash'][:16]}...` | `{row['checkpoint_path']}` |"
        )
    lines.extend(
        [
            "",
            f"Representative visual seed: `{representative_seed}` (median validation RMSE among final visual checkpoints).",
            "",
            "## Joint And Qpos Mapping",
            "",
            f"Joint order: `{manifest['joint_names']}`",
            "",
            f"Arm qpos indexes: `{arm_indexes}`",
            "",
            f"Fixed hand qpos indexes: `{gripper_indexes}`",
            "",
            f"Fixed hand qpos values from first raw frame: `{[round(v, 6) for v in hand_values]}`",
            "",
            f"Joint limit source: `{joint_limit_metadata.get('source')}`",
            "",
            f"Joint limits: `{joint_limit_metadata.get('joint_limits')}`",
            "",
            "## Image And Camera",
            "",
            f"Input image source: raw/canonical `{raw_meta.get('camera_name')}` RGB, shape `{raw_meta.get('image_shape')}`.",
            "",
            f"Free camera config: `{camera_settings['free_camera_config']}`",
            "",
            "## Blocks And Geometry",
            "",
            "Block positions are restored from raw episode metadata. Corsi block geom half-size is recorded as `[0.025, 0.025, 0.015]` m.",
            "",
            "EE plots use normalized board XY for both `target_ee_xy_norm` and `block_xy`; FK validation separately uses world-frame `target_ee_pose`.",
            "",
            "## Tier Availability",
            "",
            "Tier A is always available. Tier B is enabled only if `visualizations_v1/audit/renderer_validation.json` passes FK thresholds.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


def parse_csv(text: str) -> list[str]:
    return [token.strip() for token in text.split(",") if token.strip()]


def resolve_visualization_device(requested: str) -> tuple[torch.device, dict[str, Any]]:
    cuda_available = bool(torch.cuda.is_available())
    info = {
        "requested": requested,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": cuda_available,
    }
    if requested.startswith("cuda") and not cuda_available:
        device = torch.device("cpu")
        info["fallback_reason"] = f"requested {requested}, but torch.cuda.is_available() is False"
    else:
        device = resolve_device(requested)
    info["resolved"] = str(device)
    return device, info


def should_generate_path(path: Path, *, resume: bool) -> bool:
    return not (resume and path.exists())


def run_visualization(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    config = load_config(args.config)
    output_root = Path(args.output_root)
    ensure_output_tree(output_root)
    manifest_path = Path(str(config["canonical_root"])) / "manifest.json"
    manifest = read_json(manifest_path)
    checkpoints = resolve_checkpoints(config)
    representative_seed = int(args.seed) if args.seed is not None else select_representative_seed(checkpoints)
    selected_models = parse_csv(args.models)
    selected_modes = parse_csv(args.modes)
    device, device_info = resolve_visualization_device(args.device)
    write_json(output_root / "audit" / "device.json", device_info)
    report_audit_path = Path("reports/corsi_prediction_visualization_audit.md")
    write_audit(
        output_root=output_root,
        report_path=report_audit_path,
        config=config,
        manifest=manifest,
        checkpoints=checkpoints,
        representative_seed=representative_seed,
    )

    test_ids = list(manifest["split"]["test"])
    if args.seq_id:
        seq_ids = [args.seq_id]
    else:
        seq_ids = test_ids
    if args.max_episodes > 0:
        seq_ids = seq_ids[: int(args.max_episodes)]

    generated_rows = []
    failures_path = output_root / "manifests" / "failures.jsonl"
    if failures_path.exists():
        failures_path.unlink()

    # First pass: teacher-forced test predictions for representative visual/joint and persistence.
    metrics_visual: dict[str, dict[str, Any]] = {}
    metrics_joint: dict[str, dict[str, Any]] = {}
    cache: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for seq_id in test_ids if not args.seq_id else seq_ids:
        episode = load_episode(manifest, seq_id)
        for model_family in ["visual_joint", "joint_only", "persistence"]:
            seed = -1 if model_family == "persistence" else representative_seed
            checkpoint = None if model_family == "persistence" else checkpoints[(model_family, seed)]
            try:
                prediction = predict_sequence(
                    model_family=model_family,
                    mode="teacher_forced",
                    checkpoint=checkpoint,
                    episode=episode,
                    manifest=manifest,
                    device=device,
                )
                cache[(seq_id, model_family, seed, "teacher_forced")] = prediction
                if model_family == "visual_joint":
                    metrics_visual[seq_id] = episode_metrics(prediction)
                elif model_family == "joint_only":
                    metrics_joint[seq_id] = episode_metrics(prediction)
            except Exception as exc:
                append_jsonl(
                    failures_path,
                    {
                        "phase": "selection_prediction",
                        "seq_id": seq_id,
                        "model_family": model_family,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
    if args.seq_id:
        selection = [
            {
                "category": "single_episode",
                "seq_id": args.seq_id,
                "length": int(load_episode(manifest, args.seq_id)["sample"]["length"]),
                "tie_break": "explicit --seq-id",
            }
        ]
    else:
        selection = select_episodes(visual_metrics=metrics_visual, joint_metrics=metrics_joint)
    selected_seq_ids = unique_selected_seq_ids(selection)
    selection_manifest = {
        "schema_version": "corsi_selected_prediction_episodes_v1",
        "representative_visual_seed": representative_seed,
        "representative_seed_rule": "median validation RMSE among final visual_joint checkpoints",
        "checkpoint_selection_rule": "validation-selected final checkpoint only; test metrics used only for descriptive episode selection",
        "tie_breaking": "lexicographic seq_id after category metric",
        "selected": selection,
        "unique_seq_ids": selected_seq_ids,
    }
    write_json(output_root / "manifests" / "selected_episodes.json", selection_manifest)

    selected_episodes = [load_episode(manifest, seq_id) for seq_id in selected_seq_ids]
    joint_limit_metadata = (
        resolve_joint_limit_metadata(selected_episodes[0], manifest)
        if selected_episodes
        else fallback_joint_limit_metadata(manifest["joint_names"])
    )
    write_json(output_root / "audit" / "joint_limit_metadata.json", joint_limit_metadata)
    selected_visual_predictions = []
    for episode in selected_episodes:
        seq_id = str(episode["sample"]["seq_id"])
        prediction = cache.get((seq_id, "visual_joint", representative_seed, "teacher_forced"))
        if prediction is None:
            prediction = predict_sequence(
                model_family="visual_joint",
                mode="teacher_forced",
                checkpoint=checkpoints[("visual_joint", representative_seed)],
                episode=episode,
                manifest=manifest,
                device=device,
            )
        prediction = attach_joint_limit_metadata(prediction, joint_limit_metadata)
        selected_visual_predictions.append(prediction)
    renderer_validation = validate_fk_for_predictions(selected_episodes, selected_visual_predictions)
    write_json(output_root / "audit" / "renderer_validation.json", renderer_validation)
    tier_b_available = bool(renderer_validation.get("tier_b_available"))

    for episode in selected_episodes:
        seq_id = str(episode["sample"]["seq_id"])
        episode_prediction_cache: dict[tuple[str, str], dict[str, Any]] = {}
        for model_family in selected_models:
            if model_family not in MODEL_FAMILIES:
                raise ValueError(f"unknown model: {model_family}")
            seed = -1 if model_family == "persistence" else representative_seed
            if args.model and model_family != args.model:
                continue
            if args.seed is not None and model_family != "persistence" and seed != int(args.seed):
                continue
            checkpoint = None if model_family == "persistence" else checkpoints[(model_family, seed)]
            for mode in selected_modes:
                try:
                    prediction = cache.get((seq_id, model_family, seed, mode))
                    if prediction is None:
                        prediction = predict_sequence(
                            model_family=model_family,
                            mode=mode,
                            checkpoint=checkpoint,
                            episode=episode,
                            manifest=manifest,
                            device=device,
                        )
                    prediction = attach_joint_limit_metadata(prediction, joint_limit_metadata)
                    if tier_b_available:
                        prediction = attach_fk_ee(prediction, episode)
                    episode_prediction_cache[(model_family, mode)] = prediction
                    export_result = write_prediction_export(
                        output_root,
                        prediction,
                        manifest,
                        resume=bool(args.resume),
                        extra_metadata={
                            "tier_b_available": tier_b_available,
                            "joint_limit_source": joint_limit_metadata.get("source"),
                            "joint_limits": joint_limit_metadata.get("joint_limits"),
                        },
                    )
                    generated_rows.append({"type": "prediction", **export_result})
                    if args.export_video:
                        video_result = generate_video(
                            root=output_root,
                            prediction=prediction,
                            episode=episode,
                            tier_b_available=tier_b_available,
                            resume=bool(args.resume),
                            fps=int(args.fps),
                            keep_frames=bool(args.keep_frames),
                        )
                        generated_rows.append(
                            {
                                "type": "video",
                                "seq_id": seq_id,
                                "model_family": model_family,
                                "seed": seed,
                                "mode": mode,
                                **video_result,
                            }
                        )
                    if args.export_montage:
                        montage_result = generate_montages(output_root, prediction, resume=bool(args.resume))
                        generated_rows.append(
                            {
                                "type": "montage",
                                "seq_id": seq_id,
                                "model_family": model_family,
                                "seed": seed,
                                "mode": mode,
                                "paths": montage_result,
                            }
                        )
                except Exception as exc:
                    append_jsonl(
                        failures_path,
                        {
                            "phase": "generation",
                            "seq_id": seq_id,
                            "model_family": model_family,
                            "seed": seed,
                            "mode": mode,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    if not args.continue_on_failure:
                        raise
        if args.export_video or args.export_comparison_video:
            for mode in selected_modes:
                predictions_by_model = {
                    model_family: episode_prediction_cache[(model_family, mode)]
                    for model_family in selected_models
                    if (model_family, mode) in episode_prediction_cache
                }
                if len(predictions_by_model) < 2:
                    continue
                try:
                    comparison_result = generate_comparison_video(
                        root=output_root,
                        predictions_by_model=predictions_by_model,
                        mode=mode,
                        seq_id=seq_id,
                        resume=bool(args.resume),
                        fps=int(args.fps),
                        keep_frames=bool(args.keep_frames),
                    )
                    generated_rows.append(
                        {
                            "type": "comparison_video",
                            "seq_id": seq_id,
                            "mode": mode,
                            **comparison_result,
                        }
                    )
                except Exception as exc:
                    append_jsonl(
                        failures_path,
                        {
                            "phase": "comparison_video",
                            "seq_id": seq_id,
                            "mode": mode,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    if not args.continue_on_failure:
                        raise

    generated_path = output_root / "manifests" / "generated_outputs.jsonl"
    if generated_path.exists():
        generated_path.unlink()
    for row in generated_rows:
        append_jsonl(generated_path, row)
    report = write_final_report(
        output_root=output_root,
        report_path=Path("reports/corsi_prediction_visualization_report.md"),
        audit_path=report_audit_path,
        renderer_validation=renderer_validation,
        selection_manifest=selection_manifest,
        generated_rows=generated_rows,
        runtime_seconds=time.time() - started,
        device_info=device_info,
        args=args,
    )
    return {
        "output_root": str(output_root),
        "selected_seq_ids": selected_seq_ids,
        "tier_b_available": tier_b_available,
        "generated_count": len(generated_rows),
        "report": report,
        "runtime_seconds": time.time() - started,
    }


def write_final_report(
    *,
    output_root: Path,
    report_path: Path,
    audit_path: Path,
    renderer_validation: dict[str, Any],
    selection_manifest: dict[str, Any],
    generated_rows: Sequence[dict[str, Any]],
    runtime_seconds: float,
    device_info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    video_rows = [row for row in generated_rows if row.get("type") == "video"]
    comparison_video_rows = [row for row in generated_rows if row.get("type") == "comparison_video"]
    montage_rows = [row for row in generated_rows if row.get("type") == "montage"]
    prediction_rows = [row for row in generated_rows if row.get("type") == "prediction"]
    failure_path = output_root / "manifests" / "failures.jsonl"
    failed_count = 0
    if failure_path.exists():
        failed_count = sum(1 for _ in failure_path.open("r", encoding="utf-8"))
    storage_bytes = 0
    for path in output_root.rglob("*"):
        if path.is_file():
            storage_bytes += path.stat().st_size
    lines = [
        "# Corsi Prediction Visualization Report",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        INTERPRETATION_STATEMENT,
        "",
        "## Checkpoints And Selection",
        "",
        "Checkpoints are selected from validation RMSE only, using the post-hoc final selection file when present.",
        f"Representative visual seed: `{selection_manifest['representative_visual_seed']}`.",
        "",
            f"Audit: `{audit_path}`",
            f"Visualization audit JSON: `{output_root / 'audit' / 'visualization_audit.json'}`",
            f"Device audit JSON: `{output_root / 'audit' / 'device.json'}`",
            f"Requested device: `{device_info.get('requested')}`; resolved device: `{device_info.get('resolved')}`.",
            f"Torch CUDA available in this process: `{device_info.get('torch_cuda_available')}`.",
            (
                f"Device fallback reason: `{device_info.get('fallback_reason')}`."
                if device_info.get("fallback_reason")
                else "Device fallback reason: `none`."
            ),
            "",
            "## Selected Episodes",
        "",
        "| Category | Seq id | Length | Reason |",
        "| --- | --- | ---: | --- |",
    ]
    for row in selection_manifest["selected"]:
        lines.append(
            f"| {row['category']} | `{row['seq_id']}` | {int(row['length'])} | {row.get('tie_break', '')} |"
        )
    lines.extend(
        [
            "",
            "## Prediction Modes",
            "",
            "- `teacher_forced`: input at display index `t` is recorded `I_t, q_t`; target is `q_{t+1}`.",
            "- `open_loop_joint_feedback_with_exogenous_images`: starts from recorded `q_0`; after that, predicted joints are fed back while recorded images `I_t` remain exogenous.",
            "",
            "Neither mode is a closed-loop robot rollout.",
            "",
            "## EE Plot Coordinates",
            "",
            "`target_ee_pose` remains the raw world-frame EE pose used for FK validation. Video EE plots use `target_ee_xy_norm`, converted from raw table XY to the same normalized board coordinates as `block_xy`.",
            "",
            "## Tier Availability",
            "",
            f"Tier A: available.",
            f"Tier B: {'available' if renderer_validation.get('tier_b_available') else 'unavailable'}.",
            f"Renderer validation: `{output_root / 'audit' / 'renderer_validation.json'}`",
            f"Median FK error: `{renderer_validation.get('median_error_m')}` m.",
            f"Max FK error: `{renderer_validation.get('max_error_m')}` m.",
            f"Tier-B reason: `{renderer_validation.get('reason')}`.",
            "When Tier B is unavailable, videos retain Tier-A source RGB and numeric joint/EE plots only.",
            "",
            "## Generated Outputs",
            "",
            f"Prediction exports: {len(prediction_rows)}",
            f"Per-model videos generated/skipped: {len(video_rows)}",
            f"Comparison videos generated/skipped: {len(comparison_video_rows)}",
            f"Montage groups generated: {len(montage_rows)}",
            f"Failures: {failed_count}",
            "",
            f"Output root: `{output_root}`",
            f"Selected episodes manifest: `{output_root / 'manifests' / 'selected_episodes.json'}`",
            f"Generated outputs manifest: `{output_root / 'manifests' / 'generated_outputs.jsonl'}`",
            f"Failures manifest: `{failure_path}`",
            "",
            "## Reproduction",
            "",
            "```bash",
            "conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.visualize_predictions \\",
            f"  --config {args.config} \\",
            "  --checkpoint-source final-validation-selected \\",
            f"  --modes {args.modes} \\",
            f"  --models {args.models} \\",
            "  --select median-per-length,best,worst,worst-boundary,worst-endpoint,visual-best-gain,visual-worst-gap \\",
            "  --export-video \\",
            "  --export-comparison-video \\",
            "  --export-montage \\",
            "  --resume",
            "```",
            "",
            "## Completion Summary",
            "",
            "- files changed: `visualize_predictions.py`, visualization tests/report outputs",
            f"- tests passed: see validation command output in the assistant summary",
            f"- per-model videos generated/skipped: {len(video_rows)}",
            f"- comparison videos generated/skipped: {len(comparison_video_rows)}",
            f"- montages generated: {len(montage_rows)} groups",
            f"- failed outputs: {failed_count}",
            f"- total runtime: {runtime_seconds:.1f} seconds",
            f"- total storage: {storage_bytes / (1024 ** 2):.1f} MiB",
            "",
            "Exact resume command:",
            "",
            "```bash",
            f"conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.visualize_predictions --config {args.config} --models {args.models} --modes {args.modes} --export-video --export-comparison-video --export-montage --resume",
            "```",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "path": str(report_path),
        "videos": len(video_rows),
        "comparison_videos": len(comparison_video_rows),
        "montages": len(montage_rows),
        "failures": failed_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export and visualize Corsi 7-joint motion predictions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="corsi_artifacts/motion_baseline/visualizations_v1")
    parser.add_argument("--checkpoint-source", default="final-validation-selected")
    parser.add_argument("--modes", default="teacher_forced,open_loop_joint_feedback_with_exogenous_images")
    parser.add_argument("--models", default="visual_joint,joint_only,persistence")
    parser.add_argument("--select", default="median-per-length,best,worst,worst-boundary,worst-endpoint,visual-best-gain,visual-worst-gap")
    parser.add_argument("--seq-id", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--export-video", action="store_true")
    parser.add_argument("--export-comparison-video", action="store_true")
    parser.add_argument("--export-montage", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true", default=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_visualization(args)
    print(json.dumps(jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
