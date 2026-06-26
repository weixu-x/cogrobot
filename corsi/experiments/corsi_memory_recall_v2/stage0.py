"""Stage 0 identifiability and upper-bound gates for Corsi memory-recall V2.

The diagnostics here are intentionally cheap and isolated from Stage 1/2 full
training.  They either use the existing robosuite pointing helpers or small
synthetic/oracle batches that exercise a specific V2 subsystem.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from corsi.experiments.corsi_memory_recall_v2.model import (
    CorsiMemoryRecallV2Config,
    CorsiMemoryRecallV2Model,
)
from corsi.experiments.corsi_memory_recall_v2.train import load_config, set_seed


DEFAULT_CONFIG_PATH = Path("corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json")
DEFAULT_OUTPUT_ROOT = Path("corsi_artifacts/memory_recall_v2/stage0")
SUMMARY_NAME = "summary.json"


@dataclass
class GateResult:
    gate: str
    passed: bool
    thresholds: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, str]
    notes: list[str]
    elapsed_sec: float


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def stage0_audit() -> dict[str, Any]:
    """Return a small machine-readable audit of exact Stage 0 coverage."""

    module_path = Path(__file__)
    return {
        "exact_stage0_module": str(module_path),
        "implemented_by_this_module": [
            "gate1_motor_shell_single_block_reachability",
            "gate2_oracle_recall_upper_bound_len9",
            "gate3_frozen_linear_probe_from_M",
            "gate4_cnn_visual_motor_tiny_overfit",
        ],
        "preexisting_exact_gate_implementation_found": False,
        "missing_before_stage0_module": [
            "No exact Motor Shell reachability gate existed for V2.",
            "No oracle/teacher-forced Recall LSTM upper-bound gate existed.",
            "No frozen linear probe from final_memory/memory trace existed.",
            "No isolated CNN/Visual/Motor tiny-overfit gate existed.",
        ],
        "M_definition": (
            "M is CorsiMemoryRecallV2Model.run_memory(...)[\"final_memory\"], the final hidden "
            "state of the V2 memory LSTM after all presentation segments. Memory traces are "
            "available as run_memory(..., return_traces=True)[\"traces\"][\"h_t\"], but Gate 3 "
            "uses final_memory for the stricter sequence-level probe."
        ),
    }


def _model_config_from_experiment(config: Mapping[str, Any], **overrides: Any) -> CorsiMemoryRecallV2Config:
    values: dict[str, Any] = {
        "max_sequence_length": int(config.get("max_sequence_length", 9)),
        "num_blocks": int(config.get("num_blocks", 9)),
        "eos_token_id": int(config.get("eos_token_id", 9)),
        "ignore_index": int(config.get("ignore_index", -100)),
        "k_samples_per_segment": int(config.get("k_samples_per_segment", 12)),
    }
    if "joint_names" in config:
        values["joint_dim"] = len(config["joint_names"])
    values.update(overrides)
    return CorsiMemoryRecallV2Config(**values)


def _no_repeat_sequences(
    *,
    count: int,
    length: int,
    num_blocks: int,
    seed: int,
) -> torch.Tensor:
    rng = random.Random(int(seed))
    selected: set[tuple[int, ...]] = set()
    max_unique = math.prod(range(num_blocks, num_blocks - length, -1))
    if count > max_unique:
        raise ValueError(f"requested {count} no-repeat sequences but only {max_unique} are possible")
    while len(selected) < int(count):
        selected.add(tuple(rng.sample(range(int(num_blocks)), int(length))))
    ordered = sorted(selected)
    return torch.tensor(ordered, dtype=torch.long)


def _append_eos(sequences: Tensor, eos_token_id: int) -> Tensor:
    eos = torch.full((sequences.shape[0], 1), int(eos_token_id), dtype=torch.long, device=sequences.device)
    return torch.cat([sequences, eos], dim=1)


class OracleRecallDiagnostic(nn.Module):
    """Teacher-forced oracle path through the V2 Recall LSTM and block head."""

    def __init__(self, cfg: CorsiMemoryRecallV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.oracle_embedding = nn.Embedding(cfg.num_tokens, cfg.recall_token_dim)
        self.recall_lstm = CorsiMemoryRecallV2Model(cfg).recall_lstm
        self.block_head = nn.Linear(cfg.recall_hidden_dim, cfg.num_tokens)
        self.initial_h = nn.Parameter(torch.zeros(cfg.recall_hidden_dim))
        self.initial_c = nn.Parameter(torch.zeros(cfg.recall_hidden_dim))

    def forward(self, oracle_tokens: Tensor) -> Tensor:
        batch, steps = oracle_tokens.shape
        h = self.initial_h.view(1, -1).expand(batch, -1)
        c = self.initial_c.view(1, -1).expand(batch, -1)
        inputs = self.oracle_embedding(oracle_tokens)
        logits: list[Tensor] = []
        for step in range(steps):
            h, c, _ = self.recall_lstm(inputs[:, step], (h, c))
            logits.append(self.block_head(h))
        return torch.stack(logits, dim=1)


def run_gate2_oracle_recall(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    seed: int = 2026062402,
    sequence_count: int = 96,
    max_steps: int = 700,
    learning_rate: float = 5e-3,
    exact_accuracy_threshold: float = 1.0,
) -> GateResult:
    started = time.time()
    set_seed(seed)
    cfg = _model_config_from_experiment(
        config,
        cnn_dim=16,
        visual_hidden_dim=16,
        motor_hidden_dim=16,
        item_dim=32,
        memory_dim=32,
        recall_hidden_dim=64,
        recall_token_dim=32,
    )
    length = int(cfg.max_sequence_length)
    targets = _append_eos(
        _no_repeat_sequences(count=sequence_count, length=length, num_blocks=cfg.num_blocks, seed=seed),
        cfg.eos_token_id,
    ).to(device)
    model = OracleRecallDiagnostic(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.0)
    history: list[dict[str, float]] = []
    best_exact = 0.0
    for step in range(int(max_steps)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(targets)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step == int(max_steps) - 1:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                token_acc = float(pred.eq(targets).float().mean().item())
                exact = float(pred.eq(targets).all(dim=1).float().mean().item())
            best_exact = max(best_exact, exact)
            history.append({"step": float(step), "loss": float(loss.item()), "token_accuracy": token_acc, "exact_sequence_accuracy": exact})
            if exact >= exact_accuracy_threshold:
                break
    model.eval()
    with torch.no_grad():
        logits = model(targets)
        pred = logits.argmax(dim=-1)
        token_acc = float(pred.eq(targets).float().mean().item())
        exact = float(pred.eq(targets).all(dim=1).float().mean().item())
        final_loss = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).item())

    artifact_path = output_root / "gate2_oracle_recall.pt"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg), "history": history}, artifact_path)
    thresholds = {
        "sequence_length": length,
        "sequence_count": int(sequence_count),
        "max_steps": int(max_steps),
        "exact_sequence_accuracy_min": float(exact_accuracy_threshold),
        "oracle_path": "target token embedding at each recall step -> V2 InstrumentedLSTMCell -> block_head",
    }
    metrics = {
        "final_loss": final_loss,
        "token_accuracy": token_acc,
        "exact_sequence_accuracy": exact,
        "best_exact_sequence_accuracy": best_exact,
        "steps_run": int(history[-1]["step"]) + 1 if history else int(max_steps),
    }
    return GateResult(
        gate="gate2_oracle_recall_upper_bound_len9",
        passed=bool(exact >= exact_accuracy_threshold),
        thresholds=thresholds,
        metrics=metrics,
        artifacts={"checkpoint": str(artifact_path)},
        notes=[
            "Diagnostic-only oracle recall path; it does not validate autonomous recall from the learned constant token.",
            "This isolates Recall LSTM/head capacity from vision and presentation memory.",
        ],
        elapsed_sec=float(time.time() - started),
    )


class OracleMemoryEncoder(nn.Module):
    """Cheap diagnostic memory encoder using oracle block embeddings as item embeddings."""

    def __init__(self, cfg: CorsiMemoryRecallV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.block_embedding = nn.Embedding(cfg.num_blocks, cfg.item_dim)
        self.memory_lstm = CorsiMemoryRecallV2Model(cfg).memory_lstm

    def forward(self, block_sequences: Tensor, *, return_traces: bool = False) -> dict[str, Tensor | dict[str, Tensor]]:
        batch, length = block_sequences.shape
        dtype = self.block_embedding.weight.dtype
        device = block_sequences.device
        item_embeddings = self.block_embedding(block_sequences)
        h = torch.zeros(batch, self.cfg.memory_dim, dtype=dtype, device=device)
        c = torch.zeros_like(h)
        trace: list[Tensor] = []
        for step in range(length):
            h, c, _ = self.memory_lstm(item_embeddings[:, step], (h, c))
            if return_traces:
                trace.append(h)
        result: dict[str, Tensor | dict[str, Tensor]] = {"final_memory": h}
        if return_traces:
            result["traces"] = {"h_t": torch.stack(trace, dim=1)}
        return result


class MultiPositionProbe(nn.Module):
    def __init__(self, memory_dim: int, positions: int, num_blocks: int) -> None:
        super().__init__()
        self.positions = int(positions)
        self.num_blocks = int(num_blocks)
        self.linear = nn.Linear(int(memory_dim), int(positions) * int(num_blocks))

    def forward(self, memory: Tensor) -> Tensor:
        return self.linear(memory).reshape(memory.shape[0], self.positions, self.num_blocks)


def _probe_metrics(logits: Tensor, targets: Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    return {
        "token_order_accuracy": float(pred.eq(targets).float().mean().item()),
        "length9_exact_sequence_accuracy": float(pred.eq(targets).all(dim=1).float().mean().item()),
    }


def run_gate3_frozen_probe(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    seed: int = 2026062403,
    sequence_count: int = 128,
    encoder_steps: int = 900,
    probe_steps: int = 600,
    learning_rate: float = 3e-3,
    token_accuracy_threshold: float = 0.99,
    exact_accuracy_threshold: float = 0.95,
) -> GateResult:
    started = time.time()
    set_seed(seed)
    cfg = _model_config_from_experiment(
        config,
        cnn_dim=16,
        visual_hidden_dim=16,
        motor_hidden_dim=16,
        item_dim=64,
        memory_dim=64,
        recall_hidden_dim=64,
        recall_token_dim=32,
    )
    length = int(cfg.max_sequence_length)
    sequences = _no_repeat_sequences(count=sequence_count, length=length, num_blocks=cfg.num_blocks, seed=seed).to(device)
    encoder = OracleMemoryEncoder(cfg).to(device)
    temporary_decoder = MultiPositionProbe(cfg.memory_dim, length, cfg.num_blocks).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(temporary_decoder.parameters()),
        lr=float(learning_rate),
        weight_decay=0.0,
    )
    encoder_history: list[dict[str, float]] = []
    for step in range(int(encoder_steps)):
        encoder.train()
        temporary_decoder.train()
        optimizer.zero_grad(set_to_none=True)
        memory = encoder(sequences)["final_memory"]
        assert torch.is_tensor(memory)
        logits = temporary_decoder(memory)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), sequences.reshape(-1))
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == int(encoder_steps) - 1:
            metrics = _probe_metrics(logits.detach(), sequences)
            encoder_history.append({"step": float(step), "loss": float(loss.item()), **metrics})
            if metrics["length9_exact_sequence_accuracy"] >= 1.0:
                break

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    with torch.no_grad():
        frozen_memory = encoder(sequences)["final_memory"]
        assert torch.is_tensor(frozen_memory)
        frozen_memory = frozen_memory.detach()

    probe = MultiPositionProbe(cfg.memory_dim, length, cfg.num_blocks).to(device)
    probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=float(learning_rate), weight_decay=0.0)
    probe_history: list[dict[str, float]] = []
    for step in range(int(probe_steps)):
        probe.train()
        probe_optimizer.zero_grad(set_to_none=True)
        logits = probe(frozen_memory)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), sequences.reshape(-1))
        loss.backward()
        probe_optimizer.step()
        if step % 50 == 0 or step == int(probe_steps) - 1:
            metrics = _probe_metrics(logits.detach(), sequences)
            probe_history.append({"step": float(step), "loss": float(loss.item()), **metrics})
            if (
                metrics["token_order_accuracy"] >= token_accuracy_threshold
                and metrics["length9_exact_sequence_accuracy"] >= exact_accuracy_threshold
            ):
                break

    probe.eval()
    with torch.no_grad():
        logits = probe(frozen_memory)
        final_metrics = _probe_metrics(logits, sequences)
        final_loss = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), sequences.reshape(-1)).item())

    checkpoint_path = output_root / "gate3_oracle_memory_probe.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "probe_state_dict": probe.state_dict(),
            "config": asdict(cfg),
            "encoder_history": encoder_history,
            "probe_history": probe_history,
            "M_definition": "final_memory",
        },
        checkpoint_path,
    )
    thresholds = {
        "M": "final_memory",
        "sequence_length": length,
        "sequence_count": int(sequence_count),
        "encoder_max_steps": int(encoder_steps),
        "probe_max_steps": int(probe_steps),
        "token_order_accuracy_min": float(token_accuracy_threshold),
        "length9_exact_sequence_accuracy_min": float(exact_accuracy_threshold),
        "frozen_weights_rule": "After the diagnostic oracle-memory checkpoint is built, encoder weights are frozen and only the linear probe is trained.",
    }
    metrics = {
        "final_probe_loss": final_loss,
        **final_metrics,
        "encoder_steps_run": int(encoder_history[-1]["step"]) + 1 if encoder_history else int(encoder_steps),
        "probe_steps_run": int(probe_history[-1]["step"]) + 1 if probe_history else int(probe_steps),
    }
    return GateResult(
        gate="gate3_frozen_linear_probe_from_M",
        passed=bool(
            final_metrics["token_order_accuracy"] >= token_accuracy_threshold
            and final_metrics["length9_exact_sequence_accuracy"] >= exact_accuracy_threshold
        ),
        thresholds=thresholds,
        metrics=metrics,
        artifacts={"checkpoint": str(checkpoint_path)},
        notes=[
            "Uses a Stage 0-generated oracle item-memory checkpoint because no Stage 0 or reviewed V2 checkpoint exists in this worktree.",
            "Passing proves final_memory can be made linearly readable for ordered block identity under oracle item embeddings; it does not prove RGB presentation encodes block identity.",
        ],
        elapsed_sec=float(time.time() - started),
    )


def run_gate4_tiny_overfit(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    seed: int = 2026062404,
    max_steps: int = 500,
    learning_rate: float = 3e-3,
    loss_threshold: float = 2e-3,
) -> GateResult:
    started = time.time()
    set_seed(seed)
    cfg = _model_config_from_experiment(
        config,
        image_size=32,
        k_samples_per_segment=3,
        max_sequence_length=2,
        cnn_dim=16,
        visual_hidden_dim=16,
        motor_hidden_dim=16,
        item_dim=16,
        memory_dim=16,
        recall_hidden_dim=16,
        recall_token_dim=8,
    )
    batch_size = 2
    segments = 2
    frames = int(cfg.k_samples_per_segment)
    images = torch.rand(batch_size, segments, frames, 3, cfg.image_size, cfg.image_size, device=device)
    segment_mask = torch.ones(batch_size, segments, dtype=torch.bool, device=device)
    frame_mask = torch.ones(batch_size, segments, frames, dtype=torch.bool, device=device)
    joint_target = torch.randn(batch_size, segments, frames, cfg.joint_dim, device=device) * 0.4
    ee_xy_target = torch.randn(batch_size, segments, frames, cfg.ee_xy_dim, device=device) * 0.4

    model = CorsiMemoryRecallV2Model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.0)
    history: list[dict[str, float]] = []
    final_loss = float("inf")
    for step in range(int(max_steps)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model.encode_presentation(images=images, segment_mask=segment_mask, frame_mask=frame_mask)
        pose_loss = F.mse_loss(outputs["pred_joint"], joint_target)
        ee_loss = F.mse_loss(outputs["pred_ee_xy"], ee_xy_target)
        loss = pose_loss + 0.5 * ee_loss
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if step % 25 == 0 or step == int(max_steps) - 1:
            history.append(
                {
                    "step": float(step),
                    "loss": final_loss,
                    "pose_loss": float(pose_loss.item()),
                    "ee_loss": float(ee_loss.item()),
                }
            )
            if final_loss <= loss_threshold:
                break

    checkpoint_path = output_root / "gate4_visual_motor_tiny_overfit.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg), "history": history}, checkpoint_path)
    thresholds = {
        "loss": "L_pose + 0.5 L_EE, where L_pose is joint MSE and L_EE is EE-XY MSE",
        "loss_max": float(loss_threshold),
        "max_steps": int(max_steps),
        "batch_size": batch_size,
        "segments": segments,
        "frames_per_segment": frames,
        "inputs": "RGB plus segment/frame masks only; joint and EE are supervision only.",
    }
    metrics = {
        "final_loss": final_loss,
        "steps_run": int(history[-1]["step"]) + 1 if history else int(max_steps),
        "initial_loss": float(history[0]["loss"]) if history else None,
    }
    return GateResult(
        gate="gate4_cnn_visual_motor_tiny_overfit",
        passed=bool(final_loss <= loss_threshold),
        thresholds=thresholds,
        metrics=metrics,
        artifacts={"checkpoint": str(checkpoint_path)},
        notes=["Synthetic tiny-batch Stage 0 overfit; no Stage 1 full training is run."],
        elapsed_sec=float(time.time() - started),
    )


def _action_from_arm_and_gripper(robot: Any, arm: str, arm_action: np.ndarray, gripper_name: str, gripper_action: np.ndarray) -> np.ndarray:
    return np.asarray(robot.create_action_vector({arm: arm_action, gripper_name: gripper_action}), dtype=np.float32)


def run_gate1_motor_shell(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    seed: int = 2026062401,
    block_ids: Sequence[int] = (0, 4, 8),
    perturbations: Sequence[Sequence[float]] = ((0.0, 0.0, 0.0), (0.08, 0.0, 0.0), (0.0, -0.08, 0.0)),
    contact_radius_m: float = 0.025,
    z_tolerance_m: float = 0.055,
    max_steps: int = 500,
    speed_gain: float = 0.12,
    arrival_threshold: float = 0.012,
    dwell_steps: int = 4,
    gripper_settle_steps: int = 20,
    success_rate_threshold: float = 1.0,
) -> GateResult:
    started = time.time()
    from corsi.envs.robosuite_corsi import create_env, init_sequence_state, site_pos_in_base, step_pointing_policy

    env = create_env(render_mode="offline", offline_cameras=["agentview"], use_camera_obs=False, seed=int(seed))
    env.hard_reset = False
    trials: list[dict[str, Any]] = []
    try:
        for block_id in [int(value) for value in block_ids]:
            for perturb_index, perturb in enumerate(perturbations):
                state = init_sequence_state(
                    env,
                    [block_id],
                    target_height=float(config.get("target_height", 0.04)),
                    speed_gain=float(speed_gain),
                    arrival_threshold=float(arrival_threshold),
                    dwell_steps=int(dwell_steps),
                    gripper_settle_steps=int(gripper_settle_steps),
                )
                robot = state["robot"]
                arm = str(state["arm"])
                gripper_name = str(state["gripper_name"])
                arm_action = np.zeros(int(state["arm_dim"]), dtype=np.float32)
                arm_action[:3] = np.asarray(perturb, dtype=np.float32)
                perturb_action = _action_from_arm_and_gripper(
                    robot,
                    arm,
                    np.clip(arm_action, -1.0, 1.0),
                    gripper_name,
                    np.asarray(state["gripper_action"], dtype=np.float32),
                )
                for _ in range(4):
                    state["obs"], _, _, _ = env.step(perturb_action)

                block_center = robot.pose_in_base_from_name(state["block_names"][block_id])[:3, 3]
                desired_tip = block_center + np.array([0.0, 0.0, float(config.get("target_height", 0.04))], dtype=np.float32)
                min_xy = float("inf")
                min_z = float("inf")
                steps_run = 0
                for steps_run in range(1, int(max_steps) + 1):
                    action = step_pointing_policy(state)
                    state["obs"], _, _, _ = env.step(action)
                    tip = site_pos_in_base(robot, str(state["target_site_name"]))
                    xy_error = float(np.linalg.norm((tip - desired_tip)[:2]))
                    z_error = float(abs((tip - desired_tip)[2]))
                    min_xy = min(min_xy, xy_error)
                    min_z = min(min_z, z_error)
                    state["step"] = int(state["step"]) + 1
                    if bool(state["completed"]):
                        break
                success = bool(min_xy <= contact_radius_m and min_z <= z_tolerance_m and bool(state["completed"]))
                trials.append(
                    {
                        "block_id": block_id,
                        "perturbation_index": int(perturb_index),
                        "perturbation_arm_action_xyz": [float(value) for value in perturb],
                        "steps_run": int(steps_run),
                        "completed": bool(state["completed"]),
                        "min_xy_error_m": min_xy,
                        "min_z_error_m": min_z,
                        "success": success,
                    }
                )
    finally:
        env.close()

    success_count = sum(1 for trial in trials if trial["success"])
    success_rate = float(success_count / max(len(trials), 1))
    trials_path = output_root / "gate1_motor_shell_trials.json"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    trials_path.write_text(json.dumps(trials, indent=2, sort_keys=True), encoding="utf-8")
    thresholds = {
        "block_ids": [int(value) for value in block_ids],
        "perturbations": [[float(item) for item in value] for value in perturbations],
        "contact_zone_xy_radius_m": float(contact_radius_m),
        "z_tolerance_m": float(z_tolerance_m),
        "max_steps_per_trial": int(max_steps),
        "success_rate_min": float(success_rate_threshold),
    }
    metrics = {
        "trial_count": int(len(trials)),
        "success_count": int(success_count),
        "success_rate": success_rate,
        "max_steps_observed": int(max((trial["steps_run"] for trial in trials), default=0)),
        "worst_min_xy_error_m": float(max((trial["min_xy_error_m"] for trial in trials), default=float("nan"))),
        "worst_min_z_error_m": float(max((trial["min_z_error_m"] for trial in trials), default=float("nan"))),
    }
    return GateResult(
        gate="gate1_motor_shell_single_block_reachability",
        passed=bool(success_rate >= success_rate_threshold),
        thresholds=thresholds,
        metrics=metrics,
        artifacts={"trials": str(trials_path)},
        notes=["Uses existing robosuite Corsi pointing-policy helpers as the minimal V2 Motor Shell diagnostic."],
        elapsed_sec=float(time.time() - started),
    )


def run_stage0(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    device_name: str = "cpu",
    start_gate: int = 1,
    stop_after_gate: int = 4,
    seed: int = 2026062400,
) -> dict[str, Any]:
    config = load_config(config_path)
    output_root = Path(output_root)
    device = torch.device(device_name)
    summary: dict[str, Any] = {
        "stage": 0,
        "config_path": str(config_path),
        "output_root": str(output_root),
        "device": str(device),
        "seed": int(seed),
        "audit": stage0_audit(),
        "gates": [],
        "final_status": "running",
        "recommendation": "block_stage1_stage2_until_stage0_completes",
    }
    summary_path = output_root / SUMMARY_NAME
    write_summary(summary_path, summary)

    gate_fns = [
        lambda: run_gate1_motor_shell(config=config, output_root=output_root, seed=seed + 1),
        lambda: run_gate2_oracle_recall(config=config, output_root=output_root, device=device, seed=seed + 2),
        lambda: run_gate3_frozen_probe(config=config, output_root=output_root, device=device, seed=seed + 3),
        lambda: run_gate4_tiny_overfit(config=config, output_root=output_root, device=device, seed=seed + 4),
    ]
    for gate_number, gate_fn in enumerate(gate_fns, start=1):
        if gate_number < int(start_gate) or gate_number > int(stop_after_gate):
            continue
        try:
            result = gate_fn()
        except Exception as exc:
            result = GateResult(
                gate=f"gate{gate_number}",
                passed=False,
                thresholds={},
                metrics={"exception_type": type(exc).__name__, "exception": str(exc)},
                artifacts={},
                notes=["Gate raised an exception; later gates were not run."],
                elapsed_sec=0.0,
            )
        summary["gates"].append(asdict(result))
        if not result.passed:
            summary["final_status"] = "failed"
            summary["failed_gate"] = result.gate
            summary["recommendation"] = "block_further_stage1_stage2_training"
            write_summary(summary_path, summary)
            return summary
        write_summary(summary_path, summary)

    summary["final_status"] = "passed"
    summary["recommendation"] = "allow_stage1_stage2_resume_but_stage0_runner_did_not_start_training"
    write_summary(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V2 Stage 0 identifiability / upper-bound gates sequentially.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--start-gate", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--stop-after-gate", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--seed", type=int, default=2026062400)
    args = parser.parse_args(argv)
    summary = run_stage0(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        device_name=str(args.device),
        start_gate=int(args.start_gate),
        stop_after_gate=int(args.stop_after_gate),
        seed=int(args.seed),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0 if summary.get("final_status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
