"""Stage 1/2 training scaffolding for Corsi memory-recall V2."""

from __future__ import annotations

import argparse
import inspect
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from corsi.experiments.corsi_memory_recall_v2.analysis import compute_sequence_metrics


CHECKPOINT_VERSION = "corsi_memory_recall_v2_checkpoint_v1"
MODEL_INTERFACE_NOTE = (
    "Lane C is expected to provide corsi.experiments.corsi_memory_recall_v2.model.build_model "
    "and optionally losses.compute_stage1_loss/compute_stage2_loss. Model outputs should expose "
    "'token_logits' or 'logits' for Stage 2 and auxiliary prediction tensors for Stage 1."
)
STAGE1_WARM_START_PREFIXES = (
    "visual_encoder.",
    "visual_lstm.",
    "motor_lstm.",
    "joint_head.",
    "ee_pose_head.",
    "ee_xy_head.",
)
STAGE2_PRIMARY_SELECTION = "full_sequence"


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def capture_rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])


def canonical_manifest_path(config: Mapping[str, Any]) -> Path:
    if "canonical_manifest_path" in config:
        return Path(str(config["canonical_manifest_path"]))
    return Path(str(config["canonical_root"])) / "manifest.json"


def dataset_fingerprint(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "canonical_fingerprint": manifest.get("canonical_fingerprint"),
        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "sample_count": len(manifest.get("samples", [])),
        "split_counts": {
            key: len(value) for key, value in dict(manifest.get("split", {})).items()
        },
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    stage: int,
    best_metric: float,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    seed: int,
    scheduler: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "created_time": time.time(),
        "epoch": int(epoch),
        "stage": int(stage),
        "seed": int(seed),
        "best_metric": float(best_metric),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(),
        "config": dict(config),
        "fingerprint_state": dataset_fingerprint(manifest),
        "canonical_fingerprint": manifest.get("canonical_fingerprint"),
        "normalization": manifest.get("normalization"),
        "git_commit": git_commit(),
        "model_interface_note": MODEL_INTERFACE_NOTE,
    }
    if extra:
        payload["extra"] = dict(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        state = dict(payload["model_state_dict"])
        target_state = model.state_dict()
        for key, value in target_state.items():
            if (
                key.startswith("memory_order_head.") or key.startswith("final_memory_order_head.")
                or key.startswith("final_memory_length_head.")
                or key.startswith("memory_slot_item_projection.")
                or key.startswith("memory_slot_projection.")
                or key == "memory_slot_position"
            ) and key not in state:
                state[key] = value
        model.load_state_dict(state)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if restore_rng and payload.get("rng_state"):
        restore_rng_state(payload["rng_state"])
    return payload


def load_stage1_warm_start(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load Stage 1 grounding modules into a Stage 2 model, leaving memory/recall random."""

    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    source_stage = int(payload.get("stage", -1))
    if source_stage != 1:
        raise ValueError(f"Stage 2 warm-start requires a Stage 1 checkpoint, got stage={source_stage}: {checkpoint}")
    source_state = payload.get("model_state_dict")
    if not isinstance(source_state, Mapping):
        raise ValueError(f"checkpoint does not contain a model_state_dict: {checkpoint}")
    target_state = model.state_dict()
    loaded_keys = []
    skipped: dict[str, str] = {}
    updated_state = dict(target_state)
    for key, value in source_state.items():
        if not any(str(key).startswith(prefix) for prefix in STAGE1_WARM_START_PREFIXES):
            continue
        if key not in target_state:
            skipped[str(key)] = "missing in target model"
            continue
        if tuple(target_state[key].shape) != tuple(value.shape):
            skipped[str(key)] = f"shape {tuple(value.shape)} != {tuple(target_state[key].shape)}"
            continue
        updated_state[key] = value.to(device=target_state[key].device, dtype=target_state[key].dtype)
        loaded_keys.append(str(key))
    if not loaded_keys:
        raise ValueError(f"no Stage 1 grounding keys were loaded from {checkpoint}")
    model.load_state_dict(updated_state, strict=True)
    return {
        "path": str(checkpoint),
        "source_stage": source_stage,
        "source_epoch": int(payload.get("epoch", -1)),
        "source_best_metric": float(payload.get("best_metric", float("nan"))),
        "loaded_key_count": int(len(loaded_keys)),
        "loaded_prefixes": sorted(
            {
                prefix.rstrip(".")
                for prefix in STAGE1_WARM_START_PREFIXES
                if any(key.startswith(prefix) for key in loaded_keys)
            }
        ),
        "skipped_key_count": int(len(skipped)),
        "skipped_keys": dict(list(sorted(skipped.items()))[:20]),
    }


def selection_specs_for_stage(stage: int, val_metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return checkpoint selection scores; tuple ordering implements tie-breaks."""

    val_loss = float(val_metrics.get("loss", 0.0))
    if int(stage) != 2:
        return {
            "val_loss": {
                "path_name": "best_val_loss.pt",
                "metric": "negative_val_loss",
                "score": (-val_loss,),
            }
        }
    full = float(val_metrics.get("full_sequence_accuracy", 0.0))
    token = float(val_metrics.get("token_accuracy", 0.0))
    length = float(val_metrics.get("predicted_length_accuracy", 0.0))
    return {
        "full_sequence": {
            "path_name": "best_full_sequence.pt",
            "metric": "val_full_sequence_accuracy__token_accuracy__predicted_length_accuracy__negative_val_loss",
            "score": (full, token, length, -val_loss),
        },
        "token": {
            "path_name": "best_token.pt",
            "metric": "val_token_accuracy__full_sequence_accuracy__predicted_length_accuracy__negative_val_loss",
            "score": (token, full, length, -val_loss),
        },
        "val_loss": {
            "path_name": "best_val_loss.pt",
            "metric": "negative_val_loss__full_sequence_accuracy__token_accuracy__predicted_length_accuracy",
            "score": (-val_loss, full, token, length),
        },
    }


def validate_training_scope(
    *,
    max_epochs: int,
    overfit_episodes: int,
    allow_full_training: bool = False,
    smoke_epoch_limit: int = 5,
) -> None:
    if allow_full_training:
        return
    if int(overfit_episodes) <= 0 and int(max_epochs) > int(smoke_epoch_limit):
        raise ValueError(
            "Refusing full V2 training from Lane D scaffolding. Use --overfit-episodes for "
            "tiny smoke runs or pass --allow-full-training only after integration approval."
        )


def _import_dataset_components() -> tuple[Any, Any]:
    try:
        from corsi.experiments.corsi_memory_recall_v2.dataset import (
            CorsiMemoryRecallV2Dataset,
            collate_memory_recall_v2_batch,
        )
    except ImportError as exc:
        raise RuntimeError("Lane B dataset components are required for data-backed training.") from exc
    return CorsiMemoryRecallV2Dataset, collate_memory_recall_v2_batch


def _build_model(config: Mapping[str, Any], *, stage: int, manifest: Mapping[str, Any]) -> nn.Module:
    try:
        from corsi.experiments.corsi_memory_recall_v2.model import build_model
    except ImportError as exc:
        raise RuntimeError(f"Lane C model builder is not available. {MODEL_INTERFACE_NOTE}") from exc
    return build_model(config=config, stage=stage, manifest=manifest)


def _call_model(model: nn.Module, batch: Mapping[str, Any], *, stage: int) -> Mapping[str, Any]:
    model_inputs = batch["model_inputs"]
    if hasattr(model, "forward_stage1") and stage == 1:
        return model.forward_stage1(**model_inputs)
    if hasattr(model, "forward_stage2") and stage == 2:
        return model.forward_stage2(**model_inputs)
    forward_parameters = inspect.signature(model.forward).parameters
    call_kwargs = dict(model_inputs)
    if "stage" in forward_parameters:
        call_kwargs["stage"] = stage
    if "max_recall_steps" in forward_parameters and stage == 2:
        target_tokens = batch.get("targets", {}).get("tokens")
        if torch.is_tensor(target_tokens):
            call_kwargs["max_recall_steps"] = int(target_tokens.shape[1])
    output = model(**call_kwargs)
    if isinstance(output, Mapping):
        return output
    raise TypeError("V2 model forward must return a mapping of output tensors")


def pad_sequence_metric_batches(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
    *,
    ignore_index: int,
    eos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad batch-local recall outputs so sequence metrics can aggregate them."""

    if not predictions:
        raise ValueError("predictions must not be empty")
    max_steps = max(
        max(int(tensor.shape[1]) for tensor in predictions),
        max(int(tensor.shape[1]) for tensor in targets),
        max(int(tensor.shape[1]) for tensor in masks),
    )
    padded_predictions = []
    padded_targets = []
    padded_masks = []
    for prediction, target, mask in zip(predictions, targets, masks):
        pred_steps = int(prediction.shape[1])
        target_steps = int(target.shape[1])
        mask_steps = int(mask.shape[1])
        if pred_steps < max_steps:
            if prediction.ndim == 3:
                pad_shape = (prediction.shape[0], max_steps - pred_steps, prediction.shape[2])
                pred_pad = torch.zeros(pad_shape, dtype=prediction.dtype, device=prediction.device)
                if 0 <= int(eos_token_id) < int(prediction.shape[2]):
                    pred_pad[..., int(eos_token_id)] = torch.finfo(prediction.dtype).min
            elif prediction.ndim == 2:
                fill_token = 1 if int(eos_token_id) == 0 else 0
                pred_pad = torch.full(
                    (prediction.shape[0], max_steps - pred_steps),
                    fill_token,
                    dtype=prediction.dtype,
                    device=prediction.device,
                )
            else:
                raise ValueError(f"expected prediction [B,T] or [B,T,C], got shape={tuple(prediction.shape)}")
            prediction = torch.cat([prediction, pred_pad], dim=1)
        if target_steps < max_steps:
            target_pad = torch.full(
                (target.shape[0], max_steps - target_steps),
                int(ignore_index),
                dtype=target.dtype,
                device=target.device,
            )
            target = torch.cat([target, target_pad], dim=1)
        if mask_steps < max_steps:
            mask_pad = torch.zeros((mask.shape[0], max_steps - mask_steps), dtype=mask.dtype, device=mask.device)
            mask = torch.cat([mask, mask_pad], dim=1)
        padded_predictions.append(prediction)
        padded_targets.append(target)
        padded_masks.append(mask)
    return (
        torch.cat(padded_predictions, dim=0),
        torch.cat(padded_targets, dim=0),
        torch.cat(padded_masks, dim=0),
    )


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < prediction.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(prediction)
    if int(expanded.sum().item()) == 0:
        return prediction.sum() * 0.0
    diff = prediction[expanded] - target[expanded]
    return torch.mean(diff * diff)


def compute_stage1_loss(outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> torch.Tensor:
    try:
        from corsi.experiments.corsi_memory_recall_v2.losses import compute_stage1_loss as lane_c_loss

        return lane_c_loss(outputs, batch)
    except ImportError:
        targets = batch["targets"]
        mask = batch["model_inputs"]["frame_mask"]
        losses = []
        for output_key, target_key in (
            ("joint", "joint"),
            ("pred_joint", "joint"),
            ("ee_pose", "ee_pose"),
            ("pred_ee_pose", "ee_pose"),
            ("ee_xy", "ee_xy"),
            ("pred_ee_xy", "ee_xy"),
        ):
            if output_key in outputs and target_key in targets:
                losses.append(_masked_mse(outputs[output_key], targets[target_key], mask))
        if not losses:
            raise RuntimeError(f"Stage 1 loss needs Lane C losses or auxiliary outputs. {MODEL_INTERFACE_NOTE}")
        return sum(losses)


def compute_stage2_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    try:
        from corsi.experiments.corsi_memory_recall_v2.losses import compute_stage2_loss as lane_c_loss

        return lane_c_loss(outputs, batch, weights=weights)
    except ImportError:
        logits = outputs.get("token_logits", outputs.get("logits"))
        if logits is None:
            raise RuntimeError(f"Stage 2 loss needs token logits. {MODEL_INTERFACE_NOTE}")
        targets = batch["targets"]["tokens"]
        ignore_index = int(batch.get("ignore_index", -100))
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=ignore_index,
        )


def train_epoch(
    model: nn.Module,
    loader: Any,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stage: int,
    scaler: torch.amp.GradScaler | None = None,
    loss_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = _call_model(model, batch, stage=stage)
            loss = (
                compute_stage1_loss(outputs, batch)
                if stage == 1
                else compute_stage2_loss(outputs, batch, weights=loss_weights)
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Stage {stage} loss")
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach().cpu().item())
        count += 1
    return {"loss": float(total / max(count, 1)), "batches": float(count)}


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: Any,
    *,
    device: torch.device,
    stage: int,
    loss_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    model.eval()
    losses = []
    predictions = []
    targets = []
    masks = []
    eos_token_id = 9
    ignore_index = -100
    for batch in loader:
        batch = move_to_device(batch, device)
        eos_values = batch.get("metadata", {}).get("eos_token_id")
        if torch.is_tensor(eos_values):
            eos_token_id = int(eos_values.flatten()[0].item())
        ignore_index = int(batch.get("ignore_index", ignore_index))
        outputs = _call_model(model, batch, stage=stage)
        loss = (
            compute_stage1_loss(outputs, batch)
            if stage == 1
            else compute_stage2_loss(outputs, batch, weights=loss_weights)
        )
        losses.append(float(loss.detach().cpu().item()))
        if stage == 2:
            logits = outputs.get("token_logits", outputs.get("logits"))
            predictions.append(logits.detach().cpu())
            targets.append(batch["targets"]["tokens"].detach().cpu())
            masks.append(batch["targets"]["token_mask"].detach().cpu())
    result: dict[str, Any] = {"loss": float(np.mean(losses)) if losses else 0.0}
    if stage == 2 and predictions:
        padded_predictions, padded_targets, padded_masks = pad_sequence_metric_batches(
            predictions,
            targets,
            masks,
            ignore_index=ignore_index,
            eos_token_id=eos_token_id,
        )
        metrics = compute_sequence_metrics(
            padded_predictions,
            padded_targets,
            padded_masks,
            eos_token_id=eos_token_id,
            ignore_index=ignore_index,
        )
        result.update(metrics)
    return result


def make_loader(
    manifest_path: Path,
    *,
    split: str,
    batch_size: int,
    shuffle: bool,
    seq_ids: list[str] | None = None,
    load_images: bool = True,
) -> Any:
    dataset_cls, collate_fn = _import_dataset_components()
    dataset = dataset_cls(manifest_path, split=split, seq_ids=seq_ids, load_images=load_images)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        collate_fn=collate_fn,
    )


def train_one_run(
    config: Mapping[str, Any],
    *,
    stage: int,
    seed: int,
    run_dir: str | Path,
    max_epochs: int | None = None,
    overfit_episodes: int = 0,
    resume: bool = False,
    allow_full_training: bool = False,
    dry_run: bool = False,
    warm_start_stage1_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    max_epochs = int(max_epochs if max_epochs is not None else config.get("max_epochs", 3))
    validate_training_scope(
        max_epochs=max_epochs,
        overfit_episodes=overfit_episodes,
        allow_full_training=allow_full_training,
    )
    set_seed(seed)
    device = resolve_device(str(config.get("device", "auto")))
    manifest_path = canonical_manifest_path(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = Path(run_dir)
    configured_warm_start = warm_start_stage1_checkpoint or config.get("warm_start_stage1_checkpoint", "")
    warm_start_path = str(configured_warm_start) if configured_warm_start else ""
    if warm_start_path and int(stage) != 2:
        raise ValueError("--warm-start-stage1-checkpoint is only valid for Stage 2 training")
    loss_weights = dict(config.get("loss_weights", {})) if isinstance(config.get("loss_weights", {}), Mapping) else {}
    summary = {
        "stage": int(stage),
        "seed": int(seed),
        "run_dir": str(run_dir),
        "device": str(device),
        "max_epochs": int(max_epochs),
        "overfit_episodes": int(overfit_episodes),
        "dry_run": bool(dry_run),
        "warm_start_stage1_checkpoint": warm_start_path,
        "loss_weights": loss_weights,
        "model_interface_note": MODEL_INTERFACE_NOTE,
        "fingerprint_state": dataset_fingerprint(manifest),
    }
    if dry_run:
        return summary

    train_ids = None
    val_split = "val"
    if int(overfit_episodes) > 0:
        train_ids = list(manifest["split"]["train"])[: int(overfit_episodes)]
        val_split = "train"
    batch_size = int(config.get("batch_size", 4))
    train_loader = make_loader(
        manifest_path,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        seq_ids=train_ids,
    )
    val_loader = make_loader(
        manifest_path,
        split=val_split,
        batch_size=batch_size,
        shuffle=False,
        seq_ids=train_ids,
    )
    model = _build_model(config, stage=stage, manifest=manifest).to(device)
    latest_path = run_dir / "latest.pt"
    resuming_existing_run = bool(resume and latest_path.exists())
    warm_start_info: dict[str, Any] | None = None
    if warm_start_path and not resuming_existing_run:
        warm_start_info = load_stage1_warm_start(model, warm_start_path, map_location=device)
    elif warm_start_path and resuming_existing_run:
        warm_start_info = {
            "path": warm_start_path,
            "skipped": "resume checkpoint exists; warm-start is applied only to fresh Stage 2 runs",
        }
    if warm_start_info is not None:
        summary["warm_start"] = warm_start_info
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and bool(config.get("amp", True)) else None
    start_epoch = 0
    best_metric = -float("inf")
    primary_selection = STAGE2_PRIMARY_SELECTION if int(stage) == 2 else "val_loss"
    best_scores: dict[str, tuple[float, ...]] = {}
    best_paths: dict[str, str] = {}
    if resuming_existing_run:
        payload = load_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
            restore_rng=True,
        )
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_metric = float(payload.get("best_metric", best_metric))
        extra = payload.get("extra") or {}
        for key, value in dict(extra.get("best_scores", {})).items():
            best_scores[str(key)] = tuple(float(item) for item in value)
        best_paths.update({str(key): str(value) for key, value in dict(extra.get("best_paths", {})).items()})

    history = []
    best_path = run_dir / "best.pt"
    for epoch in range(start_epoch, max_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            stage=stage,
            scaler=scaler,
            loss_weights=loss_weights,
        )
        val_metrics = evaluate_loader(model, val_loader, device=device, stage=stage, loss_weights=loss_weights)
        selection_specs = selection_specs_for_stage(stage, val_metrics)
        selection_scores = {key: tuple(float(item) for item in spec["score"]) for key, spec in selection_specs.items()}
        primary_score = selection_scores[primary_selection]
        improved_primary = False
        row_best_scores = {key: list(value) for key, value in best_scores.items()}
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "selection_scores": {key: list(value) for key, value in selection_scores.items()},
        }
        history.append(row)
        for key, spec in selection_specs.items():
            score = selection_scores[key]
            improved = key not in best_scores or score > best_scores[key]
            if not improved:
                continue
            best_scores[key] = score
            row_best_scores[key] = list(score)
            path = run_dir / str(spec["path_name"])
            best_paths[key] = str(path)
            if key == primary_selection:
                best_metric = float(score[0])
                improved_primary = True
            checkpoint_extra = {
                "selection_metric": str(spec["metric"]),
                "selection_key": key,
                "selection_score": list(score),
                "best_scores": {score_key: list(score_value) for score_key, score_value in best_scores.items()},
                "best_paths": dict(best_paths),
            }
            if warm_start_info is not None:
                checkpoint_extra["warm_start"] = warm_start_info
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                stage=stage,
                best_metric=float(score[0]),
                config=config,
                manifest=manifest,
                seed=seed,
                extra=checkpoint_extra,
            )
        row["best_metric"] = best_metric
        row["best_scores"] = row_best_scores
        latest_extra = {
            "selection_metric": selection_specs[primary_selection]["metric"],
            "selection_key": primary_selection,
            "selection_score": list(primary_score),
            "best_scores": {key: list(value) for key, value in best_scores.items()},
            "best_paths": dict(best_paths),
        }
        if warm_start_info is not None:
            latest_extra["warm_start"] = warm_start_info
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            stage=stage,
            best_metric=best_metric,
            config=config,
            manifest=manifest,
            seed=seed,
            extra=latest_extra,
        )
        if improved_primary:
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                stage=stage,
                best_metric=best_metric,
                config=config,
                manifest=manifest,
                seed=seed,
                extra=latest_extra,
            )
    summary.update(
        {
            "history": history,
            "best_metric": float(best_metric),
            "best_checkpoint": str(best_path),
            "primary_selection": primary_selection,
            "best_checkpoints": {"primary": str(best_path), **dict(best_paths)},
            "best_scores": {key: list(value) for key, value in best_scores.items()},
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Corsi memory-recall V2 Stage 1 or Stage 2.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--overfit-episodes", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-full-training", action="store_true")
    parser.add_argument("--warm-start-stage1-checkpoint", default="")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config["device"] = args.device
    output_root = Path(args.output_root or config.get("output_root", "corsi_artifacts/memory_recall_v2/runs"))
    run_name = args.run_name or f"stage{args.stage}_seed{args.seed}"
    summary = train_one_run(
        config,
        stage=int(args.stage),
        seed=int(args.seed),
        run_dir=output_root / run_name,
        max_epochs=args.max_epochs,
        overfit_episodes=int(args.overfit_episodes),
        resume=bool(args.resume),
        allow_full_training=bool(args.allow_full_training),
        dry_run=bool(args.dry_run),
        warm_start_stage1_checkpoint=args.warm_start_stage1_checkpoint,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
