"""Training entry point for the visual robosuite Corsi sequence model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from corsi.analysis.metrics import summarize_sequence_metrics
from corsi.data import RobosuiteVisualCorsiDataset, collate_visual_batch
from corsi.models.attention import (
    CapacityGateConfig,
    LocalAttentionConfig,
    MemoryDecayConfig,
    NoisyAttentionConfig,
    ResponseSuppressionConfig,
)
from corsi.models.lstm_visual import VisualLSTMConfig, VisualSeq2SeqLSTM
from corsi.training.device import resolve_torch_device

CHECKPOINT_VERSION = 2
TRACE_SAMPLES_PER_LENGTH = 2
LENGTH_RANGE = range(2, 7)


@dataclass
class TrainVisualConfig:
    dataset_root: str = "corsi_artifacts/visual_base/datasets/robosuite_visual_dataset_preview"
    val_dataset_root: str = ""
    camera_name: str = "freecam"
    include_reset_frame: bool = False
    val_ratio: float = 0.25
    batch_size: int = 4
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    cnn_feature_dim: int = 128
    token_embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    input_image_size: int = 128
    use_attention: bool = False
    attention_dim: int = 128
    attention_type: str = "global"
    attention_temperature: float = 1.0
    local_attention: Optional[Dict[str, Any]] = None
    noisy_attention: Optional[Dict[str, Any]] = None
    memory_decay: Optional[Dict[str, Any]] = None
    capacity_gate: Optional[Dict[str, Any]] = None
    response_suppression: Optional[Dict[str, Any]] = None
    use_step_embedding: bool = False
    max_decode_steps: int = 6
    step_embedding_dim: int = 16
    scheduled_sampling: bool = False
    scheduled_sampling_start: float = 1.0
    scheduled_sampling_end: float = 0.5
    scheduled_sampling_warmup_epochs: int = 40
    early_stopping_patience: int = 0
    delay_mode: str = "none"
    delay_steps: int = 0
    blank_feature_mode: str = "zero_feature"
    return_hidden_traces: bool = False
    return_error_analysis: bool = False
    analysis_output_dir: str = ""
    seed: int = 7
    device: str = "auto"
    output_dir: str = "corsi_artifacts/visual_base/training/visual_lstm"
    checkpoint_dir: str = ""
    resume: bool = False
    init_model: str = ""
    save_rng_state: bool = True
    strict_resume: bool = True
    auto_resume: bool = False


def normalize_config_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(data)

    model_overrides = normalized.pop("model", None)
    if isinstance(model_overrides, dict):
        if "use_attention" in model_overrides:
            normalized["use_attention"] = model_overrides["use_attention"]
        if "use_step_embedding" in model_overrides:
            normalized["use_step_embedding"] = model_overrides["use_step_embedding"]
        for key in (
            "attention_type",
            "attention_temperature",
            "local_attention",
            "noisy_attention",
            "memory_decay",
            "capacity_gate",
            "response_suppression",
        ):
            if key in model_overrides:
                normalized[key] = model_overrides[key]

    training_overrides = normalized.pop("training", None)
    if isinstance(training_overrides, dict):
        alias_map = {
            "use_scheduled_sampling": "scheduled_sampling",
            "scheduled_sampling": "scheduled_sampling",
            "scheduled_sampling_start": "scheduled_sampling_start",
            "scheduled_sampling_end": "scheduled_sampling_end",
            "scheduled_sampling_warmup_epochs": "scheduled_sampling_warmup_epochs",
            "early_stopping_patience": "early_stopping_patience",
            "max_epochs": "epochs",
            "epochs": "epochs",
            "seed": "seed",
        }
        for source_key, target_key in alias_map.items():
            if source_key in training_overrides:
                normalized[target_key] = training_overrides[source_key]

    if "use_scheduled_sampling" in normalized:
        normalized["scheduled_sampling"] = normalized.pop("use_scheduled_sampling")

    if "max_epochs" in normalized:
        normalized["epochs"] = normalized.pop("max_epochs")

    if normalized.get("attention_type", "global") != "global":
        normalized.setdefault("use_attention", True)

    return normalized


def load_config_overrides(config_path: str) -> Dict[str, object]:
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object")
    base_config = data.pop("base_config", "")
    if base_config:
        base_path = Path(base_config)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base_data = load_config_overrides(str(base_path))
        base_data.update(normalize_config_overrides(data))
        return base_data
    return normalize_config_overrides(data)


def parse_args() -> TrainVisualConfig:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default="")
    bootstrap_args, _ = bootstrap.parse_known_args()

    defaults = asdict(TrainVisualConfig())
    if bootstrap_args.config:
        defaults.update(load_config_overrides(bootstrap_args.config))

    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.add_argument("--dataset-root", type=str, default=defaults["dataset_root"])
    parser.add_argument("--val-dataset-root", type=str, default=defaults["val_dataset_root"])
    parser.add_argument("--camera-name", type=str, default=defaults["camera_name"])
    parser.add_argument("--include-reset-frame", action="store_true", default=defaults["include_reset_frame"])
    parser.add_argument("--val-ratio", type=float, default=defaults["val_ratio"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--cnn-feature-dim", type=int, default=defaults["cnn_feature_dim"])
    parser.add_argument("--token-embedding-dim", type=int, default=defaults["token_embedding_dim"])
    parser.add_argument("--hidden-dim", type=int, default=defaults["hidden_dim"])
    parser.add_argument("--num-layers", type=int, default=defaults["num_layers"])
    parser.add_argument("--dropout", type=float, default=defaults["dropout"])
    parser.add_argument("--input-image-size", type=int, default=defaults["input_image_size"])
    parser.add_argument(
        "--use-attention",
        action=argparse.BooleanOptionalAction,
        default=defaults["use_attention"],
    )
    parser.add_argument("--attention-dim", type=int, default=defaults["attention_dim"])
    parser.add_argument("--attention-type", type=str, default=defaults["attention_type"])
    parser.add_argument("--attention-temperature", type=float, default=defaults["attention_temperature"])
    parser.add_argument("--local-attention", type=json.loads, default=defaults["local_attention"])
    parser.add_argument("--noisy-attention", type=json.loads, default=defaults["noisy_attention"])
    parser.add_argument("--memory-decay", type=json.loads, default=defaults["memory_decay"])
    parser.add_argument("--capacity-gate", type=json.loads, default=defaults["capacity_gate"])
    parser.add_argument(
        "--response-suppression",
        type=json.loads,
        default=defaults["response_suppression"],
    )
    parser.add_argument(
        "--use-step-embedding",
        action=argparse.BooleanOptionalAction,
        default=defaults["use_step_embedding"],
    )
    parser.add_argument("--max-decode-steps", type=int, default=defaults["max_decode_steps"])
    parser.add_argument("--step-embedding-dim", type=int, default=defaults["step_embedding_dim"])
    parser.add_argument(
        "--scheduled-sampling",
        action=argparse.BooleanOptionalAction,
        default=defaults["scheduled_sampling"],
    )
    parser.add_argument(
        "--scheduled-sampling-start",
        type=float,
        default=defaults["scheduled_sampling_start"],
    )
    parser.add_argument(
        "--scheduled-sampling-end",
        type=float,
        default=defaults["scheduled_sampling_end"],
    )
    parser.add_argument(
        "--scheduled-sampling-warmup-epochs",
        type=int,
        default=defaults["scheduled_sampling_warmup_epochs"],
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=defaults["early_stopping_patience"],
    )
    parser.add_argument(
        "--delay-mode",
        type=str,
        default=defaults["delay_mode"],
        choices=["none", "hold_state", "encoder_blanks"],
    )
    parser.add_argument("--delay-steps", type=int, default=defaults["delay_steps"])
    parser.add_argument(
        "--blank-feature-mode",
        type=str,
        default=defaults["blank_feature_mode"],
        choices=["zero_feature", "zero_image"],
    )
    parser.add_argument(
        "--return-hidden-traces",
        action=argparse.BooleanOptionalAction,
        default=defaults["return_hidden_traces"],
    )
    parser.add_argument(
        "--return-error-analysis",
        action=argparse.BooleanOptionalAction,
        default=defaults["return_error_analysis"],
    )
    parser.add_argument("--analysis-output-dir", type=str, default=defaults["analysis_output_dir"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--device", type=str, default=defaults["device"], choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output-dir", type=str, default=defaults["output_dir"])
    parser.add_argument("--checkpoint-dir", type=str, default=defaults["checkpoint_dir"])
    parser.add_argument("--resume", action="store_true", default=defaults["resume"])
    parser.add_argument("--init-model", type=str, default=defaults["init_model"])
    parser.add_argument(
        "--save-rng-state",
        action=argparse.BooleanOptionalAction,
        default=defaults["save_rng_state"],
    )
    parser.add_argument(
        "--strict-resume",
        action=argparse.BooleanOptionalAction,
        default=defaults["strict_resume"],
    )
    parser.add_argument("--auto-resume", action="store_true", default=defaults["auto_resume"])

    parsed = vars(parser.parse_args())
    parsed.pop("config", None)
    return TrainVisualConfig(**parsed)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_config(train_config: TrainVisualConfig) -> VisualLSTMConfig:
    local_attention = LocalAttentionConfig(**(train_config.local_attention or {}))
    noisy_attention = NoisyAttentionConfig(**(train_config.noisy_attention or {}))
    memory_decay = MemoryDecayConfig(**(train_config.memory_decay or {}))
    capacity_gate = CapacityGateConfig(**(train_config.capacity_gate or {}))
    response_suppression = ResponseSuppressionConfig(**(train_config.response_suppression or {}))
    return VisualLSTMConfig(
        cnn_feature_dim=train_config.cnn_feature_dim,
        token_embedding_dim=train_config.token_embedding_dim,
        hidden_dim=train_config.hidden_dim,
        num_layers=train_config.num_layers,
        dropout=train_config.dropout,
        input_image_size=train_config.input_image_size,
        use_attention=train_config.use_attention,
        attention_dim=train_config.attention_dim,
        attention_type=train_config.attention_type,
        attention_temperature=train_config.attention_temperature,
        local_attention=local_attention,
        noisy_attention=noisy_attention,
        memory_decay=memory_decay,
        capacity_gate=capacity_gate,
        response_suppression=response_suppression,
        use_step_embedding=train_config.use_step_embedding,
        max_decode_steps=train_config.max_decode_steps,
        step_embedding_dim=train_config.step_embedding_dim,
        delay_mode=train_config.delay_mode,
        delay_steps=train_config.delay_steps,
        blank_feature_mode=train_config.blank_feature_mode,
        return_hidden_traces=train_config.return_hidden_traces,
        return_error_analysis=train_config.return_error_analysis,
    )


def compute_teacher_forcing_ratio(config: TrainVisualConfig, epoch: int) -> float:
    if not config.scheduled_sampling:
        return 1.0
    if config.scheduled_sampling_warmup_epochs <= 0:
        return float(config.scheduled_sampling_end)
    progress = min(max((epoch - 1) / config.scheduled_sampling_warmup_epochs, 0.0), 1.0)
    ratio = config.scheduled_sampling_start + (
        config.scheduled_sampling_end - config.scheduled_sampling_start
    ) * progress
    return float(ratio)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_visual_batch,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = dict(batch)
    moved["frames"] = batch["frames"].to(device)
    moved["targets"] = batch["targets"].to(device)
    moved["frame_lengths"] = batch["frame_lengths"].to(device)
    moved["original_frame_lengths"] = batch["original_frame_lengths"].to(device)
    moved["target_lengths"] = batch["target_lengths"].to(device)
    moved["mask"] = batch["mask"].to(device)
    return moved


def split_dataset(dataset, val_ratio: float, seed: int) -> tuple[Subset, Subset]:
    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("Need at least 2 samples to create a train/val split from one dataset root")
    val_size = max(1, int(math.ceil(dataset_size * val_ratio)))
    train_size = dataset_size - val_size
    if train_size < 1:
        train_size = dataset_size - 1
        val_size = 1

    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)


def run_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    *,
    training: bool,
    delay_mode: str,
    delay_steps: int,
    blank_feature_mode: str,
    teacher_forcing_ratio: float = 1.0,
    global_step: int = 0,
) -> Dict[str, float]:
    model.train(training)
    loss_total = 0.0
    token_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            frames=batch["frames"],
            frame_lengths=batch["frame_lengths"],
            targets=batch["targets"],
            target_lengths=batch["target_lengths"],
            delay_mode=delay_mode,
            delay_steps=delay_steps,
            blank_feature_mode=blank_feature_mode,
            return_hidden_traces=False,
            teacher_forcing_ratio=teacher_forcing_ratio if training else 1.0,
        )
        logits = outputs["logits"]
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch["targets"].reshape(-1))

        if training:
            if optimizer is None:
                raise ValueError("Optimizer is required when training=True")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        active_tokens = int(batch["mask"].sum().item())
        loss_total += float(loss.item()) * active_tokens
        token_count += active_tokens

    return {
        "loss": loss_total / token_count if token_count > 0 else 0.0,
        "global_step": global_step,
    }


def pad_to_max_steps(tensor, *, max_steps: int, fill_value):
    if tensor.size(1) == max_steps:
        return tensor
    pad_width = max_steps - tensor.size(1)
    pad_shape = (tensor.size(0), pad_width)
    padding = torch.full(pad_shape, fill_value, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=1)


def round_nested(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: round_nested(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_nested(item, digits) for item in value]
    return value


def extract_trace_samples(
    *,
    batch: Dict[str, object],
    predictions,
    hidden_traces: Dict[str, Any],
    quota_by_length: Dict[int, int],
) -> list[Dict[str, Any]]:
    samples: list[Dict[str, Any]] = []
    metadata = hidden_traces["metadata"]
    encoder_time_lengths = metadata["encoder_time_lengths"].detach().cpu()

    for batch_index, target_length in enumerate(batch["target_lengths"].tolist()):
        target_length = int(target_length)
        if quota_by_length.get(target_length, 0) <= 0:
            continue

        frame_length = int(batch["frame_lengths"][batch_index].item())
        original_frame_length = int(batch["original_frame_lengths"][batch_index].item())
        sample_payload: Dict[str, Any] = {
            "trial_id": batch["trial_ids"][batch_index],
            "camera_name": batch["camera_names"][batch_index],
            "target_length": target_length,
            "frame_length": frame_length,
            "original_frame_length": original_frame_length,
            "encoder_time_length": int(encoder_time_lengths[batch_index].item()),
            "delay_mode": metadata["delay_mode"],
            "delay_steps": int(metadata["delay_steps"]),
            "blank_feature_mode": metadata["blank_feature_mode"],
            "blank_feature_l2_norm": float(metadata["blank_feature_l2_norm"]),
            "blank_feature_mean_abs": float(metadata["blank_feature_mean_abs"]),
            "target_sequence": batch["targets"][batch_index, :target_length].detach().cpu().tolist(),
            "prediction_sequence": predictions[batch_index, :target_length].detach().cpu().tolist(),
            "frame_paths": batch["frame_paths"][batch_index],
            "reset_path": batch["reset_paths"][batch_index],
            "sample_metadata": batch["metadata"][batch_index],
            "encoder_last_hidden": hidden_traces["encoder_last_hidden"][batch_index].detach().cpu(),
            "recall_start_hidden": hidden_traces["recall_start_hidden"][batch_index].detach().cpu(),
            "delay_hidden_trace": hidden_traces["delay_hidden_trace"][batch_index].detach().cpu(),
            "decoder_hidden_trace": hidden_traces["decoder_hidden_trace"][
                batch_index, :target_length
            ].detach().cpu(),
        }
        if "encoder_hidden_trace" in hidden_traces:
            sample_payload["encoder_hidden_trace"] = hidden_traces["encoder_hidden_trace"][
                batch_index, :frame_length
            ].detach().cpu()
        samples.append(sample_payload)
        quota_by_length[target_length] -= 1

    return samples


def save_hidden_traces(
    analysis_output_dir: Path,
    *,
    epoch: int,
    trace_samples: list[Dict[str, Any]],
) -> Optional[Path]:
    if not trace_samples:
        return None

    epoch_dir = analysis_output_dir / "hidden_traces" / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    trace_path = epoch_dir / "trace_samples.pt"
    torch.save(
        {
            "epoch": epoch,
            "num_samples": len(trace_samples),
            "samples": trace_samples,
        },
        trace_path,
    )

    length_counts = Counter(sample["target_length"] for sample in trace_samples)
    summary = {
        "epoch": epoch,
        "num_samples": len(trace_samples),
        "samples_per_length": {str(length): int(length_counts.get(length, 0)) for length in LENGTH_RANGE},
        "trace_path": str(trace_path),
    }
    with open(epoch_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return trace_path


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    loss_fn,
    device,
    *,
    delay_mode: str,
    delay_steps: int,
    blank_feature_mode: str,
    return_hidden_traces: bool,
    analysis_output_dir: Optional[Path],
    epoch: Optional[int],
) -> Dict[str, object]:
    model.eval()
    epoch_stats = run_epoch(
        model,
        loader,
        optimizer=None,
        loss_fn=loss_fn,
        device=device,
        training=False,
        delay_mode=delay_mode,
        delay_steps=delay_steps,
        blank_feature_mode=blank_feature_mode,
    )

    predictions_all = []
    targets_all = []
    target_lengths_all = []
    frame_lengths_all = []
    masks_all = []
    trace_samples: list[Dict[str, Any]] = []
    trace_quota = {length: TRACE_SAMPLES_PER_LENGTH for length in LENGTH_RANGE}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        need_traces = return_hidden_traces and any(quota > 0 for quota in trace_quota.values())
        decode_outputs = model.greedy_decode(
            frames=batch["frames"],
            frame_lengths=batch["frame_lengths"],
            target_lengths=batch["target_lengths"],
            max_steps=batch["targets"].size(1),
            delay_mode=delay_mode,
            delay_steps=delay_steps,
            blank_feature_mode=blank_feature_mode,
            return_hidden_traces=need_traces,
        )

        if need_traces:
            predictions = decode_outputs["predictions"]
            trace_samples.extend(
                extract_trace_samples(
                    batch=batch,
                    predictions=predictions,
                    hidden_traces=decode_outputs["hidden_traces"],
                    quota_by_length=trace_quota,
                )
            )
        else:
            predictions = decode_outputs

        predictions_all.append(predictions.cpu())
        targets_all.append(batch["targets"].cpu())
        target_lengths_all.append(batch["target_lengths"].cpu())
        frame_lengths_all.append(batch["frame_lengths"].cpu())
        masks_all.append(batch["mask"].cpu())

    max_steps = max(tensor.size(1) for tensor in targets_all)
    predictions = torch.cat(
        [pad_to_max_steps(tensor, max_steps=max_steps, fill_value=0) for tensor in predictions_all],
        dim=0,
    )
    targets = torch.cat(
        [pad_to_max_steps(tensor, max_steps=max_steps, fill_value=-100) for tensor in targets_all],
        dim=0,
    )
    target_lengths = torch.cat(target_lengths_all, dim=0)
    frame_lengths = torch.cat(frame_lengths_all, dim=0)
    mask = torch.cat(
        [pad_to_max_steps(tensor, max_steps=max_steps, fill_value=False) for tensor in masks_all],
        dim=0,
    )

    metrics = summarize_sequence_metrics(
        predictions,
        targets,
        target_lengths,
        mask,
        min_length=min(LENGTH_RANGE),
        max_length=max(LENGTH_RANGE),
    )
    metrics["loss"] = epoch_stats["loss"]
    metrics["delay_mode"] = delay_mode
    metrics["delay_steps"] = int(delay_steps)
    metrics["blank_feature_mode"] = blank_feature_mode
    metrics["frame_length_mean"] = float(frame_lengths.float().mean().item())

    if return_hidden_traces and analysis_output_dir is not None and epoch is not None:
        trace_path = save_hidden_traces(analysis_output_dir, epoch=epoch, trace_samples=trace_samples)
        if trace_path is not None:
            metrics["hidden_trace_path"] = str(trace_path)
        metrics["hidden_trace_sample_count"] = len(trace_samples)

    return metrics


def build_checkpoint_dir(config: TrainVisualConfig) -> Path:
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else Path(config.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_last_checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "last_checkpoint.pt"


def get_best_checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "best_model.pt"


def collect_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng_state: Optional[Dict[str, Any]]) -> None:
    if not rng_state:
        return
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch_cpu" in rng_state:
        torch.set_rng_state(rng_state["torch_cpu"])
    if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def make_checkpoint(
    *,
    epoch: int,
    global_step: int,
    best_val_metric: float,
    best_epoch: int,
    model,
    optimizer,
    scheduler,
    scaler,
    config: TrainVisualConfig,
    model_config: Optional[VisualLSTMConfig],
    dataset_info: Dict[str, object],
    save_rng_state: bool,
    metrics: Optional[Dict[str, object]] = None,
    best_metrics: Optional[Dict[str, object]] = None,
) -> Dict[str, Any]:
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_metric": best_val_metric,
        "best_epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": asdict(config),
        "model_config": asdict(model_config) if model_config is not None else None,
        "dataset_info": dataset_info,
        "rng_state": collect_rng_state() if save_rng_state else None,
        "best_metrics": best_metrics,
    }
    if metrics is not None:
        payload["metrics"] = metrics
    return payload


def save_checkpoint(checkpoint_path: Path, payload: Dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def validate_resume_compatibility(
    checkpoint_payload: Dict[str, Any],
    config: TrainVisualConfig,
    *,
    strict: bool,
) -> None:
    saved_args = checkpoint_payload.get("args", {})
    if not saved_args:
        return

    keys_to_compare = [
        "dataset_root",
        "val_dataset_root",
        "camera_name",
        "include_reset_frame",
        "cnn_feature_dim",
        "token_embedding_dim",
        "hidden_dim",
        "num_layers",
        "dropout",
        "input_image_size",
        "use_attention",
        "attention_dim",
        "attention_type",
        "attention_temperature",
        "local_attention",
        "noisy_attention",
        "memory_decay",
        "capacity_gate",
        "response_suppression",
        "use_step_embedding",
        "max_decode_steps",
        "step_embedding_dim",
        "delay_mode",
        "delay_steps",
        "blank_feature_mode",
    ]
    mismatches = []
    current_args = asdict(config)
    for key in keys_to_compare:
        saved_value = saved_args.get(key)
        current_value = current_args.get(key)
        if saved_value != current_value:
            mismatches.append((key, saved_value, current_value))

    if not mismatches:
        return

    mismatch_text = "; ".join(
        f"{key}: checkpoint={saved_value!r}, current={current_value!r}"
        for key, saved_value, current_value in mismatches
    )
    if strict:
        raise ValueError(f"Resume compatibility check failed: {mismatch_text}")
    print(f"[resume warning] Compatibility mismatches detected: {mismatch_text}")


def resume_training_state(
    *,
    checkpoint_payload: Dict[str, Any],
    model,
    optimizer,
    scheduler,
    scaler,
    config: TrainVisualConfig,
) -> Dict[str, Any]:
    validate_resume_compatibility(checkpoint_payload, config, strict=config.strict_resume)
    model.load_state_dict(checkpoint_payload["model_state_dict"])

    optimizer_state = checkpoint_payload.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint_payload.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint_payload.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    restore_rng_state(checkpoint_payload.get("rng_state"))

    saved_epoch = int(checkpoint_payload.get("epoch", 0))
    start_epoch = saved_epoch + 1
    if config.epochs < start_epoch:
        raise ValueError(
            f"Cannot resume: configured epochs={config.epochs} is less than start_epoch={start_epoch}"
        )

    return {
        "start_epoch": start_epoch,
        "global_step": int(checkpoint_payload.get("global_step", 0)),
        "best_metric": float(checkpoint_payload.get("best_val_metric", -1.0)),
        "best_epoch": int(checkpoint_payload.get("best_epoch", 0)),
        "best_metrics": dict(checkpoint_payload.get("best_metrics", {})),
    }


def load_model_only(model, checkpoint_ref: str | Path) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_ref)
    if checkpoint_path.is_dir():
        best_path = get_best_checkpoint_path(checkpoint_path)
        if best_path.exists():
            checkpoint_path = best_path
        else:
            checkpoint_path = get_last_checkpoint_path(checkpoint_path)
    payload = load_checkpoint(checkpoint_path)
    model.load_state_dict(payload["model_state_dict"])
    return payload


def save_config(output_dir: Path, config: TrainVisualConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    for filename in ("train_config.json", "config.json"):
        with open(output_dir / filename, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def append_epoch_log(output_dir: Path, record: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("epoch_logs.jsonl", "train_log.jsonl"):
        with open(output_dir / filename, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def normalize_error_breakdown(breakdown: Dict[str, Any]) -> Dict[str, float]:
    normalized = {
        "correct": 0.0,
        "wrong_block": 0.0,
        "order_error": 0.0,
        "repeat_error": 0.0,
    }
    for key, value in breakdown.items():
        if key == "repeated_block":
            normalized["repeat_error"] = float(value)
        elif key in normalized:
            normalized[key] = float(value)
    return normalized


def build_metrics_payload(
    *,
    config: TrainVisualConfig,
    output_dir: Path,
    epoch: int,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    per_length = metrics.get("per_length", {})
    error_breakdown = normalize_error_breakdown(metrics.get("error_breakdown", {}))
    error_breakdown_by_length = {
        str(length): normalize_error_breakdown(stats)
        for length, stats in metrics.get("error_breakdown_by_length", {}).items()
    }
    full_sequence_accuracy_by_length = {
        str(length): float(stats.get("full_seq_acc", 0.0))
        for length, stats in per_length.items()
    }
    token_accuracy_by_length = {
        str(length): float(stats.get("token_acc", 0.0))
        for length, stats in per_length.items()
    }
    return round_nested(
        {
            "experiment_name": output_dir.name,
            "seed": int(config.seed),
            "use_attention": bool(config.use_attention),
            "attention_type": config.attention_type,
            "use_step_embedding": bool(config.use_step_embedding),
            "use_scheduled_sampling": bool(config.scheduled_sampling),
            "best_epoch": int(epoch),
            "best_full_sequence_accuracy": float(metrics.get("full_sequence_accuracy", 0.0)),
            "best_token_accuracy": float(metrics.get("token_accuracy", 0.0)),
            "estimated_span": int(metrics.get("estimated_span", 0)),
            "full_sequence_accuracy_by_length": full_sequence_accuracy_by_length,
            "token_accuracy_by_length": token_accuracy_by_length,
            "error_breakdown": error_breakdown,
            "error_breakdown_by_length": error_breakdown_by_length,
            "order_error": error_breakdown["order_error"],
            "wrong_block": error_breakdown["wrong_block"],
            "repeat_error": float(metrics.get("error_analysis", {}).get("repeat_error_rate", 0.0)),
            "repeat_error_rate_by_length": metrics.get("repeat_error_rate_by_length", {}),
            "transposition_like_rate_by_length": metrics.get("transposition_like_rate_by_length", {}),
            "mean_first_error_pos_by_length": metrics.get("mean_first_error_pos_by_length", {}),
            "first_error_pos_distribution_by_length": {
                str(length): {} for length in range(min(LENGTH_RANGE), max(LENGTH_RANGE) + 1)
            },
            "raw_metrics": metrics,
        }
    )


def save_metrics_snapshot(
    *,
    output_dir: Path,
    filename: str,
    config: TrainVisualConfig,
    epoch: int,
    metrics: Dict[str, Any],
) -> None:
    payload = build_metrics_payload(config=config, output_dir=output_dir, epoch=epoch, metrics=metrics)
    with open(output_dir / filename, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_results_summary(output_dir: Path, summary: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(output_dir / "final_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def dataset_summary(dataset) -> Dict[str, object]:
    if isinstance(dataset, Subset):
        base = dataset.dataset
        summary = dataset_summary(base)
        summary["subset_size"] = len(dataset)
        summary["subset_indices"] = {
            "count": len(dataset.indices),
        }
        return summary

    if isinstance(dataset, RobosuiteVisualCorsiDataset):
        return {
            "dataset_root": str(dataset.dataset_root),
            "dataset_name": dataset.dataset_name,
            "split_name": dataset.split_name,
            "num_samples": len(dataset),
            "camera_names": dataset.camera_names,
            "camera_name_selected": dataset.camera_name,
            "include_reset_frame": dataset.include_reset_frame,
            "root_manifest": dataset.root_manifest,
        }

    return {
        "dataset_type": type(dataset).__name__,
        "num_samples": len(dataset),
    }


def build_dataset_info(train_dataset, val_dataset) -> Dict[str, object]:
    return {
        "train_dataset": dataset_summary(train_dataset),
        "val_dataset": dataset_summary(val_dataset),
    }


def save_dataset_info(output_dir: Path, train_dataset, val_dataset) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_dataset_info(train_dataset, val_dataset)
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def build_datasets(config: TrainVisualConfig):
    train_dataset = RobosuiteVisualCorsiDataset(
        config.dataset_root,
        camera_name=config.camera_name,
        include_reset_frame=config.include_reset_frame,
    )
    if config.val_dataset_root:
        val_dataset = RobosuiteVisualCorsiDataset(
            config.val_dataset_root,
            camera_name=config.camera_name,
            include_reset_frame=config.include_reset_frame,
        )
        return train_dataset, val_dataset
    return split_dataset(train_dataset, config.val_ratio, config.seed)


def main() -> None:
    config = parse_args()
    if config.resume and config.init_model:
        raise ValueError("--resume and --init_model cannot be used together")
    if config.delay_steps < 0:
        raise ValueError("--delay-steps must be >= 0")
    if config.max_decode_steps < 1:
        raise ValueError("--max-decode-steps must be >= 1")
    if config.step_embedding_dim < 1 and config.use_step_embedding:
        raise ValueError("--step-embedding-dim must be >= 1 when step embedding is enabled")
    if config.attention_dim < 1 and config.use_attention:
        raise ValueError("--attention-dim must be >= 1 when attention is enabled")
    if config.scheduled_sampling_warmup_epochs < 0:
        raise ValueError("--scheduled-sampling-warmup-epochs must be >= 0")
    if config.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be >= 0")
    if not 0.0 <= config.scheduled_sampling_start <= 1.0:
        raise ValueError("--scheduled-sampling-start must be in [0, 1]")
    if not 0.0 <= config.scheduled_sampling_end <= 1.0:
        raise ValueError("--scheduled-sampling-end must be in [0, 1]")
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    if not config.analysis_output_dir:
        config.analysis_output_dir = str(output_dir / "analysis")
    analysis_output_dir = Path(config.analysis_output_dir)

    device, device_info = resolve_torch_device(config.device)
    config.device = str(device)
    checkpoint_dir = build_checkpoint_dir(config)
    save_config(output_dir, config)
    with open(output_dir / "device_info.json", "w", encoding="utf-8") as handle:
        json.dump(device_info, handle, indent=2)
    print(json.dumps(device_info))

    train_dataset, val_dataset = build_datasets(config)
    dataset_info = save_dataset_info(output_dir, train_dataset, val_dataset)
    train_loader = build_dataloader(train_dataset, batch_size=config.batch_size, shuffle=True, seed=config.seed)
    val_loader = build_dataloader(val_dataset, batch_size=config.batch_size, shuffle=False, seed=config.seed + 1)

    model_config = build_model_config(config)
    model = VisualSeq2SeqLSTM(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = None
    scaler = None
    loss_fn = nn.CrossEntropyLoss(ignore_index=model_config.target_pad_value)

    resume_path = get_last_checkpoint_path(checkpoint_dir)
    start_epoch = 1
    global_step = 0
    best_metric = -1.0
    best_epoch = 0
    best_metrics: Dict[str, object] = {}
    epochs_without_improvement = 0

    if config.resume:
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume requested but no checkpoint found at {resume_path}")
        print(f"[mode] resume from {resume_path}")
        resume_state = resume_training_state(
            checkpoint_payload=load_checkpoint(resume_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
        )
        start_epoch = resume_state["start_epoch"]
        global_step = resume_state["global_step"]
        best_metric = resume_state["best_metric"]
        best_epoch = resume_state["best_epoch"]
        best_metrics = resume_state["best_metrics"]
    elif config.auto_resume and resume_path.exists():
        print(f"[mode] auto_resume from {resume_path}")
        resume_state = resume_training_state(
            checkpoint_payload=load_checkpoint(resume_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
        )
        start_epoch = resume_state["start_epoch"]
        global_step = resume_state["global_step"]
        best_metric = resume_state["best_metric"]
        best_epoch = resume_state["best_epoch"]
        best_metrics = resume_state["best_metrics"]
    elif config.init_model:
        print(f"[mode] init_model from {config.init_model}")
        load_model_only(model, config.init_model)
    else:
        print("[mode] scratch")

    for epoch in range(start_epoch, config.epochs + 1):
        teacher_forcing_ratio = compute_teacher_forcing_ratio(config, epoch)
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            training=True,
            delay_mode=config.delay_mode,
            delay_steps=config.delay_steps,
            blank_feature_mode=config.blank_feature_mode,
            teacher_forcing_ratio=teacher_forcing_ratio,
            global_step=global_step,
        )
        global_step = int(train_stats["global_step"])
        val_metrics = evaluate_model(
            model,
            val_loader,
            loss_fn,
            device,
            delay_mode=config.delay_mode,
            delay_steps=config.delay_steps,
            blank_feature_mode=config.blank_feature_mode,
            return_hidden_traces=config.return_hidden_traces,
            analysis_output_dir=analysis_output_dir,
            epoch=epoch,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(float(train_stats["loss"]), 6),
            "val_loss": round(float(val_metrics["loss"]), 6),
            "token_acc": round(float(val_metrics["token_accuracy"]), 6),
            "full_seq_acc": round(float(val_metrics["full_sequence_accuracy"]), 6),
            "token_accuracy": round(float(val_metrics["token_accuracy"]), 6),
            "full_sequence_accuracy": round(float(val_metrics["full_sequence_accuracy"]), 6),
            "delay_mode": config.delay_mode,
            "delay_steps": int(config.delay_steps),
            "blank_feature_mode": config.blank_feature_mode,
            "hidden_dim": int(config.hidden_dim),
            "num_layers": int(config.num_layers),
            "use_attention": bool(config.use_attention),
            "use_step_embedding": bool(config.use_step_embedding),
            "scheduled_sampling": bool(config.scheduled_sampling),
            "teacher_forcing_ratio": round(float(teacher_forcing_ratio), 6),
            "estimated_span": int(val_metrics["estimated_span"]),
            "per_length": round_nested(val_metrics["per_length"]),
            "serial_position_acc": round_nested(val_metrics["serial_position_acc"]),
            "serial_position_accuracy_by_length": round_nested(
                val_metrics.get("serial_position_accuracy_by_length", {})
            ),
            "error_analysis": round_nested(val_metrics["error_analysis"]),
            "error_analysis_by_length": round_nested(val_metrics.get("error_analysis_by_length", {})),
            "error_breakdown": round_nested(val_metrics["error_breakdown"]),
            "error_breakdown_by_length": round_nested(val_metrics.get("error_breakdown_by_length", {})),
            "mean_first_error_pos_by_length": round_nested(
                val_metrics.get("mean_first_error_pos_by_length", {})
            ),
            "transposition_like_rate_by_length": round_nested(
                val_metrics.get("transposition_like_rate_by_length", {})
            ),
            "repeat_error_rate_by_length": round_nested(
                val_metrics.get("repeat_error_rate_by_length", {})
            ),
        }
        if "hidden_trace_path" in val_metrics:
            epoch_record["hidden_trace_path"] = val_metrics["hidden_trace_path"]
            epoch_record["hidden_trace_sample_count"] = int(val_metrics.get("hidden_trace_sample_count", 0))

        print(json.dumps(epoch_record))
        append_epoch_log(output_dir, epoch_record)
        save_metrics_snapshot(
            output_dir=output_dir,
            filename="metrics_last.json",
            config=config,
            epoch=epoch,
            metrics=val_metrics,
        )

        last_checkpoint = make_checkpoint(
            epoch=epoch,
            global_step=global_step,
            best_val_metric=best_metric,
            best_epoch=best_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            model_config=model_config,
            dataset_info=dataset_info,
            save_rng_state=config.save_rng_state,
            metrics=val_metrics,
            best_metrics=best_metrics,
        )
        save_checkpoint(get_last_checkpoint_path(checkpoint_dir), last_checkpoint)

        if float(val_metrics["full_sequence_accuracy"]) > best_metric:
            best_metric = float(val_metrics["full_sequence_accuracy"])
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            epochs_without_improvement = 0
            best_checkpoint = make_checkpoint(
                epoch=epoch,
                global_step=global_step,
                best_val_metric=best_metric,
                best_epoch=best_epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                model_config=model_config,
                dataset_info=dataset_info,
                save_rng_state=config.save_rng_state,
                metrics=val_metrics,
                best_metrics=best_metrics,
            )
            save_checkpoint(get_best_checkpoint_path(checkpoint_dir), best_checkpoint)
            with open(output_dir / "best_metrics.json", "w", encoding="utf-8") as handle:
                json.dump(round_nested(val_metrics), handle, indent=2)
            save_metrics_snapshot(
                output_dir=output_dir,
                filename="metrics_best.json",
                config=config,
                epoch=epoch,
                metrics=val_metrics,
            )
        else:
            epochs_without_improvement += 1

        if config.early_stopping_patience > 0 and epochs_without_improvement >= config.early_stopping_patience:
            print(
                json.dumps(
                    {
                        "event": "early_stopping",
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_full_sequence_accuracy": round(best_metric, 6),
                        "patience": int(config.early_stopping_patience),
                    }
                )
            )
            break

    summary = round_nested(
        {
            "best_epoch": best_epoch,
            "best_full_seq_acc": best_metric,
            "best_full_sequence_accuracy": best_metric,
            "best_token_acc": float(best_metrics.get("token_accuracy", 0.0)),
            "best_token_accuracy": float(best_metrics.get("token_accuracy", 0.0)),
            "best_per_length": best_metrics.get("per_length", {}),
            "best_serial_position_acc": best_metrics.get("serial_position_acc", {}),
            "best_error_analysis": best_metrics.get("error_analysis", {}),
            "delay_mode": config.delay_mode,
            "delay_steps": int(config.delay_steps),
            "blank_feature_mode": config.blank_feature_mode,
            "hidden_dim": int(config.hidden_dim),
            "num_layers": int(config.num_layers),
            "use_attention": bool(config.use_attention),
            "attention_dim": int(config.attention_dim),
            "attention_type": config.attention_type,
            "attention_temperature": float(config.attention_temperature),
            "local_attention": config.local_attention or {},
            "noisy_attention": config.noisy_attention or {},
            "memory_decay": config.memory_decay or {},
            "capacity_gate": config.capacity_gate or {},
            "response_suppression": config.response_suppression or {},
            "use_step_embedding": bool(config.use_step_embedding),
            "step_embedding_dim": int(config.step_embedding_dim),
            "scheduled_sampling": bool(config.scheduled_sampling),
            "scheduled_sampling_start": float(config.scheduled_sampling_start),
            "scheduled_sampling_end": float(config.scheduled_sampling_end),
            "scheduled_sampling_warmup_epochs": int(config.scheduled_sampling_warmup_epochs),
            "early_stopping_patience": int(config.early_stopping_patience),
            "checkpoint_path": str(get_best_checkpoint_path(checkpoint_dir)),
            "analysis_output_dir": str(analysis_output_dir),
            "best_metrics": best_metrics,
        }
    )
    save_results_summary(output_dir, summary)


if __name__ == "__main__":
    main()
