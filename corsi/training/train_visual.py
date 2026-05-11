"""Training entry point for the visual robosuite Corsi sequence model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from corsi.analysis.metrics import summarize_sequence_metrics
from corsi.data import RobosuiteVisualCorsiDataset, collate_visual_batch
from corsi.heatmaps import nearest_block_decode, spatial_heatmap_loss, standard_block_norm_xy
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
    train_length_filter: Optional[list[int]] = None
    val_length_filter: Optional[list[int]] = None
    batch_size: int = 4
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float = 0.0
    cnn_feature_dim: int = 128
    token_embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    input_image_size: int = 128
    output_type: str = "index"
    target_type: str = "block_index"
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
    use_encoder_summary_input: bool = False
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
    heatmap_size: int = 32
    heatmap_sigma: float = 2.0
    heatmap_normalize: bool = True
    heatmap_loss: str = "spatial_ce"
    heatmap_decode_method: str = "argmax"
    heatmap_visualization_samples: int = 4
    xy_loss: str = "smooth_l1"
    xy_within_radius_norm: float = 0.1
    xy_visualization_samples: int = 8
    export_validation_hidden_states: bool = False
    hidden_state_export_max_batches: int = 0
    overfit_subset_size: int = 0
    max_train_samples: int = 0
    val_same_as_train: bool = False
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
        for key in (
            "output_type",
            "use_attention",
            "use_step_embedding",
            "use_encoder_summary_input",
            "max_decode_steps",
            "step_embedding_dim",
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
            "overfit_subset_size": "overfit_subset_size",
            "max_train_samples": "max_train_samples",
            "val_same_as_train": "val_same_as_train",
            "length_filter": "length_filter",
            "max_epochs": "epochs",
            "epochs": "epochs",
            "lr": "learning_rate",
            "learning_rate": "learning_rate",
            "weight_decay": "weight_decay",
            "gradient_clip_norm": "gradient_clip_norm",
            "seed": "seed",
        }
        for source_key, target_key in alias_map.items():
            if source_key in training_overrides:
                normalized[target_key] = training_overrides[source_key]

    if "length_filter" in normalized:
        normalized["train_length_filter"] = normalized["length_filter"]
        normalized["val_length_filter"] = normalized.pop("length_filter")

    target_overrides = normalized.pop("target", None)
    if isinstance(target_overrides, dict) and "type" in target_overrides:
        normalized["target_type"] = target_overrides["type"]

    xy_overrides = normalized.pop("xy", None)
    if isinstance(xy_overrides, dict):
        alias_map = {
            "loss": "xy_loss",
            "within_radius_norm": "xy_within_radius_norm",
            "visualization_samples": "xy_visualization_samples",
            "export_validation_hidden_states": "export_validation_hidden_states",
            "hidden_state_export_max_batches": "hidden_state_export_max_batches",
        }
        for source_key, target_key in alias_map.items():
            if source_key in xy_overrides:
                normalized[target_key] = xy_overrides[source_key]

    heatmap_overrides = normalized.pop("heatmap", None)
    if isinstance(heatmap_overrides, dict):
        alias_map = {
            "size": "heatmap_size",
            "sigma": "heatmap_sigma",
            "normalize": "heatmap_normalize",
            "loss": "heatmap_loss",
            "decode_method": "heatmap_decode_method",
        }
        for source_key, target_key in alias_map.items():
            if source_key in heatmap_overrides:
                normalized[target_key] = heatmap_overrides[source_key]

    if "use_scheduled_sampling" in normalized:
        normalized["scheduled_sampling"] = normalized.pop("use_scheduled_sampling")

    if "max_epochs" in normalized:
        normalized["epochs"] = normalized.pop("max_epochs")
    if "lr" in normalized:
        normalized["learning_rate"] = normalized.pop("lr")

    if normalized.get("attention_type", "global") != "global":
        normalized.setdefault("use_attention", True)

    return normalized


def parse_simple_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return json.loads(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def load_simple_yaml_config(text: str) -> Dict[str, Any]:
    """Minimal YAML reader for the flat/nested scalar config files used here."""

    root: Dict[str, Any] = {}
    current_parent: Dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key, value = line.strip().split(":", 1)
        if indent == 0:
            if value.strip():
                root[key] = parse_simple_yaml_scalar(value)
                current_parent = None
            else:
                current_parent = {}
                root[key] = current_parent
        elif indent == 2 and current_parent is not None:
            current_parent[key] = parse_simple_yaml_scalar(value)
        else:
            raise ValueError(f"Unsupported YAML indentation: {raw_line}")
    return root


def load_config_overrides(config_path: str) -> Dict[str, object]:
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ModuleNotFoundError:
                data = load_simple_yaml_config(handle.read())
            else:
                data = yaml.safe_load(handle)
        else:
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
    parser.add_argument("--train-length-filter", type=json.loads, default=defaults["train_length_filter"])
    parser.add_argument("--val-length-filter", type=json.loads, default=defaults["val_length_filter"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--gradient-clip-norm", type=float, default=defaults["gradient_clip_norm"])
    parser.add_argument("--cnn-feature-dim", type=int, default=defaults["cnn_feature_dim"])
    parser.add_argument("--token-embedding-dim", type=int, default=defaults["token_embedding_dim"])
    parser.add_argument("--hidden-dim", type=int, default=defaults["hidden_dim"])
    parser.add_argument("--num-layers", type=int, default=defaults["num_layers"])
    parser.add_argument("--dropout", type=float, default=defaults["dropout"])
    parser.add_argument("--input-image-size", type=int, default=defaults["input_image_size"])
    parser.add_argument("--output-type", type=str, default=defaults["output_type"], choices=["index", "heatmap", "xy"])
    parser.add_argument(
        "--target-type",
        type=str,
        default=defaults["target_type"],
        choices=["block_index", "block_center_xy", "end_effector_xy"],
    )
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
    parser.add_argument(
        "--use-encoder-summary-input",
        action=argparse.BooleanOptionalAction,
        default=defaults["use_encoder_summary_input"],
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
    parser.add_argument("--heatmap-size", type=int, default=defaults["heatmap_size"])
    parser.add_argument("--heatmap-sigma", type=float, default=defaults["heatmap_sigma"])
    parser.add_argument(
        "--heatmap-normalize",
        action=argparse.BooleanOptionalAction,
        default=defaults["heatmap_normalize"],
    )
    parser.add_argument("--heatmap-loss", type=str, default=defaults["heatmap_loss"], choices=["spatial_ce", "mse"])
    parser.add_argument(
        "--heatmap-decode-method",
        type=str,
        default=defaults["heatmap_decode_method"],
        choices=["argmax"],
    )
    parser.add_argument(
        "--heatmap-visualization-samples",
        type=int,
        default=defaults["heatmap_visualization_samples"],
    )
    parser.add_argument("--xy-loss", type=str, default=defaults["xy_loss"], choices=["smooth_l1", "mse"])
    parser.add_argument("--xy-within-radius-norm", type=float, default=defaults["xy_within_radius_norm"])
    parser.add_argument("--xy-visualization-samples", type=int, default=defaults["xy_visualization_samples"])
    parser.add_argument(
        "--export-validation-hidden-states",
        action=argparse.BooleanOptionalAction,
        default=defaults["export_validation_hidden_states"],
    )
    parser.add_argument(
        "--hidden-state-export-max-batches",
        type=int,
        default=defaults["hidden_state_export_max_batches"],
    )
    parser.add_argument("--overfit-subset-size", type=int, default=defaults["overfit_subset_size"])
    parser.add_argument("--max-train-samples", type=int, default=defaults["max_train_samples"])
    parser.add_argument(
        "--val-same-as-train",
        action=argparse.BooleanOptionalAction,
        default=defaults["val_same_as_train"],
    )
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
        output_type=train_config.output_type,
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
        use_encoder_summary_input=train_config.use_encoder_summary_input,
        max_decode_steps=train_config.max_decode_steps,
        step_embedding_dim=train_config.step_embedding_dim,
        delay_mode=train_config.delay_mode,
        delay_steps=train_config.delay_steps,
        blank_feature_mode=train_config.blank_feature_mode,
        return_hidden_traces=train_config.return_hidden_traces,
        return_error_analysis=train_config.return_error_analysis,
        heatmap_size=train_config.heatmap_size,
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
    if batch.get("target_heatmaps") is not None:
        moved["target_heatmaps"] = batch["target_heatmaps"].to(device)
    for key in (
        "target_xy",
        "target_block_xy_norm",
        "ee_xy_norm",
        "ee_xyz_world",
        "target_block_indices",
    ):
        if batch.get(key) is not None:
            moved[key] = batch[key].to(device)
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


def filter_dataset_by_lengths(dataset, lengths: Optional[list[int] | tuple[int, ...]]) -> Subset | Any:
    if not lengths:
        return dataset
    allowed = {int(length) for length in lengths}
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = [
            index for index in dataset.indices
            if int(base.samples[int(index)]["length"]) in allowed
        ]
        return Subset(base, indices)
    if not isinstance(dataset, RobosuiteVisualCorsiDataset):
        return dataset
    indices = [
        index for index, sample in enumerate(dataset.samples)
        if int(sample["length"]) in allowed
    ]
    if not indices:
        raise ValueError(f"Length filter {sorted(allowed)} selected no samples from {dataset.dataset_root}")
    return Subset(dataset, indices)


def compute_loss(
    *,
    config: TrainVisualConfig,
    outputs: Dict[str, Any],
    batch: Dict[str, object],
    loss_fn,
) -> torch.Tensor:
    if config.output_type == "index":
        logits = outputs["logits"]
        return loss_fn(logits.reshape(-1, logits.size(-1)), batch["targets"].reshape(-1))
    if config.output_type == "heatmap":
        return spatial_heatmap_loss(
            outputs["heatmap_logits"],
            batch["target_heatmaps"],
            batch["mask"],
            loss_type=config.heatmap_loss,
        )
    if "target_xy" not in batch:
        raise ValueError(
            f"output_type='xy' requires target_xy. Check target.type={config.target_type!r} "
            "and that the dataset manifests contain step_metadata."
        )
    pred_xy = outputs["pred_xy"]
    target_xy = batch["target_xy"]
    if pred_xy is None:
        raise ValueError("Model output_type='xy' did not return pred_xy")
    if config.xy_loss == "smooth_l1":
        token_loss = F.smooth_l1_loss(pred_xy, target_xy, reduction="none").mean(dim=-1)
    elif config.xy_loss == "mse":
        token_loss = F.mse_loss(pred_xy, target_xy, reduction="none").mean(dim=-1)
    else:
        raise ValueError(f"Unsupported xy_loss: {config.xy_loss}")
    active_mask = batch["mask"].to(dtype=token_loss.dtype)
    return (token_loss * active_mask).sum() / active_mask.sum().clamp_min(1.0)


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
    config: TrainVisualConfig,
    teacher_forcing_ratio: float = 1.0,
    global_step: int = 0,
) -> Dict[str, float]:
    model.train(training)
    loss_total = 0.0
    token_count = 0
    pred_xy_batches = []
    target_xy_batches = []
    mask_batches = []

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
        loss = compute_loss(config=config, outputs=outputs, batch=batch, loss_fn=loss_fn)
        if config.output_type == "xy":
            pred_xy_batches.append(outputs["pred_xy"].detach().cpu())
            target_xy_batches.append(batch["target_xy"].detach().cpu())
            mask_batches.append(batch["mask"].detach().cpu())

        if training:
            if optimizer is None:
                raise ValueError("Optimizer is required when training=True")
            optimizer.zero_grad()
            loss.backward()
            if config.gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip_norm))
            optimizer.step()
            global_step += 1

        active_tokens = int(batch["mask"].sum().item())
        loss_total += float(loss.item()) * active_tokens
        token_count += active_tokens

    stats = {
        "loss": loss_total / token_count if token_count > 0 else 0.0,
        "global_step": global_step,
    }
    if config.output_type == "xy" and pred_xy_batches:
        max_steps = max(tensor.size(1) for tensor in pred_xy_batches)
        pred_xy = torch.cat(
            [pad_to_max_steps(tensor, max_steps=max_steps, fill_value=0.0) for tensor in pred_xy_batches],
            dim=0,
        )
        target_xy = torch.cat(
            [pad_to_max_steps(tensor, max_steps=max_steps, fill_value=0.0) for tensor in target_xy_batches],
            dim=0,
        )
        mask = torch.cat(
            [pad_to_max_steps(tensor, max_steps=max_steps, fill_value=False) for tensor in mask_batches],
            dim=0,
        )
        stats.update(masked_xy_distribution("pred_xy", pred_xy, mask))
        stats.update(masked_xy_distribution("target_xy", target_xy, mask))
    return stats


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


def pad_to_max_steps(tensor, *, max_steps: int, fill_value):
    if tensor.size(1) == max_steps:
        return tensor
    pad_width = max_steps - tensor.size(1)
    pad_shape = (tensor.size(0), pad_width, *tensor.shape[2:])
    padding = torch.full(pad_shape, fill_value, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=1)


def heatmap_spatial_metrics(pred_xy, targets, mask, block_xy, *, radius: float) -> Dict[str, float]:
    target_xy = block_xy[targets.clamp(0, block_xy.size(0) - 1)]
    errors = torch.linalg.norm(pred_xy.float() - target_xy.float(), dim=-1)
    active_errors = errors[mask]
    if int(active_errors.numel()) == 0:
        return {
            "mean_spatial_error": 0.0,
            "median_spatial_error": 0.0,
            "within_radius_accuracy": 0.0,
            "within_2px_acc": 0.0,
            "within_4px_acc": 0.0,
            "within_8px_acc": 0.0,
        }
    return {
        "mean_spatial_error": float(active_errors.mean().item()),
        "median_spatial_error": float(active_errors.median().item()),
        "within_radius_accuracy": float((active_errors <= float(radius)).float().mean().item()),
        "within_2px_acc": float((active_errors <= 2.0).float().mean().item()),
        "within_4px_acc": float((active_errors <= 4.0).float().mean().item()),
        "within_8px_acc": float((active_errors <= 8.0).float().mean().item()),
    }


def heatmap_entropy(heatmap_logits: torch.Tensor) -> torch.Tensor:
    logits = heatmap_logits.squeeze(2)
    flat_logits = logits.flatten(start_dim=-2)
    probabilities = F.softmax(flat_logits, dim=-1)
    log_probabilities = F.log_softmax(flat_logits, dim=-1)
    return -(probabilities * log_probabilities).sum(dim=-1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    active_values = values[mask]
    if int(active_values.numel()) == 0:
        return 0.0
    return float(active_values.float().mean().item())


def _masked_median(values: torch.Tensor, mask: torch.Tensor) -> float:
    active_values = values[mask]
    if int(active_values.numel()) == 0:
        return 0.0
    return float(active_values.float().median().item())


def masked_xy_distribution(prefix: str, xy: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    active_xy = xy[mask].float()
    if int(active_xy.numel()) == 0:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
        }
    return {
        f"{prefix}_min": float(active_xy.min().item()),
        f"{prefix}_max": float(active_xy.max().item()),
        f"{prefix}_mean": float(active_xy.mean().item()),
        f"{prefix}_std": float(active_xy.std(unbiased=False).item()),
    }


def length_bounds_present(target_lengths: torch.Tensor) -> tuple[int, int]:
    if int(target_lengths.numel()) == 0:
        return min(LENGTH_RANGE), min(LENGTH_RANGE)
    return int(target_lengths.min().item()), int(target_lengths.max().item())


def heatmap_per_length_metrics(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    mask: torch.Tensor,
    spatial_errors: torch.Tensor,
    entropy: torch.Tensor,
    min_length: int,
    max_length: int,
) -> Dict[str, Dict[str, float]]:
    sequence_correct = ((predictions == targets) | (~mask)).all(dim=1)
    per_length: Dict[str, Dict[str, float]] = {}
    for length in range(min_length, max_length + 1):
        length_mask = target_lengths == length
        if int(length_mask.sum().item()) == 0:
            per_length[str(length)] = {
                "token_acc": 0.0,
                "full_seq_acc": 0.0,
                "mean_spatial_error": 0.0,
                "median_spatial_error": 0.0,
                "heatmap_entropy": 0.0,
            }
            continue
        token_mask = mask[length_mask]
        per_length[str(length)] = {
            "token_acc": float(
                (((predictions[length_mask] == targets[length_mask]) & token_mask).sum().item())
                / max(1, int(token_mask.sum().item()))
            ),
            "full_seq_acc": float(sequence_correct[length_mask].float().mean().item()),
            "mean_spatial_error": _masked_mean(spatial_errors[length_mask], token_mask),
            "median_spatial_error": _masked_median(spatial_errors[length_mask], token_mask),
            "heatmap_entropy": _masked_mean(entropy[length_mask], token_mask),
        }
    return per_length


def heatmap_recall_position_metrics(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    spatial_errors: torch.Tensor,
    entropy: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    for step_index in range(mask.size(1)):
        step_mask = mask[:, step_index]
        active = int(step_mask.sum().item())
        if active == 0:
            continue
        step_correct = (predictions[:, step_index] == targets[:, step_index]) & step_mask
        rows[str(step_index + 1)] = {
            "token_acc": float(step_correct.sum().item() / active),
            "mean_spatial_error": _masked_mean(spatial_errors[:, step_index], step_mask),
            "heatmap_entropy": _masked_mean(entropy[:, step_index], step_mask),
        }
    return rows


def heatmap_spatial_error_histogram(spatial_errors: torch.Tensor, mask: torch.Tensor) -> Dict[str, int]:
    active_errors = spatial_errors[mask].float()
    if int(active_errors.numel()) == 0:
        return {"0_2px": 0, "2_4px": 0, "4_8px": 0, "8_16px": 0, "gt_16px": 0}
    return {
        "0_2px": int((active_errors <= 2.0).sum().item()),
        "2_4px": int(((active_errors > 2.0) & (active_errors <= 4.0)).sum().item()),
        "4_8px": int(((active_errors > 4.0) & (active_errors <= 8.0)).sum().item()),
        "8_16px": int(((active_errors > 8.0) & (active_errors <= 16.0)).sum().item()),
        "gt_16px": int((active_errors > 16.0).sum().item()),
    }


def heatmap_analysis_rows(metrics: Dict[str, Any]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    per_length_rows = [
        {"length": int(length), **stats}
        for length, stats in sorted(metrics.get("heatmap_per_length", {}).items(), key=lambda item: int(item[0]))
    ]
    recall_rows = [
        {"recall_position": int(position), **stats}
        for position, stats in sorted(metrics.get("heatmap_recall_position", {}).items(), key=lambda item: int(item[0]))
    ]
    histogram_rows = [
        {"bin": bin_name, "count": count}
        for bin_name, count in metrics.get("spatial_error_histogram", {}).items()
    ]
    return per_length_rows, recall_rows, histogram_rows


def block_norm_xy_from_mapping(mapping: Any) -> torch.Tensor:
    if isinstance(mapping, dict) and mapping:
        rows = []
        for block_index in sorted(int(key) for key in mapping.keys()):
            value = mapping[str(block_index)] if str(block_index) in mapping else mapping[block_index]
            rows.append([float(value[0]), float(value[1])])
        return torch.tensor(rows, dtype=torch.float32)
    return torch.as_tensor(standard_block_norm_xy(), dtype=torch.float32)


def table_delta_scale_from_xy_normalization(xy_normalization: Any) -> torch.Tensor:
    if isinstance(xy_normalization, dict):
        bounds = xy_normalization.get("bounds", {})
    else:
        bounds = {}
    if not isinstance(bounds, dict):
        bounds = {}
    x_min = float(bounds.get("x_min", 0.0))
    x_max = float(bounds.get("x_max", 2.0))
    y_min = float(bounds.get("y_min", 0.0))
    y_max = float(bounds.get("y_max", 2.0))
    return torch.tensor([(x_max - x_min) * 0.5, (y_max - y_min) * 0.5], dtype=torch.float32)


def xy_per_length_metrics(
    *,
    nearest_predictions: torch.Tensor,
    target_block_indices: torch.Tensor,
    target_lengths: torch.Tensor,
    mask: torch.Tensor,
    xy_errors_norm: torch.Tensor,
    xy_errors_table: torch.Tensor,
    within_radius: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    sequence_correct = ((nearest_predictions == target_block_indices) | (~mask)).all(dim=1)
    per_length: Dict[str, Dict[str, float]] = {}
    min_length = int(target_lengths.min().item()) if target_lengths.numel() else min(LENGTH_RANGE)
    max_length = int(target_lengths.max().item()) if target_lengths.numel() else max(LENGTH_RANGE)
    for length in range(min_length, max_length + 1):
        length_mask = target_lengths == length
        if int(length_mask.sum().item()) == 0:
            per_length[str(length)] = {
                "nearest_block_accuracy": 0.0,
                "full_sequence_nearest_block_accuracy": 0.0,
                "mean_xy_error_norm": 0.0,
                "median_xy_error_norm": 0.0,
                "mean_xy_error_table": 0.0,
                "median_xy_error_table": 0.0,
                "within_radius_acc": 0.0,
            }
            continue
        token_mask = mask[length_mask]
        correct = (nearest_predictions[length_mask] == target_block_indices[length_mask]) & token_mask
        per_length[str(length)] = {
            "nearest_block_accuracy": float(correct.sum().item() / max(1, int(token_mask.sum().item()))),
            "full_sequence_nearest_block_accuracy": float(sequence_correct[length_mask].float().mean().item()),
            "mean_xy_error_norm": _masked_mean(xy_errors_norm[length_mask], token_mask),
            "median_xy_error_norm": _masked_median(xy_errors_norm[length_mask], token_mask),
            "mean_xy_error_table": _masked_mean(xy_errors_table[length_mask], token_mask),
            "median_xy_error_table": _masked_median(xy_errors_table[length_mask], token_mask),
            "within_radius_acc": _masked_mean(within_radius[length_mask].float(), token_mask),
        }
    return per_length


def xy_metrics(
    *,
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    target_block_indices: torch.Tensor,
    target_lengths: torch.Tensor,
    mask: torch.Tensor,
    block_xy_norm: torch.Tensor,
    table_delta_scale: torch.Tensor,
    radius_norm: float,
) -> tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
    nearest_predictions, nearest_distances = nearest_block_decode(pred_xy, block_xy=block_xy_norm)
    xy_delta = pred_xy.float() - target_xy.float()
    xy_errors_norm = torch.linalg.norm(xy_delta, dim=-1)
    table_scale = table_delta_scale.to(dtype=xy_delta.dtype, device=xy_delta.device)
    xy_errors_table = torch.linalg.norm(xy_delta * table_scale, dim=-1)
    within_radius = xy_errors_norm <= float(radius_norm)
    active_errors_norm = xy_errors_norm[mask]
    active_errors_table = xy_errors_table[mask]
    active_within = within_radius[mask]
    nearest_correct = (nearest_predictions == target_block_indices) & mask
    sequence_correct = ((nearest_predictions == target_block_indices) | (~mask)).all(dim=1)
    active_count = max(1, int(mask.sum().item()))
    metrics = {
        "mean_xy_error_norm": float(active_errors_norm.mean().item()) if active_errors_norm.numel() else 0.0,
        "median_xy_error_norm": float(active_errors_norm.median().item()) if active_errors_norm.numel() else 0.0,
        "mean_xy_error_table": float(active_errors_table.mean().item()) if active_errors_table.numel() else 0.0,
        "median_xy_error_table": float(active_errors_table.median().item()) if active_errors_table.numel() else 0.0,
        "within_radius_acc": float(active_within.float().mean().item()) if active_within.numel() else 0.0,
        "within_radius_norm": float(radius_norm),
        "nearest_block_accuracy": float(nearest_correct.sum().item() / active_count),
        "full_sequence_nearest_block_accuracy": float(sequence_correct.float().mean().item())
        if sequence_correct.numel()
        else 0.0,
        "mean_nearest_block_distance": float(nearest_distances[mask].float().mean().item())
        if int(mask.sum().item())
        else 0.0,
        "xy_per_length": xy_per_length_metrics(
            nearest_predictions=nearest_predictions,
            target_block_indices=target_block_indices,
            target_lengths=target_lengths,
            mask=mask,
            xy_errors_norm=xy_errors_norm,
            xy_errors_table=xy_errors_table,
            within_radius=within_radius,
        ),
    }
    return metrics, nearest_predictions, xy_errors_norm


def save_xy_hidden_state_export(
    output_dir: Path,
    *,
    epoch: int,
    records: list[Dict[str, torch.Tensor | list[str]]],
    block_positions: torch.Tensor,
) -> Optional[Path]:
    if not records:
        return None
    export_dir = output_dir / "analysis" / "xy_hidden_states"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / f"epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "block_positions": block_positions.detach().cpu(),
            "records": records,
        },
        export_path,
    )
    return export_path


def _jsonable_tensor_rows(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> list:
    rows = tensor.detach().cpu().tolist()
    if mask is None:
        return rows
    mask_rows = mask.detach().cpu().tolist()
    result = []
    for sample_rows, sample_mask in zip(rows, mask_rows):
        result.append([
            value if bool(active) else None
            for value, active in zip(sample_rows, sample_mask)
        ])
    return result


@torch.no_grad()
def save_xy_batch_sanity_dump(
    *,
    model,
    loader,
    device,
    config: TrainVisualConfig,
    output_dir: Path,
    epoch: int,
    split_name: str,
) -> Optional[Path]:
    if config.output_type != "xy":
        return None
    try:
        raw_batch = next(iter(loader))
    except StopIteration:
        return None

    model_was_training = model.training
    model.eval()
    batch = move_batch_to_device(raw_batch, device)
    outputs = model(
        frames=batch["frames"],
        frame_lengths=batch["frame_lengths"],
        targets=batch["targets"],
        target_lengths=batch["target_lengths"],
        delay_mode=config.delay_mode,
        delay_steps=config.delay_steps,
        blank_feature_mode=config.blank_feature_mode,
        return_hidden_traces=False,
        teacher_forcing_ratio=1.0,
    )
    pred_xy = outputs["pred_xy"]
    target_xy = batch["target_xy"]
    mask = batch["mask"]
    block_xy_norm = block_norm_xy_from_mapping(
        batch.get("block_xy_norm", [{}])[0]
        if isinstance(batch.get("block_xy_norm"), list)
        else {}
    ).to(device)
    nearest_target, _ = nearest_block_decode(target_xy, block_xy=block_xy_norm)
    nearest_pred, _ = nearest_block_decode(pred_xy, block_xy=block_xy_norm)
    if config.xy_loss == "smooth_l1":
        loss_per_step = F.smooth_l1_loss(pred_xy, target_xy, reduction="none").mean(dim=-1)
    else:
        loss_per_step = F.mse_loss(pred_xy, target_xy, reduction="none").mean(dim=-1)
    loss_per_step = loss_per_step.masked_fill(~mask, float("nan"))

    payload = {
        "epoch": int(epoch),
        "split": split_name,
        "output_type": config.output_type,
        "target_type": config.target_type,
        "pred_xy_shape": list(pred_xy.shape),
        "target_xy_shape": list(target_xy.shape),
        "mask_true_count": int(mask.sum().item()),
        "mask_total_count": int(mask.numel()),
        "xy_coordinate_system": "corsi_lower_left_normalized",
        "xy_norm_range": batch.get("xy_normalization", [{}])[0].get("range", [-1.0, 1.0])
        if isinstance(batch.get("xy_normalization"), list) and batch.get("xy_normalization")
        else [-1.0, 1.0],
        "block_positions_norm": block_xy_norm.detach().cpu().tolist(),
        "trial_ids": list(batch["trial_ids"]),
        "sequence": _jsonable_tensor_rows(batch["targets"], mask),
        "target_block_index": _jsonable_tensor_rows(batch["target_block_indices"], mask),
        "target_xy": _jsonable_tensor_rows(target_xy, mask),
        "pred_xy": _jsonable_tensor_rows(pred_xy, mask),
        "nearest_block_target_xy": _jsonable_tensor_rows(nearest_target, mask),
        "nearest_block_pred_xy": _jsonable_tensor_rows(nearest_pred, mask),
        "mask": mask.detach().cpu().tolist(),
        "loss_per_step": _jsonable_tensor_rows(loss_per_step.detach().cpu(), mask.detach().cpu()),
    }
    dump_dir = output_dir / "xy_sanity" / f"epoch_{epoch:03d}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / f"{split_name}_first_batch.json"
    dump_path.write_text(json.dumps(round_nested(payload), indent=2), encoding="utf-8")
    if model_was_training:
        model.train(True)
    return dump_path


def save_xy_training_wiring_check(
    *,
    model,
    optimizer,
    train_loader,
    device,
    config: TrainVisualConfig,
    output_dir: Path,
) -> Path:
    raw_batch = next(iter(train_loader))
    batch = move_batch_to_device(raw_batch, device)
    model.eval()
    with torch.no_grad():
        outputs = model(
            frames=batch["frames"],
            frame_lengths=batch["frame_lengths"],
            targets=batch["targets"],
            target_lengths=batch["target_lengths"],
            delay_mode=config.delay_mode,
            delay_steps=config.delay_steps,
            blank_feature_mode=config.blank_feature_mode,
            return_hidden_traces=False,
            teacher_forcing_ratio=1.0,
        )
    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    xy_head_params = list(model.xy_head.parameters())
    xy_range = batch.get("xy_normalization", [{}])[0].get("range", [-1.0, 1.0]) if batch.get("xy_normalization") else [-1.0, 1.0]
    payload = {
        "output_type": config.output_type,
        "target_type": config.target_type,
        "uses_xy_branch": config.output_type == "xy" and outputs.get("pred_xy") is not None,
        "pred_xy_shape": list(outputs["pred_xy"].shape) if outputs.get("pred_xy") is not None else None,
        "target_xy_shape": list(batch["target_xy"].shape) if batch.get("target_xy") is not None else None,
        "loss_source": "target_xy",
        "mask_true_count": int(batch["mask"].sum().item()),
        "mask_total_count": int(batch["mask"].numel()),
        "xy_head_parameter_count": int(sum(param.numel() for param in xy_head_params)),
        "xy_head_in_optimizer": all(id(param) in optimizer_param_ids for param in xy_head_params),
        "xy_coordinate_system": "corsi_lower_left_normalized",
        "xy_norm_range": xy_range,
        "coordinate_consistency": {
            "target_xy": "corsi_lower_left_normalized",
            "pred_xy": "corsi_lower_left_normalized",
            "block_positions_norm": "corsi_lower_left_normalized",
            "nearest_block_accuracy": "computed by cdist(pred_xy, block_positions_norm)",
            "within_radius_acc": "computed on L2(pred_xy-target_xy) in normalized coordinates",
            "visualization": "maps normalized coordinates to image pixels via block-position affine",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "xy_training_wiring.json"
    path.write_text(json.dumps(round_nested(payload), indent=2), encoding="utf-8")
    return path


def write_rows_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap_analysis_tables(output_dir: Path, *, epoch: int, metrics: Dict[str, Any]) -> None:
    epoch_dir = output_dir / "analysis" / f"epoch_{epoch:03d}"
    per_length_rows, recall_rows, histogram_rows = heatmap_analysis_rows(metrics)
    write_rows_csv(epoch_dir / "per_length_metrics.csv", per_length_rows)
    write_rows_csv(epoch_dir / "recall_position_metrics.csv", recall_rows)
    write_rows_csv(epoch_dir / "spatial_error_histogram.csv", histogram_rows)
    with open(epoch_dir / "heatmap_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(round_nested(metrics), handle, indent=2, allow_nan=True)


def normalize_heatmap_image(heatmap: np.ndarray) -> np.ndarray:
    values = np.asarray(heatmap, dtype=np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value > min_value:
        values = (values - min_value) / (max_value - min_value)
    else:
        values = np.zeros_like(values)
    return (values * 255.0).clip(0, 255).astype(np.uint8)


def _is_order_drift(target_sequence: list[int], prediction_sequence: list[int], step_index: int) -> bool:
    if prediction_sequence == target_sequence:
        return False
    if Counter(prediction_sequence) == Counter(target_sequence):
        return True
    predicted_token = prediction_sequence[step_index]
    target_token = target_sequence[step_index]
    return predicted_token != target_token and predicted_token in target_sequence


def _heatmap_step_categories(
    *,
    target_sequence: list[int],
    prediction_sequence: list[int],
    step_index: int,
    spatial_error: float,
    entropy: float,
) -> list[str]:
    correct = prediction_sequence[step_index] == target_sequence[step_index]
    categories: list[str] = []
    if correct and entropy <= 4.5:
        categories.append("correct_confident")
    if not correct and spatial_error <= 4.0:
        categories.append("near_miss")
    if not correct and entropy >= 6.0:
        categories.append("diffuse_uncertain_failure")
    if _is_order_drift(target_sequence, prediction_sequence, step_index):
        categories.append("order_drift_failure")
    return categories


def save_heatmap_visualization_candidates(
    *,
    output_dir: Path,
    epoch: int,
    batch: Dict[str, object],
    decode_outputs: Dict[str, torch.Tensor],
    spatial_errors: torch.Tensor,
    entropy: torch.Tensor,
    max_per_category: int,
    saved_counts: Dict[str, int],
) -> Dict[str, int]:
    categories = [
        "correct_confident",
        "near_miss",
        "diffuse_uncertain_failure",
        "order_drift_failure",
    ]
    for category in categories:
        saved_counts.setdefault(category, 0)
    if max_per_category <= 0 or all(saved_counts[category] >= max_per_category for category in categories):
        return saved_counts
    viz_dir = output_dir / "heatmap_visualizations" / f"epoch_{epoch:03d}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    frames = batch["frames"].detach().cpu()
    targets = batch["targets"].detach().cpu()
    target_heatmaps = batch["target_heatmaps"].detach().cpu()
    predictions = decode_outputs["predictions"].detach().cpu()
    pred_heatmaps = decode_outputs["heatmap_logits"].detach().cpu()
    spatial_errors = spatial_errors.detach().cpu()
    entropy = entropy.detach().cpu()

    for batch_index in range(frames.size(0)):
        length = int(batch["target_lengths"][batch_index].item())
        target_sequence = [int(value) for value in targets[batch_index, :length].tolist()]
        prediction_sequence = [int(value) for value in predictions[batch_index, :length].tolist()]
        for step_index in range(length):
            step_error = float(spatial_errors[batch_index, step_index].item())
            step_entropy = float(entropy[batch_index, step_index].item())
            step_categories = _heatmap_step_categories(
                target_sequence=target_sequence,
                prediction_sequence=prediction_sequence,
                step_index=step_index,
                spatial_error=step_error,
                entropy=step_entropy,
            )
            if not step_categories:
                continue
            frame = frames[batch_index, step_index].permute(1, 2, 0).numpy()
            frame_u8 = (frame * 255.0).clip(0, 255).astype(np.uint8)
            target_heatmap_u8 = normalize_heatmap_image(target_heatmaps[batch_index, step_index].numpy())
            pred_heatmap_u8 = normalize_heatmap_image(pred_heatmaps[batch_index, step_index, 0].numpy())
            for category in step_categories:
                if saved_counts[category] >= max_per_category:
                    continue
                sample_dir = (
                    viz_dir
                    / category
                    / f"sample_{saved_counts[category]:03d}_{batch['trial_ids'][batch_index]}_step_{step_index + 1:02d}"
                )
                sample_dir.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(sample_dir / "input_frame.png", frame_u8)
                imageio.imwrite(sample_dir / "target_heatmap.png", target_heatmap_u8)
                imageio.imwrite(sample_dir / "predicted_heatmap.png", pred_heatmap_u8)
                with open(sample_dir / "metadata.json", "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "category": category,
                            "trial_id": batch["trial_ids"][batch_index],
                            "camera_name": batch["camera_names"][batch_index],
                            "recall_position": step_index + 1,
                            "target_index": int(targets[batch_index, step_index].item()),
                            "predicted_nearest_index": int(predictions[batch_index, step_index].item()),
                            "target_sequence": target_sequence,
                            "predicted_sequence": prediction_sequence,
                            "spatial_error": step_error,
                            "entropy": step_entropy,
                        },
                        handle,
                        indent=2,
                    )
                saved_counts[category] += 1
    for category in categories:
        (viz_dir / category).mkdir(parents=True, exist_ok=True)
    with open(viz_dir / "visualization_counts.json", "w", encoding="utf-8") as handle:
        json.dump(saved_counts, handle, indent=2)
    return saved_counts


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    loss_fn,
    device,
    *,
    config: TrainVisualConfig,
    output_dir: Path,
    epoch: int,
) -> Dict[str, object]:
    model.eval()
    epoch_stats = run_epoch(
        model,
        loader,
        optimizer=None,
        loss_fn=loss_fn,
        device=device,
        training=False,
        delay_mode=config.delay_mode,
        delay_steps=config.delay_steps,
        blank_feature_mode=config.blank_feature_mode,
        config=config,
    )

    predictions_all = []
    targets_all = []
    target_lengths_all = []
    frame_lengths_all = []
    masks_all = []
    trace_samples: list[Dict[str, Any]] = []
    trace_quota = {length: TRACE_SAMPLES_PER_LENGTH for length in LENGTH_RANGE}
    pred_xy_all = []
    target_xy_all = []
    target_block_indices_all = []
    heatmap_logits_all = []
    visualization_counts: Dict[str, int] = {}
    xy_hidden_records: list[Dict[str, torch.Tensor | list[str]]] = []
    block_xy_norm = None
    table_delta_scale = None
    hidden_export_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if block_xy_norm is None:
            block_xy_norm = block_norm_xy_from_mapping(
                batch.get("block_xy_norm", [{}])[0]
                if isinstance(batch.get("block_xy_norm"), list)
                else {}
            )
        if table_delta_scale is None:
            table_delta_scale = table_delta_scale_from_xy_normalization(
                batch.get("xy_normalization", [{}])[0]
                if isinstance(batch.get("xy_normalization"), list)
                else {}
            )
        need_traces = (
            config.return_hidden_traces
            and config.output_type == "index"
            and any(quota > 0 for quota in trace_quota.values())
        )
        need_xy_hidden_export = (
            config.output_type == "xy"
            and config.export_validation_hidden_states
            and (
                config.hidden_state_export_max_batches <= 0
                or hidden_export_batches < config.hidden_state_export_max_batches
            )
        )
        decode_outputs = model.greedy_decode(
            frames=batch["frames"],
            frame_lengths=batch["frame_lengths"],
            target_lengths=batch["target_lengths"],
            max_steps=batch["targets"].size(1),
            delay_mode=config.delay_mode,
            delay_steps=config.delay_steps,
            blank_feature_mode=config.blank_feature_mode,
            return_hidden_traces=need_traces or need_xy_hidden_export,
        )
        if config.output_type == "heatmap":
            predictions = decode_outputs["predictions"]
            batch_pred_xy = decode_outputs["heatmap_xy"].cpu()
            batch_heatmap_logits = decode_outputs["heatmap_logits"].cpu()
            batch_entropy = heatmap_entropy(decode_outputs["heatmap_logits"]).cpu()
            block_xy = model.block_heatmap_xy.detach().cpu()
            batch_targets_cpu = batch["targets"].detach().cpu()
            batch_target_xy = block_xy[batch_targets_cpu.clamp(0, block_xy.size(0) - 1)]
            batch_spatial_errors = torch.linalg.norm(batch_pred_xy.float() - batch_target_xy.float(), dim=-1)
            pred_xy_all.append(batch_pred_xy)
            heatmap_logits_all.append(batch_heatmap_logits)
            visualization_counts = save_heatmap_visualization_candidates(
                output_dir=output_dir,
                epoch=epoch,
                batch=batch,
                decode_outputs=decode_outputs,
                spatial_errors=batch_spatial_errors,
                entropy=batch_entropy,
                max_per_category=config.heatmap_visualization_samples,
                saved_counts=visualization_counts,
            )
        elif config.output_type == "xy":
            predictions = decode_outputs["predictions"]
            batch_pred_xy = decode_outputs["pred_xy"].detach().cpu()
            pred_xy_all.append(batch_pred_xy)
            target_xy_all.append(batch["target_xy"].detach().cpu())
            target_block_indices_all.append(batch["target_block_indices"].detach().cpu())
            if need_xy_hidden_export and "hidden_traces" in decode_outputs:
                hidden_traces = decode_outputs["hidden_traces"]
                xy_hidden_records.append(
                    {
                        "trial_ids": list(batch["trial_ids"]),
                        "decoder_hidden": hidden_traces["decoder_hidden_trace"].detach().cpu(),
                        "pred_xy": batch_pred_xy,
                        "target_xy": batch["target_xy"].detach().cpu(),
                        "target_block_index": batch["target_block_indices"].detach().cpu(),
                        "recall_position": torch.arange(
                            1,
                            batch_pred_xy.size(1) + 1,
                            dtype=torch.long,
                        ).unsqueeze(0).expand(batch_pred_xy.size(0), -1),
                        "sequence_length": batch["target_lengths"].detach().cpu(),
                    }
                )
                hidden_export_batches += 1
        elif need_traces:
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

    sequence_metric_targets = targets
    if config.output_type == "xy":
        sequence_metric_targets = torch.cat(
            [pad_to_max_steps(tensor, max_steps=max_steps, fill_value=-100) for tensor in target_block_indices_all],
            dim=0,
        )

    min_length_present, max_length_present = length_bounds_present(target_lengths)
    metrics = summarize_sequence_metrics(
        predictions,
        sequence_metric_targets,
        target_lengths,
        mask,
        min_length=min_length_present,
        max_length=max_length_present,
    )
    if config.output_type == "heatmap":
        pred_xy = torch.cat(
            [pad_to_max_steps(tensor, max_steps=max_steps, fill_value=0) for tensor in pred_xy_all],
            dim=0,
        )
        heatmap_logits = torch.cat(
            [pad_to_max_steps(tensor, max_steps=max_steps, fill_value=0.0) for tensor in heatmap_logits_all],
            dim=0,
        )
        entropy = heatmap_entropy(heatmap_logits)
        block_xy = model.block_heatmap_xy.detach().cpu()
        target_xy = block_xy[targets.clamp(0, block_xy.size(0) - 1)]
        spatial_errors = torch.linalg.norm(pred_xy.float() - target_xy.float(), dim=-1)
        _, nearest_distances = nearest_block_decode(
            pred_xy,
            block_xy=block_xy,
        )
        metrics.update(
            heatmap_spatial_metrics(
                pred_xy,
                targets,
                mask,
                block_xy,
                radius=config.heatmap_sigma,
            )
        )
        metrics["heatmap_entropy"] = _masked_mean(entropy, mask)
        metrics["token_acc_from_heatmap"] = float(metrics.get("token_accuracy", 0.0))
        metrics["val_heatmap_loss"] = float(epoch_stats["loss"])
        metrics["heatmap_per_length"] = heatmap_per_length_metrics(
            predictions=predictions,
            targets=targets,
            target_lengths=target_lengths,
            mask=mask,
            spatial_errors=spatial_errors,
            entropy=entropy,
            min_length=int(target_lengths.min().item()) if target_lengths.numel() else min(LENGTH_RANGE),
            max_length=int(target_lengths.max().item()) if target_lengths.numel() else max(LENGTH_RANGE),
        )
        metrics["heatmap_recall_position"] = heatmap_recall_position_metrics(
            predictions=predictions,
            targets=targets,
            mask=mask,
            spatial_errors=spatial_errors,
            entropy=entropy,
        )
        metrics["spatial_error_histogram"] = heatmap_spatial_error_histogram(spatial_errors, mask)
        metrics["heatmap_visualization_counts"] = visualization_counts
        metrics["mean_nearest_block_distance"] = float(nearest_distances[mask].float().mean().item())
        metrics["heatmap_visualization_dir"] = str(
            output_dir / "heatmap_visualizations" / f"epoch_{epoch:03d}"
        )
    elif config.output_type == "xy":
        if block_xy_norm is None:
            block_xy_norm = torch.as_tensor(standard_block_norm_xy(), dtype=torch.float32)
        if table_delta_scale is None:
            table_delta_scale = torch.tensor([1.0, 1.0], dtype=torch.float32)
        pred_xy = torch.cat(
            [pad_to_max_steps(tensor, max_steps=max_steps, fill_value=0.0) for tensor in pred_xy_all],
            dim=0,
        )
        target_xy = torch.cat(
            [pad_to_max_steps(tensor, max_steps=max_steps, fill_value=0.0) for tensor in target_xy_all],
            dim=0,
        )
        target_block_indices = sequence_metric_targets
        xy_metric_payload, nearest_predictions, xy_errors_norm = xy_metrics(
            pred_xy=pred_xy,
            target_xy=target_xy,
            target_block_indices=target_block_indices,
            target_lengths=target_lengths,
            mask=mask,
            block_xy_norm=block_xy_norm,
            table_delta_scale=table_delta_scale,
            radius_norm=config.xy_within_radius_norm,
        )
        metrics.update(xy_metric_payload)
        metrics["token_accuracy"] = metrics["nearest_block_accuracy"]
        metrics["full_sequence_accuracy"] = metrics["full_sequence_nearest_block_accuracy"]
        metrics["nearest_predictions_shape"] = list(nearest_predictions.shape)
        metrics["mean_xy_error_norm_debug"] = _masked_mean(xy_errors_norm, mask)
        metrics["block_positions_norm"] = block_xy_norm.tolist()
        metrics.update(masked_xy_distribution("pred_xy", pred_xy, mask))
        metrics.update(masked_xy_distribution("target_xy", target_xy, mask))
        metrics["xy_coordinate_system"] = "corsi_lower_left_normalized"
        metrics["xy_norm_range"] = (
            [-1.0, 1.0]
            if float(block_xy_norm.min().item()) < 0.0
            else [0.0, 1.0]
        )
        hidden_export_path = save_xy_hidden_state_export(
            output_dir,
            epoch=epoch,
            records=xy_hidden_records,
            block_positions=block_xy_norm,
        )
        if hidden_export_path is not None:
            metrics["xy_hidden_state_export_path"] = str(hidden_export_path)
    metrics["loss"] = epoch_stats["loss"]
    metrics["delay_mode"] = config.delay_mode
    metrics["delay_steps"] = int(config.delay_steps)
    metrics["blank_feature_mode"] = config.blank_feature_mode
    metrics["frame_length_mean"] = float(frame_lengths.float().mean().item())

    analysis_output_dir = Path(config.analysis_output_dir) if config.analysis_output_dir else None
    if config.return_hidden_traces and analysis_output_dir is not None and config.output_type == "index":
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


def get_last_alias_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "last.pt"


def get_best_checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "best_model.pt"


def get_best_val_loss_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "best_val_loss.pt"


def get_best_token_acc_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "best_token_acc.pt"


def get_best_nearest_block_acc_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "best_nearest_block_acc.pt"


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
        "output_type",
        "target_type",
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
        "use_encoder_summary_input",
        "max_decode_steps",
        "step_embedding_dim",
        "delay_mode",
        "delay_steps",
        "blank_feature_mode",
        "heatmap_size",
        "heatmap_sigma",
        "heatmap_normalize",
        "heatmap_loss",
        "heatmap_decode_method",
        "xy_loss",
        "xy_within_radius_norm",
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
    payload = {
            "experiment_name": output_dir.name,
            "seed": int(config.seed),
            "output_type": config.output_type,
            "target_type": config.target_type,
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
    if config.output_type == "heatmap":
        payload.update(
            {
                "heatmap_size": int(config.heatmap_size),
                "heatmap_sigma": float(config.heatmap_sigma),
                "heatmap_loss": config.heatmap_loss,
                "mean_spatial_error": float(metrics.get("mean_spatial_error", 0.0)),
                "median_spatial_error": float(metrics.get("median_spatial_error", 0.0)),
                "within_2px_acc": float(metrics.get("within_2px_acc", 0.0)),
                "within_4px_acc": float(metrics.get("within_4px_acc", 0.0)),
                "within_8px_acc": float(metrics.get("within_8px_acc", 0.0)),
                "heatmap_entropy": float(metrics.get("heatmap_entropy", 0.0)),
                "heatmap_per_length": metrics.get("heatmap_per_length", {}),
                "heatmap_recall_position": metrics.get("heatmap_recall_position", {}),
                "spatial_error_histogram": metrics.get("spatial_error_histogram", {}),
                "heatmap_visualization_dir": metrics.get("heatmap_visualization_dir", ""),
                "heatmap_visualization_counts": metrics.get("heatmap_visualization_counts", {}),
            }
        )
    if config.output_type == "xy":
        payload.update(
            {
                "xy_loss": config.xy_loss,
                "xy_within_radius_norm": float(config.xy_within_radius_norm),
                "mean_xy_error_norm": float(metrics.get("mean_xy_error_norm", 0.0)),
                "median_xy_error_norm": float(metrics.get("median_xy_error_norm", 0.0)),
                "mean_xy_error_table": float(metrics.get("mean_xy_error_table", 0.0)),
                "median_xy_error_table": float(metrics.get("median_xy_error_table", 0.0)),
                "within_radius_acc": float(metrics.get("within_radius_acc", 0.0)),
                "nearest_block_accuracy": float(metrics.get("nearest_block_accuracy", 0.0)),
                "full_sequence_nearest_block_accuracy": float(
                    metrics.get("full_sequence_nearest_block_accuracy", 0.0)
                ),
                "pred_xy_min": float(metrics.get("pred_xy_min", 0.0)),
                "pred_xy_max": float(metrics.get("pred_xy_max", 0.0)),
                "pred_xy_mean": float(metrics.get("pred_xy_mean", 0.0)),
                "pred_xy_std": float(metrics.get("pred_xy_std", 0.0)),
                "target_xy_min": float(metrics.get("target_xy_min", 0.0)),
                "target_xy_max": float(metrics.get("target_xy_max", 0.0)),
                "target_xy_mean": float(metrics.get("target_xy_mean", 0.0)),
                "target_xy_std": float(metrics.get("target_xy_std", 0.0)),
                "xy_coordinate_system": metrics.get("xy_coordinate_system", ""),
                "xy_norm_range": metrics.get("xy_norm_range", []),
                "xy_per_length": metrics.get("xy_per_length", {}),
                "xy_hidden_state_export_path": metrics.get("xy_hidden_state_export_path", ""),
            }
        )
    return round_nested(payload)


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
            "target_type": dataset.target_type,
            "xy_normalization": dataset.xy_normalization,
            "block_xy_norm": dataset.block_xy_norm,
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
        heatmap_size=config.heatmap_size,
        heatmap_sigma=config.heatmap_sigma,
        heatmap_normalize=config.heatmap_normalize,
        target_type=config.target_type,
    )
    train_dataset = filter_dataset_by_lengths(train_dataset, config.train_length_filter)
    subset_request = int(config.max_train_samples or config.overfit_subset_size)
    if subset_request > 0:
        subset_size = min(subset_request, len(train_dataset))
        if subset_size < 1:
            raise ValueError("--max-train-samples / --overfit-subset-size must select at least one sample")
        generator = torch.Generator().manual_seed(config.seed)
        indices = torch.randperm(len(train_dataset), generator=generator)[:subset_size].tolist()
        train_dataset = Subset(train_dataset, indices)
        if config.val_same_as_train:
            return train_dataset, train_dataset
    if config.val_dataset_root:
        val_dataset = RobosuiteVisualCorsiDataset(
            config.val_dataset_root,
            camera_name=config.camera_name,
            include_reset_frame=config.include_reset_frame,
            heatmap_size=config.heatmap_size,
            heatmap_sigma=config.heatmap_sigma,
            heatmap_normalize=config.heatmap_normalize,
            target_type=config.target_type,
        )
        val_dataset = filter_dataset_by_lengths(val_dataset, config.val_length_filter)
        return train_dataset, val_dataset
    if config.val_same_as_train:
        return train_dataset, train_dataset
    train_split, val_split = split_dataset(train_dataset, config.val_ratio, config.seed)
    return train_split, filter_dataset_by_lengths(val_split, config.val_length_filter)


def main() -> None:
    config = parse_args()
    if config.resume and config.init_model:
        raise ValueError("--resume and --init_model cannot be used together")
    if config.output_type == "heatmap" and config.use_attention:
        raise ValueError("Heatmap output mode requires model.use_attention=false")
    if config.output_type == "heatmap" and config.scheduled_sampling:
        raise ValueError("Scheduled sampling must be disabled for heatmap mode")
    if config.output_type == "xy" and config.target_type == "block_index":
        raise ValueError("output_type='xy' requires target.type to be block_center_xy or end_effector_xy")
    if config.output_type != "xy" and config.target_type != "block_index":
        print(
            json.dumps(
                {
                    "warning": "target.type is ignored unless model.output_type='xy'",
                    "target_type": config.target_type,
                    "output_type": config.output_type,
                }
            )
        )
    if config.heatmap_decode_method != "argmax":
        raise ValueError("Only heatmap.decode_method='argmax' is supported")
    if config.heatmap_size < 2:
        raise ValueError("--heatmap-size must be >= 2")
    if config.heatmap_sigma <= 0:
        raise ValueError("--heatmap-sigma must be > 0")
    if config.xy_within_radius_norm <= 0:
        raise ValueError("--xy-within-radius-norm must be > 0")
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
    if config.overfit_subset_size < 0:
        raise ValueError("--overfit-subset-size must be >= 0")
    if config.max_train_samples < 0:
        raise ValueError("--max-train-samples must be >= 0")
    if config.gradient_clip_norm < 0.0:
        raise ValueError("--gradient-clip-norm must be >= 0")
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
    if config.output_type == "xy":
        wiring_path = save_xy_training_wiring_check(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            config=config,
            output_dir=output_dir,
        )
        print(json.dumps({"xy_training_wiring": str(wiring_path)}))

    resume_path = get_last_checkpoint_path(checkpoint_dir)
    start_epoch = 1
    global_step = 0
    best_metric = -1.0
    best_val_loss = float("inf")
    best_token_acc = -1.0
    best_nearest_block_acc = -1.0
    best_val_loss_epoch = 0
    best_nearest_block_acc_epoch = 0
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
            config=config,
        )
        global_step = int(train_stats["global_step"])
        val_metrics = evaluate_model(
            model,
            val_loader,
            loss_fn,
            device,
            config=config,
            output_dir=output_dir,
            epoch=epoch,
        )
        xy_sanity_paths: Dict[str, str] = {}
        if config.output_type == "xy":
            train_sanity_path = save_xy_batch_sanity_dump(
                model=model,
                loader=train_loader,
                device=device,
                config=config,
                output_dir=output_dir,
                epoch=epoch,
                split_name="train",
            )
            val_sanity_path = save_xy_batch_sanity_dump(
                model=model,
                loader=val_loader,
                device=device,
                config=config,
                output_dir=output_dir,
                epoch=epoch,
                split_name="val",
            )
            if train_sanity_path is not None:
                xy_sanity_paths["train"] = str(train_sanity_path)
            if val_sanity_path is not None:
                xy_sanity_paths["val"] = str(val_sanity_path)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(float(train_stats["loss"]), 6),
            "val_loss": round(float(val_metrics["loss"]), 6),
            "output_type": config.output_type,
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

        if config.output_type == "heatmap":
            epoch_record = {
                "epoch": epoch,
                "train_loss": round(float(train_stats["loss"]), 6),
                "val_loss": round(float(val_metrics["loss"]), 6),
                "token_acc_from_heatmap": round(float(val_metrics["token_accuracy"]), 6),
                "full_sequence_accuracy": round(float(val_metrics["full_sequence_accuracy"]), 6),
                "mean_spatial_error": round(float(val_metrics["mean_spatial_error"]), 6),
                "median_spatial_error": round(float(val_metrics["median_spatial_error"]), 6),
                "within_4px_acc": round(float(val_metrics["within_4px_acc"]), 6),
                "within_8px_acc": round(float(val_metrics["within_8px_acc"]), 6),
                "heatmap_entropy": round(float(val_metrics["heatmap_entropy"]), 6),
            }
        elif config.output_type == "xy":
            epoch_record = {
                "epoch": epoch,
                "train_loss": round(float(train_stats["loss"]), 6),
                "val_loss": round(float(val_metrics["loss"]), 6),
                "train_pred_xy_min": round(float(train_stats.get("pred_xy_min", 0.0)), 6),
                "train_pred_xy_max": round(float(train_stats.get("pred_xy_max", 0.0)), 6),
                "train_pred_xy_mean": round(float(train_stats.get("pred_xy_mean", 0.0)), 6),
                "train_pred_xy_std": round(float(train_stats.get("pred_xy_std", 0.0)), 6),
                "train_target_xy_min": round(float(train_stats.get("target_xy_min", 0.0)), 6),
                "train_target_xy_max": round(float(train_stats.get("target_xy_max", 0.0)), 6),
                "train_target_xy_mean": round(float(train_stats.get("target_xy_mean", 0.0)), 6),
                "train_target_xy_std": round(float(train_stats.get("target_xy_std", 0.0)), 6),
                "pred_xy_min": round(float(val_metrics.get("pred_xy_min", 0.0)), 6),
                "pred_xy_max": round(float(val_metrics.get("pred_xy_max", 0.0)), 6),
                "pred_xy_mean": round(float(val_metrics.get("pred_xy_mean", 0.0)), 6),
                "pred_xy_std": round(float(val_metrics.get("pred_xy_std", 0.0)), 6),
                "target_xy_min": round(float(val_metrics.get("target_xy_min", 0.0)), 6),
                "target_xy_max": round(float(val_metrics.get("target_xy_max", 0.0)), 6),
                "target_xy_mean": round(float(val_metrics.get("target_xy_mean", 0.0)), 6),
                "target_xy_std": round(float(val_metrics.get("target_xy_std", 0.0)), 6),
                "mean_xy_error_norm": round(float(val_metrics["mean_xy_error_norm"]), 6),
                "median_xy_error_norm": round(float(val_metrics["median_xy_error_norm"]), 6),
                "mean_xy_error_table": round(float(val_metrics["mean_xy_error_table"]), 6),
                "median_xy_error_table": round(float(val_metrics["median_xy_error_table"]), 6),
                "within_radius_acc": round(float(val_metrics["within_radius_acc"]), 6),
                "nearest_block_accuracy": round(float(val_metrics["nearest_block_accuracy"]), 6),
                "full_sequence_nearest_block_accuracy": round(
                    float(val_metrics["full_sequence_nearest_block_accuracy"]),
                    6,
                ),
                "xy_per_length": round_nested(val_metrics.get("xy_per_length", {})),
                "xy_sanity_paths": xy_sanity_paths,
            }
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
        save_checkpoint(get_last_alias_path(checkpoint_dir), last_checkpoint)

        current_val_loss = float(val_metrics["loss"])
        current_token_acc = float(val_metrics["token_accuracy"])
        current_nearest_block_acc = float(val_metrics.get("nearest_block_accuracy", current_token_acc))
        val_loss_improved = current_val_loss < best_val_loss
        token_acc_improved = current_token_acc > best_token_acc
        nearest_block_acc_improved = current_nearest_block_acc > best_nearest_block_acc
        if val_loss_improved:
            best_val_loss = current_val_loss
            best_val_loss_epoch = epoch
            save_checkpoint(get_best_val_loss_path(checkpoint_dir), last_checkpoint)
        if token_acc_improved:
            best_token_acc = current_token_acc
            save_checkpoint(get_best_token_acc_path(checkpoint_dir), last_checkpoint)
        if nearest_block_acc_improved:
            best_nearest_block_acc = current_nearest_block_acc
            best_nearest_block_acc_epoch = epoch
            save_checkpoint(get_best_nearest_block_acc_path(checkpoint_dir), last_checkpoint)

        full_sequence_improved = float(val_metrics["full_sequence_accuracy"]) > best_metric
        if full_sequence_improved:
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

        if config.output_type == "heatmap":
            if val_loss_improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        elif not full_sequence_improved:
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
            "best_val_loss_epoch": best_val_loss_epoch,
            "best_nearest_block_acc_epoch": best_nearest_block_acc_epoch,
            "best_full_seq_acc": best_metric,
            "best_full_sequence_accuracy": best_metric,
            "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
            "best_token_acc": best_token_acc,
            "best_nearest_block_accuracy": best_nearest_block_acc,
            "overall_nearest_block_accuracy": float(best_metrics.get("nearest_block_accuracy", 0.0)),
            "full_sequence_nearest_block_accuracy": float(
                best_metrics.get("full_sequence_nearest_block_accuracy", 0.0)
            ),
            "length_2_nearest_block_accuracy": float(
                best_metrics.get("xy_per_length", {}).get("2", {}).get("nearest_block_accuracy", 0.0)
            ),
            "length_3_nearest_block_accuracy": float(
                best_metrics.get("xy_per_length", {}).get("3", {}).get("nearest_block_accuracy", 0.0)
            ),
            "mean_xy_error_norm": float(best_metrics.get("mean_xy_error_norm", 0.0)),
            "mean_xy_error_table": float(best_metrics.get("mean_xy_error_table", 0.0)),
            "output_type": config.output_type,
            "target_type": config.target_type,
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
            "heatmap": {
                "size": int(config.heatmap_size),
                "sigma": float(config.heatmap_sigma),
                "normalize": bool(config.heatmap_normalize),
                "loss": config.heatmap_loss,
                "decode_method": config.heatmap_decode_method,
            },
            "best_metrics": best_metrics,
        }
    )
    save_results_summary(output_dir, summary)


if __name__ == "__main__":
    main()
