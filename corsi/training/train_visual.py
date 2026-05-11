"""Training entry point for the minimal visual robosuite Corsi model."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from corsi.analysis.metrics import summarize_sequence_metrics
from corsi.data import RobosuiteVisualCorsiDataset, collate_visual_batch
from corsi.heatmaps import nearest_block_decode, spatial_heatmap_loss
from corsi.models.lstm_visual import VisualLSTMConfig, VisualSeq2SeqLSTM
from corsi.training.device import resolve_torch_device


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
    output_type: str = "index"
    use_attention: bool = False
    use_step_embedding: bool = False
    max_decode_steps: int = 6
    step_embedding_dim: int = 16
    scheduled_sampling: bool = False
    heatmap_size: int = 32
    heatmap_sigma: float = 2.0
    heatmap_normalize: bool = True
    heatmap_loss: str = "spatial_ce"
    heatmap_decode_method: str = "argmax"
    heatmap_visualization_samples: int = 4
    seed: int = 7
    device: str = "auto"
    output_dir: str = "corsi_artifacts/visual_base/training/visual_lstm"


def normalize_config_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(data)
    model_overrides = normalized.pop("model", None)
    if isinstance(model_overrides, dict):
        for key in (
            "output_type",
            "use_attention",
            "use_step_embedding",
            "max_decode_steps",
            "step_embedding_dim",
        ):
            if key in model_overrides:
                normalized[key] = model_overrides[key]

    training_overrides = normalized.pop("training", None)
    if isinstance(training_overrides, dict):
        alias_map = {
            "use_scheduled_sampling": "scheduled_sampling",
            "scheduled_sampling": "scheduled_sampling",
            "epochs": "epochs",
            "max_epochs": "epochs",
            "seed": "seed",
        }
        for source_key, target_key in alias_map.items():
            if source_key in training_overrides:
                normalized[target_key] = training_overrides[source_key]

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
    return normalized


def parse_simple_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
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
    parser.add_argument("--output-type", type=str, default=defaults["output_type"], choices=["index", "heatmap"])
    parser.add_argument(
        "--use-attention",
        action=argparse.BooleanOptionalAction,
        default=defaults["use_attention"],
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
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--device", type=str, default=defaults["device"], choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output-dir", type=str, default=defaults["output_dir"])

    parsed = vars(parser.parse_args())
    parsed.pop("config", None)
    return TrainVisualConfig(**parsed)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_config(train_config: TrainVisualConfig) -> VisualLSTMConfig:
    return VisualLSTMConfig(
        cnn_feature_dim=train_config.cnn_feature_dim,
        token_embedding_dim=train_config.token_embedding_dim,
        hidden_dim=train_config.hidden_dim,
        num_layers=train_config.num_layers,
        dropout=train_config.dropout,
        input_image_size=train_config.input_image_size,
        output_type=train_config.output_type,
        use_attention=train_config.use_attention,
        use_step_embedding=train_config.use_step_embedding,
        max_decode_steps=train_config.max_decode_steps,
        step_embedding_dim=train_config.step_embedding_dim,
        heatmap_size=train_config.heatmap_size,
    )


def build_dataloader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_visual_batch,
    )


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = dict(batch)
    moved["frames_pad"] = batch["frames_pad"].to(device)
    moved["targets_pad"] = batch["targets_pad"].to(device)
    moved["lengths"] = batch["lengths"].to(device)
    moved["mask"] = batch["mask"].to(device)
    if batch.get("target_heatmaps") is not None:
        moved["target_heatmaps"] = batch["target_heatmaps"].to(device)
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


def compute_loss(
    *,
    config: TrainVisualConfig,
    outputs: Dict[str, Any],
    batch: Dict[str, object],
    loss_fn,
) -> torch.Tensor:
    if config.output_type == "index":
        logits = outputs["logits"]
        return loss_fn(logits.reshape(-1, logits.size(-1)), batch["targets_pad"].reshape(-1))
    return spatial_heatmap_loss(
        outputs["heatmap_logits"],
        batch["target_heatmaps"],
        batch["mask"],
        loss_type=config.heatmap_loss,
    )


def run_epoch(model, loader, optimizer, loss_fn, device, training: bool, config: TrainVisualConfig) -> Dict[str, float]:
    model.train(training)
    loss_total = 0.0
    token_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            frames=batch["frames_pad"],
            lengths=batch["lengths"],
            targets=batch["targets_pad"],
        )
        loss = compute_loss(config=config, outputs=outputs, batch=batch, loss_fn=loss_fn)

        if training:
            if optimizer is None:
                raise ValueError("Optimizer is required when training=True")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        active_tokens = int(batch["mask"].sum().item())
        loss_total += float(loss.item()) * active_tokens
        token_count += active_tokens

    return {"loss": loss_total / token_count if token_count > 0 else 0.0}


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
        }
    return {
        "mean_spatial_error": float(active_errors.mean().item()),
        "median_spatial_error": float(active_errors.median().item()),
        "within_radius_accuracy": float((active_errors <= float(radius)).float().mean().item()),
    }


def normalize_heatmap_image(heatmap: np.ndarray) -> np.ndarray:
    values = np.asarray(heatmap, dtype=np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value > min_value:
        values = (values - min_value) / (max_value - min_value)
    else:
        values = np.zeros_like(values)
    return (values * 255.0).clip(0, 255).astype(np.uint8)


def save_heatmap_visualizations(
    *,
    output_dir: Path,
    epoch: int,
    batch: Dict[str, object],
    decode_outputs: Dict[str, torch.Tensor],
    max_samples: int,
    already_saved: int,
) -> int:
    if max_samples <= already_saved:
        return already_saved
    viz_dir = output_dir / "heatmap_visualizations" / f"epoch_{epoch:03d}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    frames = batch["frames_pad"].detach().cpu()
    targets = batch["targets_pad"].detach().cpu()
    target_heatmaps = batch["target_heatmaps"].detach().cpu()
    predictions = decode_outputs["predictions"].detach().cpu()
    pred_heatmaps = decode_outputs["heatmap_logits"].detach().cpu()

    for batch_index in range(frames.size(0)):
        if already_saved >= max_samples:
            break
        sample_dir = viz_dir / f"sample_{already_saved:03d}_{batch['trial_ids'][batch_index]}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        length = int(batch["lengths"][batch_index].item())
        steps = []
        for step_index in range(length):
            frame = frames[batch_index, step_index].permute(1, 2, 0).numpy()
            frame_u8 = (frame * 255.0).clip(0, 255).astype(np.uint8)
            target_heatmap_u8 = normalize_heatmap_image(target_heatmaps[batch_index, step_index].numpy())
            pred_heatmap_u8 = normalize_heatmap_image(pred_heatmaps[batch_index, step_index, 0].numpy())
            imageio.imwrite(sample_dir / f"input_frame_step_{step_index:02d}.png", frame_u8)
            imageio.imwrite(sample_dir / f"target_heatmap_step_{step_index:02d}.png", target_heatmap_u8)
            imageio.imwrite(sample_dir / f"predicted_heatmap_step_{step_index:02d}.png", pred_heatmap_u8)
            steps.append(
                {
                    "step": step_index,
                    "target_index": int(targets[batch_index, step_index].item()),
                    "predicted_nearest_index": int(predictions[batch_index, step_index].item()),
                }
            )
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "trial_id": batch["trial_ids"][batch_index],
                    "camera_name": batch["camera_names"][batch_index],
                    "steps": steps,
                },
                handle,
                indent=2,
            )
        already_saved += 1
    return already_saved


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
        config=config,
    )

    predictions_all = []
    targets_all = []
    lengths_all = []
    masks_all = []
    pred_xy_all = []
    visualizations_saved = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        decode_outputs = model.greedy_decode(
            frames=batch["frames_pad"],
            lengths=batch["lengths"],
            max_steps=batch["targets_pad"].size(1),
        )
        if config.output_type == "heatmap":
            predictions = decode_outputs["predictions"]
            pred_xy_all.append(decode_outputs["heatmap_xy"].cpu())
            visualizations_saved = save_heatmap_visualizations(
                output_dir=output_dir,
                epoch=epoch,
                batch=batch,
                decode_outputs=decode_outputs,
                max_samples=config.heatmap_visualization_samples,
                already_saved=visualizations_saved,
            )
        else:
            predictions = decode_outputs
        predictions_all.append(predictions.cpu())
        targets_all.append(batch["targets_pad"].cpu())
        lengths_all.append(batch["lengths"].cpu())
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
    lengths = torch.cat(lengths_all, dim=0)
    mask = torch.cat(
        [pad_to_max_steps(tensor, max_steps=max_steps, fill_value=False) for tensor in masks_all],
        dim=0,
    )

    metrics = summarize_sequence_metrics(predictions, targets, lengths, mask)
    if config.output_type == "heatmap":
        pred_xy = torch.cat(
            [pad_to_max_steps(tensor, max_steps=max_steps, fill_value=0) for tensor in pred_xy_all],
            dim=0,
        )
        _, nearest_distances = nearest_block_decode(
            pred_xy,
            block_xy=model.block_heatmap_xy.detach().cpu(),
        )
        metrics.update(
            heatmap_spatial_metrics(
                pred_xy,
                targets,
                mask,
                model.block_heatmap_xy.detach().cpu(),
                radius=config.heatmap_sigma,
            )
        )
        metrics["mean_nearest_block_distance"] = float(nearest_distances[mask].float().mean().item())
        metrics["heatmap_visualization_dir"] = str(
            output_dir / "heatmap_visualizations" / f"epoch_{epoch:03d}"
        )
    metrics["loss"] = epoch_stats["loss"]
    return metrics


def save_checkpoint(output_dir: Path, model, config: TrainVisualConfig, metrics: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": asdict(config),
            "metrics": metrics,
        },
        output_dir / "best_model.pt",
    )
    with open(output_dir / "best_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_config(output_dir: Path, config: TrainVisualConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)


def append_epoch_log(output_dir: Path, record: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "epoch_logs.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def save_final_summary(output_dir: Path, summary: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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


def save_dataset_info(output_dir: Path, train_dataset, val_dataset) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_dataset": dataset_summary(train_dataset),
        "val_dataset": dataset_summary(val_dataset),
    }
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_datasets(config: TrainVisualConfig):
    train_dataset = RobosuiteVisualCorsiDataset(
        config.dataset_root,
        camera_name=config.camera_name,
        include_reset_frame=config.include_reset_frame,
        heatmap_size=config.heatmap_size,
        heatmap_sigma=config.heatmap_sigma,
        heatmap_normalize=config.heatmap_normalize,
    )
    if config.val_dataset_root:
        val_dataset = RobosuiteVisualCorsiDataset(
            config.val_dataset_root,
            camera_name=config.camera_name,
            include_reset_frame=config.include_reset_frame,
            heatmap_size=config.heatmap_size,
            heatmap_sigma=config.heatmap_sigma,
            heatmap_normalize=config.heatmap_normalize,
        )
        return train_dataset, val_dataset
    return split_dataset(train_dataset, config.val_ratio, config.seed)


def main() -> None:
    config = parse_args()
    if config.output_type == "heatmap" and config.use_attention:
        raise ValueError("Heatmap output mode requires model.use_attention=false")
    if config.output_type == "heatmap" and config.scheduled_sampling:
        raise ValueError("Scheduled sampling must be disabled for heatmap mode")
    if config.heatmap_decode_method != "argmax":
        raise ValueError("Only heatmap.decode_method='argmax' is supported")
    if config.heatmap_size < 2:
        raise ValueError("--heatmap-size must be >= 2")
    if config.heatmap_sigma <= 0:
        raise ValueError("--heatmap-sigma must be > 0")
    if config.step_embedding_dim < 1 and config.use_step_embedding:
        raise ValueError("--step-embedding-dim must be >= 1 when step embedding is enabled")
    if config.max_decode_steps < 1:
        raise ValueError("--max-decode-steps must be >= 1")
    set_seed(config.seed)

    device, device_info = resolve_torch_device(config.device)
    config.device = str(device)
    output_dir = Path(config.output_dir)
    save_config(output_dir, config)
    with open(output_dir / "device_info.json", "w", encoding="utf-8") as handle:
        json.dump(device_info, handle, indent=2)
    print(json.dumps(device_info))

    train_dataset, val_dataset = build_datasets(config)
    save_dataset_info(output_dir, train_dataset, val_dataset)
    train_loader = build_dataloader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model_config = build_model_config(config)
    model = VisualSeq2SeqLSTM(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=model_config.target_pad_value)

    best_metric = -1.0
    best_epoch = 0
    best_metrics: Dict[str, object] = {}
    for epoch in range(1, config.epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            training=True,
            config=config,
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            loss_fn,
            device,
            config=config,
            output_dir=output_dir,
            epoch=epoch,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_stats["loss"], 6),
            "val_loss": round(float(val_metrics["loss"]), 6),
            "output_type": config.output_type,
            "token_accuracy": round(float(val_metrics["token_accuracy"]), 6),
            "full_sequence_accuracy": round(float(val_metrics["full_sequence_accuracy"]), 6),
            "estimated_span": int(val_metrics["estimated_span"]),
            "accuracy_by_length": val_metrics["accuracy_by_length"],
            "error_breakdown": val_metrics["error_breakdown"],
        }
        if config.output_type == "heatmap":
            epoch_record.update(
                {
                    "mean_spatial_error": round(float(val_metrics["mean_spatial_error"]), 6),
                    "median_spatial_error": round(float(val_metrics["median_spatial_error"]), 6),
                    "within_radius_accuracy": round(float(val_metrics["within_radius_accuracy"]), 6),
                }
            )
        print(json.dumps(epoch_record))
        append_epoch_log(output_dir, epoch_record)

        if float(val_metrics["full_sequence_accuracy"]) > best_metric:
            best_metric = float(val_metrics["full_sequence_accuracy"])
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            save_checkpoint(output_dir, model, config, val_metrics)

    save_final_summary(
        output_dir,
        {
            "best_epoch": best_epoch,
            "best_full_sequence_accuracy": best_metric,
            "output_type": config.output_type,
            "use_attention": bool(config.use_attention),
            "use_step_embedding": bool(config.use_step_embedding),
            "heatmap": {
                "size": int(config.heatmap_size),
                "sigma": float(config.heatmap_sigma),
                "normalize": bool(config.heatmap_normalize),
                "loss": config.heatmap_loss,
                "decode_method": config.heatmap_decode_method,
            },
            "best_metrics": best_metrics,
        },
    )


if __name__ == "__main__":
    main()
