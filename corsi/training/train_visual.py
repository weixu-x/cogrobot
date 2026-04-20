"""Training entry point for the minimal visual robosuite Corsi model."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from corsi.analysis.metrics import summarize_sequence_metrics
from corsi.data import RobosuiteVisualCorsiDataset, collate_visual_batch
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
    seed: int = 7
    device: str = "auto"
    output_dir: str = "corsi_artifacts/visual_base/training/visual_lstm"


def load_config_overrides(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object")
    return data


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


def run_epoch(model, loader, optimizer, loss_fn, device, training: bool) -> Dict[str, float]:
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
        logits = outputs["logits"]
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch["targets_pad"].reshape(-1))

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        active_tokens = int(batch["mask"].sum().item())
        loss_total += float(loss.item()) * active_tokens
        token_count += active_tokens

    return {"loss": loss_total / token_count if token_count > 0 else 0.0}


@torch.no_grad()
def evaluate_model(model, loader, loss_fn, device) -> Dict[str, object]:
    model.eval()
    epoch_stats = run_epoch(model, loader, optimizer=None, loss_fn=loss_fn, device=device, training=False)

    predictions_all = []
    targets_all = []
    lengths_all = []
    masks_all = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        predictions = model.greedy_decode(
            frames=batch["frames_pad"],
            lengths=batch["lengths"],
            max_steps=batch["targets_pad"].size(1),
        )
        predictions_all.append(predictions.cpu())
        targets_all.append(batch["targets_pad"].cpu())
        lengths_all.append(batch["lengths"].cpu())
        masks_all.append(batch["mask"].cpu())

    predictions = torch.cat(predictions_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    lengths = torch.cat(lengths_all, dim=0)
    mask = torch.cat(masks_all, dim=0)

    metrics = summarize_sequence_metrics(predictions, targets, lengths, mask)
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
    set_seed(config.seed)

    device, device_info = resolve_torch_device(config.device)
    config.device = str(device)
    output_dir = Path(config.output_dir)
    save_config(output_dir, config)
    with open(output_dir / "device_info.json", "w", encoding="utf-8") as handle:
        json.dump(device_info, handle, indent=2)
    print(json.dumps(device_info))

    train_dataset, val_dataset = build_datasets(config)
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
        train_stats = run_epoch(model, train_loader, optimizer, loss_fn, device, training=True)
        val_metrics = evaluate_model(model, val_loader, loss_fn, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_stats["loss"], 6),
            "val_loss": round(float(val_metrics["loss"]), 6),
            "token_accuracy": round(float(val_metrics["token_accuracy"]), 6),
            "full_sequence_accuracy": round(float(val_metrics["full_sequence_accuracy"]), 6),
            "estimated_span": int(val_metrics["estimated_span"]),
            "accuracy_by_length": val_metrics["accuracy_by_length"],
            "error_breakdown": val_metrics["error_breakdown"],
        }
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
            "best_metrics": best_metrics,
        },
    )


if __name__ == "__main__":
    main()
