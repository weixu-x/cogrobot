"""Training entry point for the minimal coordinate-based Corsi model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader

from corsi.analysis.metrics import summarize_sequence_metrics
from corsi.data import CoordinateCorsiDataset, collate_coordinate_batch
from corsi.models.lstm_coord import CoordLSTMConfig, CoordinateSeq2SeqLSTM
from corsi.training.device import resolve_torch_device


@dataclass
class TrainCoordConfig:
    train_trials: int = 20000
    val_trials: int = 2000
    seq_min: int = 2
    seq_max: int = 6
    feature_mode: str = "xydxdy"
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    coord_embedding_dim: int = 64
    token_embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    seed: int = 7
    device: str = "auto"
    output_dir: str = "corsi_artifacts/coord_lstm"


def load_config_overrides(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object")
    return data


def parse_args() -> TrainCoordConfig:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default="")
    bootstrap_args, _ = bootstrap.parse_known_args()

    defaults = asdict(TrainCoordConfig())
    if bootstrap_args.config:
        defaults.update(load_config_overrides(bootstrap_args.config))

    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.add_argument("--train-trials", type=int, default=defaults["train_trials"])
    parser.add_argument("--val-trials", type=int, default=defaults["val_trials"])
    parser.add_argument("--seq-min", type=int, default=defaults["seq_min"])
    parser.add_argument("--seq-max", type=int, default=defaults["seq_max"])
    parser.add_argument("--feature-mode", type=str, default=defaults["feature_mode"], choices=["xy", "xydxdy"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--coord-embedding-dim", type=int, default=defaults["coord_embedding_dim"])
    parser.add_argument("--token-embedding-dim", type=int, default=defaults["token_embedding_dim"])
    parser.add_argument("--hidden-dim", type=int, default=defaults["hidden_dim"])
    parser.add_argument("--num-layers", type=int, default=defaults["num_layers"])
    parser.add_argument("--dropout", type=float, default=defaults["dropout"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--device", type=str, default=defaults["device"], choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output-dir", type=str, default=defaults["output_dir"])

    parsed = vars(parser.parse_args())
    parsed.pop("config", None)
    return TrainCoordConfig(**parsed)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_config(train_config: TrainCoordConfig) -> CoordLSTMConfig:
    input_dim = 2 if train_config.feature_mode == "xy" else 4
    return CoordLSTMConfig(
        input_dim=input_dim,
        coord_embedding_dim=train_config.coord_embedding_dim,
        token_embedding_dim=train_config.token_embedding_dim,
        hidden_dim=train_config.hidden_dim,
        num_layers=train_config.num_layers,
        dropout=train_config.dropout,
    )


def build_dataloader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_coordinate_batch,
    )


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = dict(batch)
    moved["coords_pad"] = batch["coords_pad"].to(device)
    moved["targets_pad"] = batch["targets_pad"].to(device)
    moved["lengths"] = batch["lengths"].to(device)
    moved["mask"] = batch["mask"].to(device)
    return moved


def run_epoch(model, loader, optimizer, loss_fn, device, training: bool) -> Dict[str, float]:
    model.train(training)
    loss_total = 0.0
    token_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            coords=batch["coords_pad"],
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
            coords=batch["coords_pad"],
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


def save_checkpoint(output_dir: Path, model, config: TrainCoordConfig, metrics: Dict[str, object]) -> None:
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


def save_config(output_dir: Path, config: TrainCoordConfig) -> None:
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

    train_dataset = CoordinateCorsiDataset(
        num_trials=config.train_trials,
        seq_len_range=(config.seq_min, config.seq_max),
        feature_mode=config.feature_mode,
        seed=config.seed,
    )
    val_dataset = CoordinateCorsiDataset(
        num_trials=config.val_trials,
        seq_len_range=(config.seq_min, config.seq_max),
        feature_mode=config.feature_mode,
        seed=config.seed + 1,
    )

    train_loader = build_dataloader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model_config = build_model_config(config)
    model = CoordinateSeq2SeqLSTM(model_config).to(device)
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
