"""Evaluate visual Corsi checkpoints with teacher-forced or free-running decoding."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.analysis.attention_analysis import analyze_attention_weights, summarize_attention_rows
from corsi.analysis.error_taxonomy import classify_trials
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
from corsi.training.train_visual import move_batch_to_device, pad_to_max_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--camera-name", default="")
    parser.add_argument("--include-reset-frame", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mode", choices=["free_running", "teacher_forced"], default="free_running")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-attention", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _dataclass_from_dict(cls, value: Any):
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        field_names = {field.name for field in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in field_names})
    return cls()


def build_model_config_from_checkpoint(payload: Dict[str, Any]) -> VisualLSTMConfig:
    saved_config = payload.get("model_config") or {}
    args = payload.get("args") or {}
    if not saved_config:
        saved_config = {
            "cnn_feature_dim": args.get("cnn_feature_dim", 128),
            "token_embedding_dim": args.get("token_embedding_dim", 64),
            "hidden_dim": args.get("hidden_dim", 128),
            "num_layers": args.get("num_layers", 1),
            "dropout": args.get("dropout", 0.0),
            "input_image_size": args.get("input_image_size", 128),
            "use_attention": args.get("use_attention", False),
            "attention_dim": args.get("attention_dim", 128),
            "use_step_embedding": args.get("use_step_embedding", False),
            "max_decode_steps": args.get("max_decode_steps", 6),
            "step_embedding_dim": args.get("step_embedding_dim", 16),
        }
    if args.get("attention_type", "global") != "global":
        saved_config["use_attention"] = True

    saved_config["local_attention"] = _dataclass_from_dict(
        LocalAttentionConfig, saved_config.get("local_attention") or args.get("local_attention")
    )
    saved_config["noisy_attention"] = _dataclass_from_dict(
        NoisyAttentionConfig, saved_config.get("noisy_attention") or args.get("noisy_attention")
    )
    saved_config["memory_decay"] = _dataclass_from_dict(
        MemoryDecayConfig, saved_config.get("memory_decay") or args.get("memory_decay")
    )
    saved_config["capacity_gate"] = _dataclass_from_dict(
        CapacityGateConfig, saved_config.get("capacity_gate") or args.get("capacity_gate")
    )
    saved_config["response_suppression"] = _dataclass_from_dict(
        ResponseSuppressionConfig,
        saved_config.get("response_suppression") or args.get("response_suppression"),
    )

    field_names = {field.name for field in fields(VisualLSTMConfig)}
    return VisualLSTMConfig(
        **{key: value for key, value in saved_config.items() if key in field_names}
    )


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pad_attention_to_shape(tensor: torch.Tensor, *, max_steps: int, max_encoder_steps: int) -> torch.Tensor:
    if tensor.size(1) == max_steps and tensor.size(2) == max_encoder_steps:
        return tensor
    padded = torch.zeros(
        tensor.size(0),
        max_steps,
        max_encoder_steps,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:, : tensor.size(1), : tensor.size(2)] = tensor
    return padded


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = build_model_config_from_checkpoint(payload)
    model_config.return_hidden_traces = bool(args.save_attention)

    device, _ = resolve_torch_device(args.device)
    model = VisualSeq2SeqLSTM(model_config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    saved_args = payload.get("args") or {}
    camera_name = args.camera_name or saved_args.get("camera_name", "freecam")
    include_reset_frame = (
        bool(saved_args.get("include_reset_frame", False))
        if args.include_reset_frame is None
        else bool(args.include_reset_frame)
    )
    dataset = RobosuiteVisualCorsiDataset(
        args.data_root,
        camera_name=camera_name,
        include_reset_frame=include_reset_frame,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_visual_batch)
    loss_fn = nn.CrossEntropyLoss(ignore_index=model_config.target_pad_value)

    predictions_all = []
    targets_all = []
    target_lengths_all = []
    frame_lengths_all = []
    masks_all = []
    losses = []
    attention_all = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            if args.mode == "teacher_forced":
                outputs = model(
                    frames=batch["frames"],
                    frame_lengths=batch["frame_lengths"],
                    targets=batch["targets"],
                    target_lengths=batch["target_lengths"],
                    return_hidden_traces=args.save_attention,
                    teacher_forcing_ratio=1.0,
                )
                logits = outputs["logits"]
                predictions = logits.argmax(dim=-1)
                if args.save_attention and "hidden_traces" in outputs:
                    attention_all.append(outputs["hidden_traces"]["attention_trace"].detach().cpu())
            else:
                outputs = model.greedy_decode(
                    frames=batch["frames"],
                    frame_lengths=batch["frame_lengths"],
                    target_lengths=batch["target_lengths"],
                    max_steps=batch["targets"].size(1),
                    return_hidden_traces=args.save_attention,
                )
                if isinstance(outputs, dict):
                    predictions = outputs["predictions"]
                    if args.save_attention:
                        attention_all.append(outputs["hidden_traces"]["attention_trace"].detach().cpu())
                    logits = None
                else:
                    predictions = outputs
                    logits = None

            if logits is not None:
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch["targets"].reshape(-1))
                losses.append(float(loss.item()))
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

    max_length = int(target_lengths.max().item()) if target_lengths.numel() else 0
    min_length = int(target_lengths.min().item()) if target_lengths.numel() else 0
    metrics = summarize_sequence_metrics(
        predictions,
        targets,
        target_lengths,
        mask,
        min_length=min_length,
        max_length=max_length,
    )
    if losses:
        metrics["loss"] = float(np.mean(losses))

    target_lists = [row[: int(length)].tolist() for row, length in zip(targets, target_lengths)]
    prediction_lists = [row[: int(length)].tolist() for row, length in zip(predictions, target_lengths)]
    error_rows = classify_trials(target_lists, prediction_lists, target_lengths.tolist())

    attention_rows: List[Dict[str, Any]] = []
    if attention_all:
        max_attention_steps = max(max_steps, max(tensor.size(1) for tensor in attention_all))
        max_encoder_steps = max(tensor.size(2) for tensor in attention_all)
        attention_tensor = torch.cat(
            [
                pad_attention_to_shape(
                    tensor,
                    max_steps=max_attention_steps,
                    max_encoder_steps=max_encoder_steps,
                )
                for tensor in attention_all
            ],
            dim=0,
        )
        attention_np = attention_tensor.numpy()
        attention_rows = analyze_attention_weights(
            attention_np,
            target_lengths.numpy(),
            encoder_lengths=frame_lengths.numpy(),
        )
        metrics["attention"] = summarize_attention_rows(attention_rows)
    else:
        attention_np = None

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / f"eval_{args.mode}_{Path(args.data_root).name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_predictions:
        prediction_rows = []
        for index, (target, pred, length) in enumerate(zip(target_lists, prediction_lists, target_lengths.tolist())):
            prediction_rows.append(
                {
                    "trial_index": index,
                    "length": int(length),
                    "target_sequence": " ".join(map(str, target)),
                    "predicted_sequence": " ".join(map(str, pred)),
                    "exact_match": target == pred,
                }
            )
        write_rows_csv(output_dir / "predictions.csv", prediction_rows)
        write_rows_csv(output_dir / "error_taxonomy.csv", error_rows)
    if attention_rows:
        write_rows_csv(output_dir / "attention_diagnostics.csv", attention_rows)
    if args.save_attention and attention_np is not None:
        np.savez_compressed(
            output_dir / "attention_weights.npz",
            attention_weights=attention_np,
            target_lengths=target_lengths.numpy(),
            frame_lengths=frame_lengths.numpy(),
        )

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "data_root": args.data_root,
        "split": args.split,
        "eval_mode": args.mode,
        "attention_type": model_config.attention_type,
        "metrics": metrics,
    }
    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    flat_summary = {
        "checkpoint_path": str(checkpoint_path),
        "data_root": args.data_root,
        "eval_mode": args.mode,
        "attention_type": model_config.attention_type,
        "token_accuracy": metrics["token_accuracy"],
        "full_sequence_accuracy": metrics["full_sequence_accuracy"],
    }
    if "attention" in metrics:
        flat_summary.update(metrics["attention"])
    write_rows_csv(output_dir / "summary_metrics.csv", [flat_summary])
    print(json.dumps(flat_summary, allow_nan=True))


if __name__ == "__main__":
    main()
