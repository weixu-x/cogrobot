"""Train and evaluate the 7-joint Corsi motion baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from corsi.experiments.corsi_motion_baseline.canonicalize import load_config
from corsi.experiments.corsi_motion_baseline.dataset import (
    CorsiMotionCanonicalDataset,
    collate_motion_prediction_batch,
)
from corsi.experiments.corsi_motion_baseline.model import (
    build_model,
    parameter_count,
    persistence_prediction,
)


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


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        raise ValueError("empty loss mask")
    diff = pred[mask] - target[mask]
    return torch.mean(diff * diff)


def _denormalize_joints(tensor: torch.Tensor, manifest: dict[str, Any]) -> torch.Tensor:
    stats = manifest["normalization"]
    mean = torch.as_tensor(stats["joint_mean"], dtype=tensor.dtype, device=tensor.device).reshape(1, 1, -1)
    std = torch.as_tensor(stats["joint_std"], dtype=tensor.dtype, device=tensor.device).reshape(1, 1, -1)
    return tensor * std + mean


def _metric_bucket(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(str(row[key]), []).append(float(row["sqerr_mean"]))
    return {
        bucket: {
            "mse": float(np.mean(values)),
            "rmse": float(math.sqrt(np.mean(values))),
            "count": int(len(values)),
        }
        for bucket, values in sorted(buckets.items())
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module | None,
    loader: DataLoader,
    *,
    device: torch.device,
    manifest: dict[str, Any],
    mode: str = "normal",
    model_type: str = "visual_joint",
) -> dict[str, Any]:
    if model is not None:
        model.eval()
    total_sq = 0.0
    total_count = 0
    total_abs_deg = None
    per_step_rows: list[dict[str, Any]] = []
    joint_dim = int(manifest["joint_dim"])
    for batch in loader:
        batch = move_batch(batch, device)
        images = batch["images"]
        joints = batch["joints"]
        if mode == "shuffled_vision":
            # Shuffle complete visual sequences among episodes of the same length inside the batch.
            shuffled = images.clone()
            lengths = batch["length"].detach().cpu().tolist()
            for length in sorted(set(lengths)):
                indices = [index for index, value in enumerate(lengths) if value == length]
                if len(indices) > 1:
                    rotated = indices[1:] + indices[:1]
                    shuffled[indices] = images[rotated]
            images = shuffled
        if model is None:
            pred = persistence_prediction(joints)
        else:
            outputs = model(
                images=images if model_type == "visual_joint" else None,
                joints=joints,
                valid_mask=batch["valid_mask"],
                zero_vision=(mode == "zero_vision"),
            )
            pred = outputs["pred_joints_next"]  # type: ignore[index]
        target = batch["targets_next"]
        mask = batch["loss_mask"]
        diff = pred[mask] - target[mask]
        total_sq += float(torch.sum(diff * diff).item())
        total_count += int(diff.numel())

        pred_phys = _denormalize_joints(pred, manifest)
        target_phys = _denormalize_joints(target, manifest)
        abs_deg = torch.abs(pred_phys[mask] - target_phys[mask]) * (180.0 / math.pi)
        if total_abs_deg is None:
            total_abs_deg = torch.sum(abs_deg, dim=0).detach().cpu()
        else:
            total_abs_deg += torch.sum(abs_deg, dim=0).detach().cpu()

        sqerr_step = torch.mean((pred - target) ** 2, dim=-1)
        for batch_index, seq_id in enumerate(batch["seq_id"]):
            valid_indices = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
            for step_tensor in valid_indices:
                step = int(step_tensor.item())
                per_step_rows.append(
                    {
                        "seq_id": seq_id,
                        "length": int(batch["length"][batch_index].item()),
                        "rank": int(batch["rank"][batch_index, step].item()),
                        "boundary": bool(batch["segment_boundary_transition"][batch_index, step].item()),
                        "within_segment": bool(batch["within_segment_transition"][batch_index, step].item()),
                        "sqerr_mean": float(sqerr_step[batch_index, step].item()),
                    }
                )

    mse = total_sq / max(total_count, 1)
    joint_mae_deg = (
        (total_abs_deg / max(total_count // joint_dim, 1)).tolist()
        if total_abs_deg is not None
        else [0.0] * joint_dim
    )
    boundary_rows = [row for row in per_step_rows if row["boundary"]]
    within_rows = [row for row in per_step_rows if row["within_segment"]]

    def mse_from_rows(rows: list[dict[str, Any]]) -> float:
        return float(np.mean([row["sqerr_mean"] for row in rows])) if rows else 0.0

    return {
        "normalized_mse": float(mse),
        "normalized_rmse": float(math.sqrt(mse)),
        "joint_mae_deg": [float(value) for value in joint_mae_deg],
        "joint_mae_deg_mean": float(np.mean(joint_mae_deg)) if joint_mae_deg else 0.0,
        "per_length": _metric_bucket(per_step_rows, "length"),
        "per_rank": _metric_bucket(per_step_rows, "rank"),
        "within_segment_mse": mse_from_rows(within_rows),
        "within_segment_rmse": float(math.sqrt(mse_from_rows(within_rows))) if within_rows else 0.0,
        "segment_boundary_mse": mse_from_rows(boundary_rows),
        "segment_boundary_rmse": float(math.sqrt(mse_from_rows(boundary_rows))) if boundary_rows else 0.0,
        "valid_transition_count": int(len(per_step_rows)),
        "mode": mode,
    }


def make_loader(
    manifest_path: Path,
    *,
    split: str,
    batch_size: int,
    shuffle: bool,
    seq_ids: list[str] | None = None,
    load_images: bool = True,
) -> DataLoader:
    dataset = CorsiMotionCanonicalDataset(manifest_path, split=split, seq_ids=seq_ids, load_images=load_images)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_motion_prediction_batch,
    )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
    model_type: str,
    seed: int,
    manifest: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_type": model_type,
            "seed": int(seed),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "scheduler_state_dict": None,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy_rng_state": np.random.get_state(),
            "python_random_state": random.getstate(),
            "best_metric": float(best_metric),
            "config": config,
            "git_commit": git_commit(),
            "canonical_fingerprint": manifest["canonical_fingerprint"],
            "normalization": manifest["normalization"],
            "joint_names": manifest["joint_names"],
        },
        path,
    )


def train_one_run(
    config: dict[str, Any],
    *,
    model_type: str,
    seed: int,
    run_dir: Path,
    max_epochs_override: int | None = None,
    overfit_episodes: int = 0,
    resume: bool = False,
) -> dict[str, Any]:
    set_seed(seed)
    device = resolve_device(str(config.get("device", "auto")))
    manifest_path = Path(str(config["canonical_root"])) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batch_size = int(config.get("batch_size", 16))
    train_seq_ids = None
    val_split = "val"
    if overfit_episodes > 0:
        train_seq_ids = list(manifest["split"]["train"])[: int(overfit_episodes)]
        val_split = "train"
    train_loader = make_loader(
        manifest_path,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        seq_ids=train_seq_ids,
        load_images=(model_type == "visual_joint"),
    )
    val_loader = make_loader(
        manifest_path,
        split=val_split,
        batch_size=batch_size,
        shuffle=False,
        seq_ids=train_seq_ids if overfit_episodes > 0 else None,
        load_images=(model_type == "visual_joint"),
    )
    model = build_model(model_type, joint_dim=int(manifest["joint_dim"]), hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    max_epochs = int(max_epochs_override or config.get("max_epochs", 300))
    min_epochs = int(config.get("min_epochs", 100)) if overfit_episodes <= 0 else 1
    patience = int(config.get("early_stopping_patience", 50)) if overfit_episodes <= 0 else max_epochs
    grad_clip = float(config.get("gradient_clip_norm", 1.0))
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    summary_path = run_dir / "summary.json"
    start_epoch = 1
    best_metric = float("inf")
    if resume and latest_path.exists():
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(payload["scaler_state_dict"])
        if payload.get("torch_rng_state") is not None:
            torch_rng_state = payload["torch_rng_state"]
            if isinstance(torch_rng_state, torch.Tensor):
                torch.set_rng_state(torch_rng_state.detach().cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng_state_all"):
            cuda_rng_state_all = [
                state.detach().cpu() if isinstance(state, torch.Tensor) else state
                for state in payload["cuda_rng_state_all"]
            ]
            torch.cuda.set_rng_state_all(cuda_rng_state_all)
        if payload.get("numpy_rng_state") is not None:
            np.random.set_state(payload["numpy_rng_state"])
        if payload.get("python_random_state") is not None:
            random.setstate(payload["python_random_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_metric = float(payload["best_metric"])
        if start_epoch > max_epochs:
            if summary_path.exists():
                return json.loads(summary_path.read_text(encoding="utf-8"))
            return {
                "model_type": model_type,
                "seed": int(seed),
                "run_dir": str(run_dir),
                "best_checkpoint": str(best_path),
                "latest_checkpoint": str(latest_path),
                "best_val_normalized_rmse": float(best_metric),
                "epochs_completed": int(payload["epoch"]),
                "runtime_seconds": 0.0,
                "device": str(device),
                "parameter_count": parameter_count(model),
                "canonical_fingerprint": manifest["canonical_fingerprint"],
                "git_commit": git_commit(),
                "hardware": {
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                    "cpu_count": os.cpu_count(),
                },
            }

    curves = []
    curves_path = run_dir / "curves.json"
    if resume and curves_path.exists():
        try:
            curves = json.loads(curves_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            curves = []
    start_time = time.time()
    epochs_without_improvement = 0
    stopped_reason = "max_epochs_reached"
    if resume and curves:
        val_metrics = [float(row["val_normalized_rmse"]) for row in curves]
        best_epoch_index = int(np.argmin(val_metrics))
        epochs_without_improvement = int(curves[-1]["epoch"]) - int(curves[best_epoch_index]["epoch"])
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    images=batch["images"] if model_type == "visual_joint" else None,
                    joints=batch["joints"],
                    valid_mask=batch["valid_mask"],
                )
                loss = masked_mse(outputs["pred_joints_next"], batch["targets_next"], batch["loss_mask"])  # type: ignore[arg-type]
            if not torch.isfinite(loss):
                stopped_reason = f"non_finite_loss_epoch_{epoch}"
                break
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    stopped_reason = f"non_finite_gradient_{name}_epoch_{epoch}"
                    break
            if stopped_reason.startswith("non_finite"):
                break
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item())
            train_steps += 1
        if stopped_reason.startswith("non_finite"):
            break
        val_metrics = evaluate_model(model, val_loader, device=device, manifest=manifest, model_type=model_type)
        train_loss = train_loss_sum / max(train_steps, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_normalized_rmse": val_metrics["normalized_rmse"],
            "val_normalized_mse": val_metrics["normalized_mse"],
        }
        curves.append(row)
        improved = float(val_metrics["normalized_rmse"]) < best_metric
        if improved:
            best_metric = float(val_metrics["normalized_rmse"])
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                config=config,
                model_type=model_type,
                seed=seed,
                manifest=manifest,
            )
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_metric,
            config=config,
            model_type=model_type,
            seed=seed,
            manifest=manifest,
        )
        curves_path.write_text(json.dumps(curves, indent=2), encoding="utf-8")
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            stopped_reason = "early_stopping"
            break

    runtime = time.time() - start_time
    summary = {
        "model_type": model_type,
        "seed": int(seed),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "best_val_normalized_rmse": float(best_metric),
        "epochs_completed": int(curves[-1]["epoch"] if curves else 0),
        "stopped_reason": stopped_reason,
        "runtime_seconds": float(runtime),
        "device": str(device),
        "parameter_count": parameter_count(model),
        "canonical_fingerprint": manifest["canonical_fingerprint"],
        "git_commit": git_commit(),
        "hardware": {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "cpu_count": os.cpu_count(),
        },
    }
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Corsi 7-joint motion baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-type", choices=["visual_joint", "joint_only"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--overfit-episodes", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    run_name = args.run_name or f"{args.model_type}_seed{args.seed}"
    run_dir = Path(str(config["output_root"])) / run_name
    result = train_one_run(
        config,
        model_type=args.model_type,
        seed=int(args.seed),
        run_dir=run_dir,
        max_epochs_override=args.max_epochs,
        overfit_episodes=int(args.overfit_episodes),
        resume=bool(args.resume),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
