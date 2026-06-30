"""Frozen grounding and item-embedding analysis for paused Corsi V2 D_mem64."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.experiments.corsi_memory_recall_v2.model import build_model, parameter_count
from corsi.experiments.corsi_memory_recall_v2.train import _call_model, make_loader, move_to_device


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stats_tensor(manifest: dict[str, Any], key: str, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    stats = manifest["normalization"]
    mean = torch.as_tensor(stats[f"{key}_mean"], dtype=torch.float32, device=device).view(1, 1, 1, -1)
    std = torch.as_tensor(stats[f"{key}_std"], dtype=torch.float32, device=device).view(1, 1, 1, -1)
    return mean, std


def _init_metric() -> dict[str, Any]:
    return {
        "scalar_abs_sum": 0.0,
        "scalar_sq_sum": 0.0,
        "scalar_count": 0,
        "vector_l2_sum": 0.0,
        "vector_l2_sq_sum": 0.0,
        "vector_count": 0,
        "per_dim_abs_sum": None,
        "per_dim_sq_sum": None,
    }


def _update_metric(metric: dict[str, Any], pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
    valid = mask.to(dtype=torch.bool)
    if int(valid.sum().item()) == 0:
        return
    pred_valid = pred[valid]
    target_valid = target[valid]
    diff = pred_valid - target_valid
    abs_diff = diff.abs()
    sq_diff = diff.square()
    if metric["per_dim_abs_sum"] is None:
        dim = int(diff.shape[-1])
        metric["per_dim_abs_sum"] = [0.0] * dim
        metric["per_dim_sq_sum"] = [0.0] * dim
    metric["scalar_abs_sum"] += float(abs_diff.sum().detach().cpu().item())
    metric["scalar_sq_sum"] += float(sq_diff.sum().detach().cpu().item())
    metric["scalar_count"] += int(diff.numel())
    vector_l2 = torch.linalg.vector_norm(diff, ord=2, dim=-1)
    metric["vector_l2_sum"] += float(vector_l2.sum().detach().cpu().item())
    metric["vector_l2_sq_sum"] += float(vector_l2.square().sum().detach().cpu().item())
    metric["vector_count"] += int(vector_l2.numel())
    per_dim_abs = abs_diff.sum(dim=0).detach().cpu().numpy()
    per_dim_sq = sq_diff.sum(dim=0).detach().cpu().numpy()
    for index, value in enumerate(per_dim_abs.tolist()):
        metric["per_dim_abs_sum"][index] += float(value)
    for index, value in enumerate(per_dim_sq.tolist()):
        metric["per_dim_sq_sum"][index] += float(value)


def _finalize_metric(metric: dict[str, Any]) -> dict[str, Any]:
    scalar_count = max(int(metric["scalar_count"]), 1)
    vector_count = max(int(metric["vector_count"]), 1)
    dim = len(metric["per_dim_abs_sum"] or [])
    return {
        "scalar_mae": float(metric["scalar_abs_sum"] / scalar_count),
        "scalar_rmse": float(math.sqrt(metric["scalar_sq_sum"] / scalar_count)),
        "vector_l2_mean": float(metric["vector_l2_sum"] / vector_count),
        "vector_l2_rmse": float(math.sqrt(metric["vector_l2_sq_sum"] / vector_count)),
        "frame_vectors": int(metric["vector_count"]),
        "scalar_values": int(metric["scalar_count"]),
        "per_dim_mae": [
            float(value / vector_count)
            for value in (metric["per_dim_abs_sum"] or [0.0] * dim)
        ],
        "per_dim_rmse": [
            float(math.sqrt(value / vector_count))
            for value in (metric["per_dim_sq_sum"] or [0.0] * dim)
        ],
    }


def _ridge_probe(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, *, num_classes: int) -> np.ndarray:
    train_mean = train_x.mean(axis=0, keepdims=True)
    train_std = train_x.std(axis=0, keepdims=True)
    train_std[train_std < 1e-6] = 1.0
    x_train = (train_x - train_mean) / train_std
    x_eval = (eval_x - train_mean) / train_std
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=x_train.dtype)], axis=1)
    x_eval = np.concatenate([x_eval, np.ones((x_eval.shape[0], 1), dtype=x_eval.dtype)], axis=1)
    y_onehot = np.zeros((train_y.shape[0], num_classes), dtype=np.float64)
    y_onehot[np.arange(train_y.shape[0]), train_y.astype(int)] = 1.0
    reg = np.eye(x_train.shape[1], dtype=np.float64) * 1e-3
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ y_onehot)
    return np.argmax(x_eval @ weights, axis=1)


def _nearest_centroid(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, *, num_classes: int) -> np.ndarray:
    train_mean = train_x.mean(axis=0, keepdims=True)
    train_std = train_x.std(axis=0, keepdims=True)
    train_std[train_std < 1e-6] = 1.0
    x_train = (train_x - train_mean) / train_std
    x_eval = (eval_x - train_mean) / train_std
    centroids = np.stack([x_train[train_y == label].mean(axis=0) for label in range(num_classes)], axis=0)
    distances = ((x_eval[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    return np.argmin(distances, axis=1)


def _accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred.astype(int) == target.astype(int)).mean()) if target.size else 0.0


def _confusion(pred: np.ndarray, target: np.ndarray, *, num_classes: int) -> list[list[int]]:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, guess in zip(target.astype(int), pred.astype(int), strict=False):
        matrix[int(truth), int(guess)] += 1
    return matrix.tolist()


def _save_pca_figure(
    train_x: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    output_path: Path,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        return {"path": str(output_path), "created": False, "error": str(exc)}

    mean = train_x.mean(axis=0, keepdims=True)
    centered_train = train_x - mean
    _, _, vh = np.linalg.svd(centered_train, full_matrices=False)
    basis = vh[:2].T
    projected = (eval_x - mean) @ basis
    explained = np.var(centered_train @ basis, axis=0) / np.var(centered_train, axis=0).sum()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=140)
    cmap = plt.get_cmap("tab10")
    for label in sorted(set(eval_y.astype(int).tolist())):
        mask = eval_y == label
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=10,
            alpha=0.65,
            color=cmap(label % 10),
            label=f"block {label}",
            linewidths=0,
        )
    ax.set_title("D_mem64 direct Stage 2 item embeddings: test split PCA")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% train variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% train variance)")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return {
        "path": str(output_path),
        "created": True,
        "explained_variance_ratio": [float(explained[0]), float(explained[1])],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--pca-figure", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    config = _load_json(args.config)
    manifest_path = Path(str(config["canonical_root"])) / "manifest.json"
    manifest = _load_json(manifest_path)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    model = build_model(config=config, stage=2, manifest=manifest).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    norm_stats = {
        key: _stats_tensor(manifest, key, device=device)
        for key in ("joint", "ee_pose", "ee_xy")
    }

    split_metrics: dict[str, Any] = {}
    embedding_by_split: dict[str, dict[str, list[np.ndarray]]] = {}
    with torch.no_grad():
        for split in ("train", "val", "test"):
            loader = make_loader(
                manifest_path,
                split=split,
                batch_size=int(args.batch_size),
                shuffle=False,
                cache=False,
                num_workers=int(args.num_workers),
                pin_memory=device.type == "cuda",
                persistent_workers=int(args.num_workers) > 0,
                prefetch_factor=2 if int(args.num_workers) > 0 else None,
            )
            metrics = {
                "joint_normalized": _init_metric(),
                "ee_pose_normalized": _init_metric(),
                "ee_xy_normalized": _init_metric(),
                "joint_physical": _init_metric(),
                "ee_pose_physical": _init_metric(),
                "ee_xy_physical": _init_metric(),
            }
            embeddings: list[np.ndarray] = []
            labels: list[np.ndarray] = []
            lengths: list[np.ndarray] = []
            frame_vectors = 0
            segments = 0
            sequences = 0
            for batch in loader:
                batch = move_to_device(batch, device)
                outputs = _call_model(model, batch, stage=2)
                frame_mask = batch["model_inputs"]["frame_mask"]
                segment_mask = batch["model_inputs"]["segment_mask"]
                for field, output_key in (
                    ("joint", "pred_joint"),
                    ("ee_pose", "pred_ee_pose"),
                    ("ee_xy", "pred_ee_xy"),
                ):
                    pred_norm = outputs[output_key]
                    target_norm = batch["targets"][field]
                    _update_metric(metrics[f"{field}_normalized"], pred_norm, target_norm, frame_mask)
                    mean, std = norm_stats[field]
                    pred_phys = pred_norm * std + mean
                    target_phys = batch["targets"][f"physical_{field}"]
                    _update_metric(metrics[f"{field}_physical"], pred_phys, target_phys, frame_mask)
                item = outputs["item_embeddings"].detach()
                block_id = batch["metadata"]["block_id"]
                valid_segments = segment_mask.to(dtype=torch.bool)
                embeddings.append(item[valid_segments].detach().cpu().numpy().astype(np.float64))
                labels.append(block_id[valid_segments].detach().cpu().numpy().astype(np.int64))
                lengths.append(batch["metadata"]["length"].detach().cpu().numpy().astype(np.int64))
                frame_vectors += int(frame_mask.sum().item())
                segments += int(valid_segments.sum().item())
                sequences += int(segment_mask.shape[0])

            split_metrics[split] = {
                "sequences": sequences,
                "segments": segments,
                "valid_frame_vectors": frame_vectors,
                "aux": {key: _finalize_metric(value) for key, value in metrics.items()},
            }
            embedding_by_split[split] = {
                "x": [np.concatenate(embeddings, axis=0)],
                "y": [np.concatenate(labels, axis=0)],
                "sequence_lengths": [np.concatenate(lengths, axis=0)],
            }

    train_x = embedding_by_split["train"]["x"][0]
    train_y = embedding_by_split["train"]["y"][0]
    probe_results: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        x = embedding_by_split[split]["x"][0]
        y = embedding_by_split[split]["y"][0]
        centroid_pred = _nearest_centroid(train_x, train_y, x, num_classes=int(config["num_blocks"]))
        ridge_pred = _ridge_probe(train_x, train_y, x, num_classes=int(config["num_blocks"]))
        probe_results[split] = {
            "segments": int(y.shape[0]),
            "nearest_centroid_accuracy": _accuracy(centroid_pred, y),
            "ridge_linear_probe_accuracy": _accuracy(ridge_pred, y),
            "ridge_confusion_matrix": _confusion(ridge_pred, y, num_classes=int(config["num_blocks"])),
            "nearest_centroid_confusion_matrix": _confusion(centroid_pred, y, num_classes=int(config["num_blocks"])),
        }

    pca_info = _save_pca_figure(
        train_x,
        embedding_by_split["test"]["x"][0],
        embedding_by_split["test"]["y"][0],
        Path(args.pca_figure),
    )

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_extra": checkpoint.get("extra", {}),
        "device": str(device),
        "model": {
            "memory_dim": int(config["memory_dim"]),
            "memory_slot_dim": int(config["memory_slot_dim"]),
            "memory_write_mode": config["memory_write_mode"],
            "recall_readout_mode": config["recall_readout_mode"],
            "parameter_count": parameter_count(model),
        },
        "split_metrics": split_metrics,
        "item_embedding_probe": probe_results,
        "pca": pca_info,
        "interpretation": {
            "stage1_grounding_question": (
                "If aux physical/normalized errors are low and item probes are high, direct Stage 2 learned the "
                "grounding signals Stage 1 would have pretrained. If aux errors are high but item probes are high, "
                "the recall task is likely using visual/item information while motion aux grounding remains weak."
            )
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    def aux_line(split: str, field: str) -> str:
        aux = split_metrics[split]["aux"]
        norm = aux[f"{field}_normalized"]
        phys = aux[f"{field}_physical"]
        return (
            f"| {split} | {field} | {norm['scalar_mae']:.6f} | {norm['scalar_rmse']:.6f} | "
            f"{phys['scalar_mae']:.6f} | {phys['scalar_rmse']:.6f} | {phys['vector_l2_mean']:.6f} | "
            f"{phys['vector_l2_rmse']:.6f} |"
        )

    lines = [
        "# D_mem64 Direct Stage 2 Grounding Analysis - 2026-06-30",
        "",
        "This is a frozen checkpoint analysis. It does not resume model training and does not update weights.",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        f"Checkpoint stage/epoch: `{checkpoint.get('stage')}` / `{checkpoint.get('epoch')}`",
        "",
        "## Motion Aux Heads",
        "",
        "| Split | Field | Norm scalar MAE | Norm scalar RMSE | Physical scalar MAE | Physical scalar RMSE | Physical vector L2 mean | Physical vector L2 RMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "val", "test"):
        for field in ("joint", "ee_pose", "ee_xy"):
            lines.append(aux_line(split, field))
    lines.extend(
        [
            "",
            "## Item Embedding Block Separability",
            "",
            "| Split | Segments | Nearest-centroid acc | Ridge linear-probe acc |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for split in ("train", "val", "test"):
        probe = probe_results[split]
        lines.append(
            f"| {split} | {probe['segments']} | {probe['nearest_centroid_accuracy']:.6f} | "
            f"{probe['ridge_linear_probe_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "PCA figure:",
            "",
            f"`{pca_info['path']}`" if pca_info.get("created") else f"not created: {pca_info.get('error')}",
            "",
            "## Interpretation",
            "",
            "- Stage 1 would warm-start the visual/motor grounding prefixes.",
            "- Low aux errors indicate those prefixes and aux heads were learned during direct Stage 2.",
            "- High item-probe accuracy indicates item embeddings already linearly separate the 9 block identities.",
            "- If item-probe accuracy is high but aux errors are poor, motion grounding is not necessary for recall in this setup.",
        ]
    )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md), "pca": pca_info}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
