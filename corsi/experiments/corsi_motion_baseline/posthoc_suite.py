"""Posthoc convergence and accuracy suite for the Corsi 7-joint baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from corsi.experiments.corsi_motion_baseline.canonicalize import load_config
from corsi.experiments.corsi_motion_baseline.dataset import CorsiMotionCanonicalDataset
from corsi.experiments.corsi_motion_baseline.model import build_model
from corsi.experiments.corsi_motion_baseline.train import resolve_device, train_one_run


SEEDS = [0, 5, 10, 15, 20]
MODEL_TYPES = ["visual_joint", "joint_only"]
VISUAL_MODES = ["normal", "shuffled_vision", "zero_vision"]
STEP_THRESHOLDS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
TOLERANCE_CURVE_THRESHOLDS = [round(value, 2) for value in np.arange(0.0, 5.0 + 1e-9, 0.05)]
POSTHOC_ROOT = Path("corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1")
AUDIT_REPORT_PATH = Path("reports/corsi_motion_convergence_audit.md")
FINAL_REPORT_PATH = Path("reports/corsi_motion_convergence_accuracy.md")


@dataclass(frozen=True)
class RunRef:
    model_type: str
    seed: int
    run_name: str
    run_dir: Path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2), encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def original_run_refs(config: dict[str, Any]) -> list[RunRef]:
    root = Path(str(config["output_root"]))
    refs = []
    for model_type in MODEL_TYPES:
        for seed in SEEDS:
            run_name = f"{model_type}_seed{seed}_full"
            refs.append(RunRef(model_type=model_type, seed=seed, run_name=run_name, run_dir=root / run_name))
    return refs


def extension_run_name(ref: RunRef) -> str:
    return f"{ref.model_type}_seed{ref.seed}_extended_continuation_from_best"


def load_curves(run_dir: Path) -> list[dict[str, float]]:
    curves_path = run_dir / "curves.json"
    if not curves_path.exists():
        return []
    rows = read_json(curves_path)
    return [dict(row) for row in rows]


def ols_slope(values: Iterable[float]) -> float:
    y = np.asarray(list(values), dtype=np.float64)
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=np.float64)
    x_centered = x - float(x.mean())
    denom = float(np.sum(x_centered * x_centered))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x_centered * (y - float(y.mean()))) / denom)


def theil_sen_slope(values: Iterable[float]) -> float:
    y = np.asarray(list(values), dtype=np.float64)
    if y.size < 2:
        return 0.0
    slopes = []
    for start in range(y.size):
        for end in range(start + 1, y.size):
            slopes.append(float((y[end] - y[start]) / (end - start)))
    return float(np.median(np.asarray(slopes, dtype=np.float64))) if slopes else 0.0


def _median_edge_change(values: list[float]) -> float:
    edge = min(10, len(values))
    if edge == 0:
        return 0.0
    first = float(np.median(values[:edge]))
    last = float(np.median(values[-edge:]))
    return 0.0 if first == 0.0 else float((first - last) / first)


def convergence_stats(curves: list[dict[str, Any]], window: int) -> dict[str, Any]:
    if not curves:
        return {
            "available": False,
            "window": 0,
            "confidence": "none",
            "classification": "UNCERTAIN",
        }
    values = [float(row["val_normalized_rmse"]) for row in curves]
    train = [float(row["train_loss"]) for row in curves]
    actual_window = min(int(window), len(values))
    confidence = "normal"
    if actual_window < int(window):
        if actual_window >= 20:
            confidence = "lower"
        else:
            confidence = "insufficient"
    val_window = values[-actual_window:]
    train_window = train[-actual_window:]
    ols = ols_slope(val_window)
    robust = theil_sen_slope(val_window)
    mean_val = float(np.mean(val_window)) if val_window else 0.0
    rel_change = _median_edge_change(val_window)
    predicted_ols_decrease = 0.0 if mean_val == 0.0 else max(0.0, -ols * (actual_window - 1) / mean_val)
    predicted_robust_decrease = 0.0 if mean_val == 0.0 else max(0.0, -robust * (actual_window - 1) / mean_val)
    train_rel_change = _median_edge_change(train_window)
    val_worsening = -rel_change
    best_index = int(np.argmin(np.asarray(values, dtype=np.float64)))
    final_gap = float(train[-1] - values[-1] * values[-1])

    if val_worsening >= 0.005 and train_rel_change > 0.0:
        classification = "OVERFITTING"
    elif rel_change < 0.005 and predicted_ols_decrease < 0.005 and predicted_robust_decrease < 0.005:
        classification = "CONVERGED"
    elif rel_change >= 0.01 or (predicted_ols_decrease >= 0.01 and predicted_robust_decrease >= 0.01):
        classification = "STILL_IMPROVING"
    else:
        classification = "UNCERTAIN"

    return {
        "available": True,
        "window": int(actual_window),
        "requested_window": int(window),
        "confidence": confidence,
        "ols_slope": float(ols),
        "theil_sen_slope": float(robust),
        "normalized_ols_slope": float(0.0 if mean_val == 0.0 else ols / mean_val),
        "normalized_theil_sen_slope": float(0.0 if mean_val == 0.0 else robust / mean_val),
        "predicted_relative_decrease_ols": float(predicted_ols_decrease),
        "predicted_relative_decrease_theil_sen": float(predicted_robust_decrease),
        "relative_change": float(rel_change),
        "best_epoch": int(curves[best_index]["epoch"]),
        "epochs_since_best": int(curves[-1]["epoch"]) - int(curves[best_index]["epoch"]),
        "best_val_rmse": float(values[best_index]),
        "final_val_rmse": float(values[-1]),
        "final_train_loss": float(train[-1]),
        "train_validation_gap_train_loss_minus_val_mse": float(final_gap),
        "train_loss_relative_change": float(train_rel_change),
        "validation_worsens_while_train_decreases": bool(val_worsening >= 0.005 and train_rel_change > 0.0),
        "classification": classification,
    }


def checkpoint_schema(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    keys = sorted(payload.keys())
    optimizer_lrs = []
    if "optimizer_state_dict" in payload:
        optimizer_lrs = [
            float(group.get("lr"))
            for group in payload["optimizer_state_dict"].get("param_groups", [])
            if "lr" in group
        ]
    return {
        "exists": True,
        "path": str(path),
        "keys": keys,
        "epoch": int(payload.get("epoch", -1)),
        "best_metric": float(payload.get("best_metric", float("nan"))),
        "has_model_state": "model_state_dict" in payload,
        "has_optimizer_state": "optimizer_state_dict" in payload,
        "optimizer_learning_rates": optimizer_lrs,
        "has_scaler_state": payload.get("scaler_state_dict") is not None,
        "has_scheduler_state": payload.get("scheduler_state_dict") is not None,
        "has_torch_rng_state": payload.get("torch_rng_state") is not None,
        "has_cuda_rng_state": bool(payload.get("cuda_rng_state_all")),
        "has_numpy_rng_state": payload.get("numpy_rng_state") is not None,
        "has_python_random_state": payload.get("python_random_state") is not None,
    }


def exact_resume_possible(schema: dict[str, Any]) -> bool:
    return bool(
        schema.get("exists")
        and schema.get("has_model_state")
        and schema.get("has_optimizer_state")
        and schema.get("has_scaler_state")
        and schema.get("has_scheduler_state")
        and schema.get("has_torch_rng_state")
        and schema.get("has_numpy_rng_state")
        and schema.get("has_python_random_state")
    )


def audit_loss_mask(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    totals = {
        "samples_checked": 0,
        "transition_count": 0,
        "within_segment_transition_count": 0,
        "segment_boundary_transition_count": 0,
        "last_step_excluded_count": 0,
    }
    examples = []
    for sample in manifest["samples"]:
        path = Path(str(sample["canonical_path"]))
        if not path.exists():
            path = Path(str(manifest["canonical_root"])) / "episodes" / path.name
        with np.load(path) as arrays:
            transition = arrays["transition_mask"].astype(bool)
            within = arrays["within_segment_transition"].astype(bool)
            boundary = arrays["segment_boundary_transition"].astype(bool)
        totals["samples_checked"] += 1
        totals["transition_count"] += int(transition.sum())
        totals["within_segment_transition_count"] += int(within.sum())
        totals["segment_boundary_transition_count"] += int(boundary.sum())
        totals["last_step_excluded_count"] += int((~transition)[-1])
        if len(examples) < 3:
            examples.append(
                {
                    "seq_id": sample["seq_id"],
                    "length": int(sample["length"]),
                    "transition_count": int(transition.sum()),
                    "within_segment_transition_count": int(within.sum()),
                    "segment_boundary_transition_count": int(boundary.sum()),
                    "transition_equals_within_or_boundary": bool(np.array_equal(transition, within | boundary)),
                }
            )
    totals["transition_equals_within_plus_boundary"] = (
        totals["transition_count"]
        == totals["within_segment_transition_count"] + totals["segment_boundary_transition_count"]
    )
    return {
        **totals,
        "examples": examples,
        "conclusion": "training_loss_includes_within_segment_and_segment_boundary_transitions",
    }


def _scale_points(values: list[float], width: int, height: int, pad: int) -> list[tuple[int, int]]:
    if not values:
        return []
    lo = float(min(values))
    hi = float(max(values))
    if hi == lo:
        hi = lo + 1.0
    points = []
    for index, value in enumerate(values):
        x = pad + int(round(index * (width - 2 * pad) / max(len(values) - 1, 1)))
        frac = (float(value) - lo) / (hi - lo)
        y = height - pad - int(round(frac * (height - 2 * pad)))
        points.append((x, y))
    return points


def write_curve_png(path: Path, curves: list[dict[str, Any]], title: str) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 4.2), dpi=120)
    if curves:
        train_rmse = [math.sqrt(max(float(row["train_loss"]), 0.0)) for row in curves]
        val_rmse = [float(row["val_normalized_rmse"]) for row in curves]
        epochs = [int(row["epoch"]) for row in curves]
        axis.plot(epochs, train_rmse, color="#3366cc", linewidth=1.2, alpha=0.65, label="train RMSE raw")
        axis.plot(epochs, val_rmse, color="#cc3333", linewidth=1.2, alpha=0.75, label="val RMSE raw")
        for values, color, label in [
            (train_rmse, "#3366cc", "train RMSE smoothed"),
            (val_rmse, "#cc3333", "val RMSE smoothed"),
        ]:
            if len(values) >= 7:
                kernel = np.ones((7,), dtype=np.float64) / 7.0
                smooth = np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="valid")
                axis.plot(epochs[6:], smooth, color=color, linewidth=2.0, linestyle="--", label=label)
    axis.set_title(title)
    axis.set_xlabel("epoch")
    axis.set_ylabel("RMSE")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_combined_curve_png(path: Path, runs: list[dict[str, Any]], title: str) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 4.2), dpi=120)
    for run in runs:
        if not run["curves"]:
            continue
        epochs = [int(row["epoch"]) for row in run["curves"]]
        val = [float(row["val_normalized_rmse"]) for row in run["curves"]]
        axis.plot(epochs, val, linewidth=1.2, alpha=0.65, label=run["run_name"])
        if len(val) >= 7:
            kernel = np.ones((7,), dtype=np.float64) / 7.0
            smooth = np.convolve(np.asarray(val, dtype=np.float64), kernel, mode="valid")
            axis.plot(epochs[6:], smooth, linewidth=1.8, linestyle="--", alpha=0.8)
    axis.set_title(title + " validation RMSE")
    axis.set_xlabel("epoch")
    axis.set_ylabel("validation RMSE")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def audit_and_classify(config: dict[str, Any], posthoc_root: Path = POSTHOC_ROOT) -> dict[str, Any]:
    manifest_path = Path(str(config["canonical_root"])) / "manifest.json"
    convergence_dir = posthoc_root / "convergence"
    curves_dir = convergence_dir / "curves"
    convergence_dir.mkdir(parents=True, exist_ok=True)
    loss_mask = audit_loss_mask(manifest_path)

    per_run = []
    by_model_for_curves: dict[str, list[dict[str, Any]]] = {model: [] for model in MODEL_TYPES}
    for ref in original_run_refs(config):
        curves = load_curves(ref.run_dir)
        summary = read_json(ref.run_dir / "summary.json") if (ref.run_dir / "summary.json").exists() else {}
        latest_schema = checkpoint_schema(ref.run_dir / "latest.pt")
        best_schema = checkpoint_schema(ref.run_dir / "best.pt")
        stats30 = convergence_stats(curves, 30)
        stats50 = convergence_stats(curves, 50)
        max_epochs = int(config.get("max_epochs", 300))
        stopped_before_max = bool(summary.get("epochs_completed", 0) < max_epochs)
        reached_max = bool(summary.get("epochs_completed", 0) >= max_epochs)
        extend_required = bool(reached_max and stats50["classification"] in {"STILL_IMPROVING", "UNCERTAIN"})
        early_stop_inconsistent = False
        if stopped_before_max:
            early_stop_inconsistent = bool(
                stats50["classification"] == "STILL_IMPROVING"
                and int(stats50.get("epochs_since_best", 0)) < int(config.get("early_stopping_patience", 50))
            )
        decision = {
            "extend": bool(extend_required or early_stop_inconsistent),
            "reason": "none",
        }
        if extend_required:
            decision["reason"] = "reached_epoch_300_and_classified_" + stats50["classification"]
        elif early_stop_inconsistent:
            decision["reason"] = "early_stop_inconsistent_with_validation_history"
        elif stopped_before_max:
            decision["reason"] = "stopped_before_max_by_recorded_early_stopping"
        else:
            decision["reason"] = "reached_max_but_not_improving"

        row = {
            "model_type": ref.model_type,
            "seed": int(ref.seed),
            "run_name": ref.run_name,
            "run_dir": str(ref.run_dir),
            "history_available": bool(curves),
            "history_fields": sorted(curves[0].keys()) if curves else [],
            "recoverable_history": {
                "epoch": bool(curves and "epoch" in curves[0]),
                "train_loss": bool(curves and "train_loss" in curves[0]),
                "train_rmse": "recoverable_as_sqrt_train_loss",
                "val_loss": bool(curves and "val_normalized_mse" in curves[0]),
                "val_rmse": bool(curves and "val_normalized_rmse" in curves[0]),
                "learning_rate": latest_schema.get("optimizer_learning_rates", []),
                "gradient_norm": False,
                "best_epoch": True,
                "latest_epoch": True,
                "early_stopping_state": "derived_from_curves_only",
                "optimizer_state": bool(latest_schema.get("has_optimizer_state")),
                "amp_scaler_state": bool(latest_schema.get("has_scaler_state")),
                "rng_states": bool(
                    latest_schema.get("has_torch_rng_state")
                    and latest_schema.get("has_numpy_rng_state")
                    and latest_schema.get("has_python_random_state")
                ),
            },
            "summary": summary,
            "latest_checkpoint": latest_schema,
            "best_checkpoint": best_schema,
            "exact_latest_resume_possible": exact_resume_possible(latest_schema),
            "best_warm_start_possible": bool(best_schema.get("has_model_state")),
            "stats_30": stats30,
            "stats_50": stats50,
            "final_classification": stats50["classification"],
            "continuation_decision": decision,
        }
        per_run.append(row)
        by_model_for_curves[ref.model_type].append({"run_name": ref.run_name, "curves": curves})
        write_curve_png(curves_dir / f"{ref.run_name}.png", curves, ref.run_name)

    for model_type, runs in by_model_for_curves.items():
        write_combined_curve_png(curves_dir / f"{model_type}_combined.png", runs, f"{model_type} combined")

    lineage_path = convergence_dir / "continuation_lineage.json"
    if lineage_path.exists():
        lineage = read_json(lineage_path)
    else:
        lineage = {"schema_version": "posthoc_convergence_lineage_v1", "runs": []}

    result = {
        "schema_version": "posthoc_convergence_v1",
        "config_path": "corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json",
        "posthoc_root": str(posthoc_root),
        "loss_mask_audit": loss_mask,
        "per_run": per_run,
    }
    write_json(convergence_dir / "convergence_per_run.json", result)
    write_json(lineage_path, lineage)
    write_convergence_csv(convergence_dir / "convergence_per_run.csv", per_run)
    write_convergence_audit_report(result, AUDIT_REPORT_PATH)
    return result


def write_convergence_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_type",
        "seed",
        "run_name",
        "epochs_completed",
        "classification",
        "best_epoch",
        "epochs_since_best",
        "best_val_rmse",
        "final_val_rmse",
        "relative_change_50",
        "predicted_decrease_ols_50",
        "predicted_decrease_theil_sen_50",
        "exact_latest_resume_possible",
        "extend",
        "extend_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            stats = row["stats_50"]
            writer.writerow(
                {
                    "model_type": row["model_type"],
                    "seed": row["seed"],
                    "run_name": row["run_name"],
                    "epochs_completed": row.get("summary", {}).get("epochs_completed", ""),
                    "classification": row["final_classification"],
                    "best_epoch": stats.get("best_epoch", ""),
                    "epochs_since_best": stats.get("epochs_since_best", ""),
                    "best_val_rmse": stats.get("best_val_rmse", ""),
                    "final_val_rmse": stats.get("final_val_rmse", ""),
                    "relative_change_50": stats.get("relative_change", ""),
                    "predicted_decrease_ols_50": stats.get("predicted_relative_decrease_ols", ""),
                    "predicted_decrease_theil_sen_50": stats.get("predicted_relative_decrease_theil_sen", ""),
                    "exact_latest_resume_possible": row["exact_latest_resume_possible"],
                    "extend": row["continuation_decision"]["extend"],
                    "extend_reason": row["continuation_decision"]["reason"],
                }
            )


def write_convergence_audit_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Corsi Motion Convergence Audit",
        "",
        "Date: 2026-06-22",
        "",
        "## Loss Mask Audit",
        "",
        f"Conclusion: `{result['loss_mask_audit']['conclusion']}`.",
        "",
        "The training objective uses `loss_mask == transition_mask`. In the canonical dataset,",
        "`transition_mask` is exactly `within_segment_transition | segment_boundary_transition`,",
        "excluding only the final timestep of each episode. Continuation training must preserve this mask.",
        "",
        "Counts across the canonical manifest:",
        "",
        f"- transitions: {result['loss_mask_audit']['transition_count']}",
        f"- within-segment transitions: {result['loss_mask_audit']['within_segment_transition_count']}",
        f"- segment-boundary transitions: {result['loss_mask_audit']['segment_boundary_transition_count']}",
        f"- samples checked: {result['loss_mask_audit']['samples_checked']}",
        "",
        "## Resume Audit",
        "",
        "Original checkpoints contain model state, optimizer state, epoch, best metric, config,",
        "normalization, and joint names. They do not contain AMP scaler state or RNG states.",
        "Therefore exact latest-state resume is not available for the original runs. Any continuation",
        "from the original checkpoints must be labeled warm-start from `best.pt`, not exact resume.",
        "",
        "## Per-Run Convergence",
        "",
        "| Model | Seed | Epochs | Class | Best epoch | Since best | Best val RMSE | Final val RMSE | Extend | Reason |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result["per_run"]:
        stats = row["stats_50"]
        summary = row.get("summary", {})
        lines.append(
            "| {model} | {seed} | {epochs} | {cls} | {best_epoch} | {since} | {best:.6f} | {final:.6f} | {extend} | {reason} |".format(
                model=row["model_type"],
                seed=row["seed"],
                epochs=summary.get("epochs_completed", 0),
                cls=row["final_classification"],
                best_epoch=stats.get("best_epoch", 0),
                since=stats.get("epochs_since_best", 0),
                best=float(stats.get("best_val_rmse", float("nan"))),
                final=float(stats.get("final_val_rmse", float("nan"))),
                extend=row["continuation_decision"]["extend"],
                reason=row["continuation_decision"]["reason"],
            )
        )
    lines.extend(
        [
            "",
            "## Missing History Fields",
            "",
            "Learning rate is recoverable from optimizer param groups in the checkpoint, but it was not",
            "recorded per epoch in `curves.json`. Gradient norm, AMP scaler state, scheduler state,",
            "and RNG state were not recorded in the original artifacts. These values are not fabricated.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _copy_warm_start_inputs(original_ref: RunRef, extension_dir: Path) -> dict[str, Any]:
    extension_dir.mkdir(parents=True, exist_ok=True)
    best_src = original_ref.run_dir / "best.pt"
    latest_dst = extension_dir / "latest.pt"
    best_dst = extension_dir / "best.pt"
    if not latest_dst.exists():
        shutil.copy2(best_src, latest_dst)
    if not best_dst.exists():
        shutil.copy2(best_src, best_dst)
    best_payload = torch.load(best_src, map_location="cpu", weights_only=False)
    best_epoch = int(best_payload["epoch"])
    original_curves = load_curves(original_ref.run_dir)
    warm_curves = [row for row in original_curves if int(row["epoch"]) <= best_epoch]
    curves_path = extension_dir / "curves.json"
    if not curves_path.exists():
        write_json(curves_path, warm_curves)
    metadata = {
        "continuation_type": "warm_start_from_best",
        "exact_resume": False,
        "reason_exact_resume_unavailable": "original checkpoints lack AMP scaler and RNG states",
        "original_run": original_ref.run_name,
        "original_run_dir": str(original_ref.run_dir),
        "source_checkpoint": str(best_src),
        "source_checkpoint_epoch": best_epoch,
        "extension_run_dir": str(extension_dir),
    }
    write_json(extension_dir / "continuation_metadata.json", metadata)
    return metadata


def run_continuations(
    config: dict[str, Any],
    convergence: dict[str, Any],
    *,
    device: str,
    posthoc_root: Path = POSTHOC_ROOT,
) -> dict[str, Any]:
    lineage_path = posthoc_root / "convergence" / "continuation_lineage.json"
    lineage = read_json(lineage_path) if lineage_path.exists() else {"schema_version": "posthoc_convergence_lineage_v1", "runs": []}
    existing_by_run = {row.get("extension_run_name"): row for row in lineage.get("runs", [])}
    output_root = Path(str(config["output_root"]))
    continuation_config = dict(config)
    continuation_config["device"] = device
    continuation_config["early_stopping_patience"] = 75
    max_total_epoch = 600
    started = time.time()
    for row in convergence["per_run"]:
        if not row["continuation_decision"]["extend"]:
            continue
        ref = RunRef(
            model_type=str(row["model_type"]),
            seed=int(row["seed"]),
            run_name=str(row["run_name"]),
            run_dir=Path(str(row["run_dir"])),
        )
        ext_name = extension_run_name(ref)
        ext_dir = output_root / ext_name
        if ext_name in existing_by_run and (ext_dir / "summary.json").exists():
            continue
        metadata = _copy_warm_start_inputs(ref, ext_dir)
        summary_path = ext_dir / "summary.json"
        summary = read_json(summary_path) if summary_path.exists() else {}
        if int(summary.get("epochs_completed", 0)) < max_total_epoch:
            summary = train_one_run(
                continuation_config,
                model_type=ref.model_type,
                seed=ref.seed,
                run_dir=ext_dir,
                max_epochs_override=max_total_epoch,
                resume=True,
            )
        extension_stats_50 = convergence_stats(load_curves(ext_dir), 50)
        second_extension_summary = None
        second_extension_stats_50 = None
        if (
            int(summary.get("epochs_completed", 0)) >= max_total_epoch
            and float(extension_stats_50.get("relative_change", 0.0)) >= 0.01
        ):
            if int(summary.get("epochs_completed", 0)) >= 900:
                second_extension_summary = summary
            else:
                second_extension_summary = train_one_run(
                    continuation_config,
                    model_type=ref.model_type,
                    seed=ref.seed,
                    run_dir=ext_dir,
                    max_epochs_override=900,
                    resume=True,
                )
            second_extension_stats_50 = convergence_stats(load_curves(ext_dir), 50)
        lineage_row = {
            **metadata,
            "extension_run_name": ext_name,
            "extension_summary": summary,
            "extension_stats_50": extension_stats_50,
            "second_extension_summary": second_extension_summary,
            "second_extension_stats_50": second_extension_stats_50,
            "patience": 75,
            "max_total_epoch": max_total_epoch,
            "hard_max_epoch": 900,
            "objective_preserved": True,
            "loss_mask": "transition_mask_includes_within_and_boundary_transitions",
        }
        existing_by_run[ext_name] = lineage_row
        lineage["runs"] = list(existing_by_run.values())
        write_json(lineage_path, lineage)
    lineage["runtime_seconds"] = float(time.time() - started)
    write_json(lineage_path, lineage)
    return lineage


def select_final_checkpoints(config: dict[str, Any], posthoc_root: Path = POSTHOC_ROOT) -> dict[str, Any]:
    """Select final learned checkpoints using validation RMSE only."""

    output_root = Path(str(config["output_root"]))
    lineage_path = posthoc_root / "convergence" / "continuation_lineage.json"
    lineage = read_json(lineage_path) if lineage_path.exists() else {"runs": []}
    lineage_by_original = {row.get("original_run"): row for row in lineage.get("runs", [])}
    selections = []
    for ref in original_run_refs(config):
        original_summary = read_json(ref.run_dir / "summary.json")
        original_best = float(original_summary["best_val_normalized_rmse"])
        candidates = [
            {
                "source": "original",
                "run_name": ref.run_name,
                "checkpoint": str(ref.run_dir / "best.pt"),
                "summary": str(ref.run_dir / "summary.json"),
                "best_val_rmse": original_best,
            }
        ]
        ext = lineage_by_original.get(ref.run_name)
        if ext:
            ext_dir = Path(str(ext["extension_run_dir"]))
            summary_path = ext_dir / "summary.json"
            if summary_path.exists():
                ext_summary = read_json(summary_path)
                candidates.append(
                    {
                        "source": "extension",
                        "run_name": ext["extension_run_name"],
                        "checkpoint": str(ext_dir / "best.pt"),
                        "summary": str(summary_path),
                        "best_val_rmse": float(ext_summary["best_val_normalized_rmse"]),
                    }
                )
        best_candidate = min(candidates, key=lambda item: float(item["best_val_rmse"]))
        selections.append(
            {
                "model_type": ref.model_type,
                "seed": int(ref.seed),
                "original_run_name": ref.run_name,
                "original_checkpoint": str(ref.run_dir / "best.pt"),
                "original_best_val_rmse": original_best,
                "final_run_name": best_candidate["run_name"],
                "final_checkpoint": best_candidate["checkpoint"],
                "final_best_val_rmse": float(best_candidate["best_val_rmse"]),
                "final_source": best_candidate["source"],
                "all_candidates": candidates,
            }
        )
    result = {"schema_version": "posthoc_final_checkpoint_selection_v1", "selections": selections}
    write_json(posthoc_root / "convergence" / "final_checkpoint_selection.json", result)
    return result


def export_final_visual_states(
    config: dict[str, Any],
    selection: dict[str, Any],
    *,
    device_name: str,
    posthoc_root: Path = POSTHOC_ROOT,
) -> dict[str, Any]:
    from corsi.experiments.corsi_motion_baseline.extract_states import export_states

    states_dir = posthoc_root / "states"
    exports = []
    for row in selection.get("selections", []):
        if row.get("model_type") != "visual_joint" or row.get("final_source") != "extension":
            continue
        for split in ["train", "val", "test"]:
            output_path = states_dir / f"{row['final_run_name']}_{split}.h5"
            metadata_path = output_path.with_suffix(".json")
            if output_path.exists() and metadata_path.exists():
                metadata = read_json(metadata_path)
            else:
                metadata = export_states(
                    config=config,
                    checkpoint=row["final_checkpoint"],
                    split=split,
                    output_path=output_path,
                    batch_size=16,
                    device_name=device_name,
                )
            metadata["model_type"] = row["model_type"]
            metadata["seed"] = int(row["seed"])
            metadata["final_run_name"] = row["final_run_name"]
            metadata["final_source"] = row["final_source"]
            exports.append(metadata)
    result = {
        "schema_version": "posthoc_final_visual_state_exports_v1",
        "exports": exports,
    }
    write_json(states_dir / "final_visual_state_exports.json", result)
    return result


def _load_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return read_json(Path(str(config["canonical_root"])) / "manifest.json")


def _load_test_items(manifest_path: Path, *, load_images: bool = True) -> list[dict[str, Any]]:
    dataset = CorsiMotionCanonicalDataset(manifest_path, split="test", load_images=load_images, cache=True)
    return [dataset[index] for index in range(len(dataset))]


def _sample_by_seq_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(sample["seq_id"]): sample for sample in manifest["samples"]}


def _shuffled_images_by_seq_id(items: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    by_length: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        by_length.setdefault(int(item["length"]), []).append(item)
    shuffled = {}
    for _, rows in by_length.items():
        rows = sorted(rows, key=lambda item: str(item["seq_id"]))
        rotated = rows[1:] + rows[:1]
        for item, replacement in zip(rows, rotated):
            shuffled[str(item["seq_id"])] = np.asarray(replacement["images"], dtype=np.float32)
    return shuffled


def _joint_denormalize_np(joints: np.ndarray, manifest: dict[str, Any]) -> np.ndarray:
    stats = manifest["normalization"]
    mean = np.asarray(stats["joint_mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(stats["joint_std"], dtype=np.float32).reshape(1, -1)
    return joints * std + mean


def _load_model_for_checkpoint(model_type: str, checkpoint: str | Path, manifest: dict[str, Any], device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(model_type, joint_dim=int(manifest["joint_dim"]), hidden_dim=128).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def _teacher_forced_prediction(
    *,
    item: dict[str, Any],
    model: torch.nn.Module | None,
    model_type: str,
    mode: str,
    manifest: dict[str, Any],
    device: torch.device,
    shuffled_images: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    joints_np = np.asarray(item["joints"], dtype=np.float32)
    joints = torch.as_tensor(joints_np, dtype=torch.float32, device=device).unsqueeze(0)
    valid = torch.ones((1, joints.shape[1]), dtype=torch.bool, device=device)
    if model_type == "persistence":
        pred = joints.clone()
    else:
        images = None
        if model_type == "visual_joint":
            if mode == "shuffled_vision":
                image_np = shuffled_images[str(item["seq_id"])] if shuffled_images is not None else item["images"]
            else:
                image_np = item["images"]
            images = torch.as_tensor(np.asarray(image_np, dtype=np.float32), device=device).unsqueeze(0)
            if mode == "zero_vision":
                images = torch.zeros_like(images)
        outputs = model(images=images if model_type == "visual_joint" else None, joints=joints, valid_mask=valid)
        pred = outputs["pred_joints_next"]  # type: ignore[index]
    pred_norm = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
    joints_phys = np.asarray(item["physical_joints"], dtype=np.float32)
    pred_phys = _joint_denormalize_np(pred_norm, manifest).astype(np.float32)
    return {
        "seq_id": str(item["seq_id"]),
        "length": int(item["length"]),
        "block_order": list(item["block_order"]),
        "joints_norm": joints_np,
        "joints_phys": joints_phys,
        "pred_norm": pred_norm,
        "pred_phys": pred_phys,
        "loss_mask": np.asarray(item["transition_mask"], dtype=bool),
        "within_segment_transition": np.asarray(item["within_segment_transition"], dtype=bool),
        "segment_boundary_transition": np.asarray(item["segment_boundary_transition"], dtype=bool),
        "rank": np.asarray(item["rank"], dtype=np.int64),
        "block_id": np.asarray(item["block_id"], dtype=np.int64),
        "source_frame_index": np.asarray(item["source_frame_index"], dtype=np.int64),
        "segment_progress": np.asarray(item["segment_progress"], dtype=np.float32),
    }


@torch.no_grad()
def _autoregressive_prediction(
    *,
    item: dict[str, Any],
    model: torch.nn.Module | None,
    model_type: str,
    mode: str,
    manifest: dict[str, Any],
    device: torch.device,
    shuffled_images: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    joints_np = np.asarray(item["joints"], dtype=np.float32)
    steps, joint_dim = joints_np.shape
    pred_norm = np.zeros_like(joints_np, dtype=np.float32)
    q_input = torch.as_tensor(joints_np[0], dtype=torch.float32, device=device).reshape(1, joint_dim)
    if model_type == "persistence":
        for step in range(steps):
            pred_norm[step] = q_input.detach().cpu().numpy()[0]
        pred_phys = _joint_denormalize_np(pred_norm, manifest).astype(np.float32)
        base = _teacher_forced_prediction(
            item=item,
            model=None,
            model_type="persistence",
            mode=mode,
            manifest=manifest,
            device=device,
        )
        base["pred_norm"] = pred_norm
        base["pred_phys"] = pred_phys
        return base

    h = torch.zeros(1, model.config.hidden_dim, dtype=torch.float32, device=device)
    c = torch.zeros_like(h)
    if model_type == "visual_joint":
        if mode == "shuffled_vision":
            image_np = shuffled_images[str(item["seq_id"])] if shuffled_images is not None else item["images"]
        else:
            image_np = item["images"]
        image_tensor = torch.as_tensor(np.asarray(image_np, dtype=np.float32), device=device).unsqueeze(0)
        if mode == "zero_vision":
            image_tensor = torch.zeros_like(image_tensor)
    else:
        image_tensor = None

    for step in range(steps):
        joint_feature = model.joint_encoder(q_input)
        if model_type == "visual_joint":
            visual_feature = model._encode_visual(image_tensor[:, step : step + 1]).reshape(1, -1)
            fused_input = torch.cat([visual_feature, joint_feature], dim=-1)
        else:
            fused_input = joint_feature
        fused = model.fusion(fused_input)
        h, c, _ = model.recurrent(fused, (h, c))
        pred = model.output(h)
        pred_norm[step] = pred.detach().cpu().numpy()[0]
        q_input = pred.detach()
    pred_phys = _joint_denormalize_np(pred_norm, manifest).astype(np.float32)
    base = _teacher_forced_prediction(
        item=item,
        model=model,
        model_type=model_type,
        mode=mode,
        manifest=manifest,
        device=device,
        shuffled_images=shuffled_images,
    )
    base["pred_norm"] = pred_norm
    base["pred_phys"] = pred_phys
    return base


def _condition_specs(selection: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for row in selection["selections"]:
        if row["model_type"] == "visual_joint":
            for mode in VISUAL_MODES:
                specs.append(
                    {
                        "model_type": "visual_joint",
                        "mode": mode,
                        "seed": int(row["seed"]),
                        "checkpoint": row["final_checkpoint"],
                        "final_source": row["final_source"],
                        "condition": f"visual_joint_{mode}_seed{row['seed']}",
                    }
                )
        elif row["model_type"] == "joint_only":
            specs.append(
                {
                    "model_type": "joint_only",
                    "mode": "normal",
                    "seed": int(row["seed"]),
                    "checkpoint": row["final_checkpoint"],
                    "final_source": row["final_source"],
                    "condition": f"joint_only_normal_seed{row['seed']}",
                }
            )
    specs.append(
        {
            "model_type": "persistence",
            "mode": "normal",
            "seed": -1,
            "checkpoint": "",
            "final_source": "stateless",
            "condition": "persistence_normal",
        }
    )
    return specs


def _bootstrap_ci(success: np.ndarray, total: np.ndarray, *, seed: int = 20260622, reps: int = 1000) -> list[float]:
    if len(success) == 0 or int(total.sum()) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    values = []
    indexes = np.arange(len(success))
    for _ in range(int(reps)):
        sample = rng.choice(indexes, size=len(indexes), replace=True)
        denom = int(total[sample].sum())
        values.append(0.0 if denom == 0 else float(success[sample].sum() / denom))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _step_records_for_episode(prediction: dict[str, Any], threshold: float) -> dict[str, Any]:
    mask = prediction["loss_mask"].astype(bool)
    step_indices = np.nonzero(mask)[0]
    target = prediction["joints_phys"][step_indices + 1]
    pred = prediction["pred_phys"][step_indices]
    err_deg = np.abs(pred - target) * (180.0 / math.pi)
    per_joint = err_deg <= float(threshold)
    all_joint = np.all(per_joint, axis=1)
    return {
        "step_indices": step_indices,
        "err_deg": err_deg,
        "per_joint_correct": per_joint,
        "all_joint_correct": all_joint,
        "within": prediction["within_segment_transition"][step_indices].astype(bool),
        "boundary": prediction["segment_boundary_transition"][step_indices].astype(bool),
        "rank": prediction["rank"][step_indices].astype(np.int64),
    }


def _step_accuracy_summary_for_predictions(
    predictions: list[dict[str, Any]],
    *,
    thresholds: list[float],
    condition_meta: dict[str, Any],
) -> dict[str, Any]:
    threshold_rows = {}
    episode_bootstrap_inputs: dict[float, tuple[list[int], list[int]]] = {
        threshold: ([], []) for threshold in thresholds
    }
    for threshold in thresholds:
        all_correct = 0
        all_total = 0
        per_joint_correct = np.zeros((7,), dtype=np.int64)
        per_joint_total = 0
        within_correct = 0
        within_total = 0
        boundary_correct = 0
        boundary_total = 0
        by_length: dict[int, dict[str, int]] = {}
        normalized_sqerr = []
        joint_abs_deg = []
        for prediction in predictions:
            records = _step_records_for_episode(prediction, threshold)
            step_indices = records["step_indices"]
            target_norm = prediction["joints_norm"][step_indices + 1]
            pred_norm = prediction["pred_norm"][step_indices]
            normalized_sqerr.extend(np.mean((pred_norm - target_norm) ** 2, axis=1).tolist())
            target_phys = prediction["joints_phys"][step_indices + 1]
            pred_phys = prediction["pred_phys"][step_indices]
            joint_abs_deg.append(np.abs(pred_phys - target_phys) * (180.0 / math.pi))
            all_joint = records["all_joint_correct"]
            per_joint = records["per_joint_correct"]
            all_correct += int(all_joint.sum())
            all_total += int(all_joint.size)
            per_joint_correct += per_joint.sum(axis=0).astype(np.int64)
            per_joint_total += int(per_joint.shape[0])
            within = records["within"]
            boundary = records["boundary"]
            within_correct += int(all_joint[within].sum())
            within_total += int(within.sum())
            boundary_correct += int(all_joint[boundary].sum())
            boundary_total += int(boundary.sum())
            length = int(prediction["length"])
            by_length.setdefault(length, {"correct": 0, "total": 0})
            by_length[length]["correct"] += int(all_joint.sum())
            by_length[length]["total"] += int(all_joint.size)
            successes, totals = episode_bootstrap_inputs[threshold]
            successes.append(int(all_joint.sum()))
            totals.append(int(all_joint.size))
        abs_deg = np.concatenate(joint_abs_deg, axis=0) if joint_abs_deg else np.zeros((0, 7), dtype=np.float32)
        threshold_rows[str(threshold)] = {
            "all_joint_step_accuracy": 0.0 if all_total == 0 else float(all_correct / all_total),
            "all_joint_step_correct": int(all_correct),
            "all_joint_step_total": int(all_total),
            "per_joint_tolerance_accuracy": [
                0.0 if per_joint_total == 0 else float(value / per_joint_total)
                for value in per_joint_correct.tolist()
            ],
            "within_segment_step_accuracy": 0.0 if within_total == 0 else float(within_correct / within_total),
            "boundary_transition_step_accuracy": 0.0 if boundary_total == 0 else float(boundary_correct / boundary_total),
            "accuracy_by_length": {
                str(length): 0.0 if row["total"] == 0 else float(row["correct"] / row["total"])
                for length, row in sorted(by_length.items())
            },
            "episode_bootstrap_95ci": _bootstrap_ci(
                np.asarray(episode_bootstrap_inputs[threshold][0], dtype=np.int64),
                np.asarray(episode_bootstrap_inputs[threshold][1], dtype=np.int64),
                seed=20260622 + int(condition_meta.get("seed", 0)) + int(round(threshold * 100)),
            ),
            "normalized_rmse": float(math.sqrt(np.mean(normalized_sqerr))) if normalized_sqerr else 0.0,
            "joint_mae_deg_mean": float(abs_deg.mean()) if abs_deg.size else 0.0,
            "joint_mae_deg": abs_deg.mean(axis=0).tolist() if abs_deg.size else [0.0] * 7,
        }
    return {
        **condition_meta,
        "thresholds": threshold_rows,
    }


def aggregate_step_accuracy(per_seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    persistence_rows = []
    for row in per_seed_rows:
        if row["model_type"] == "persistence":
            persistence_rows.append(row)
            continue
        grouped.setdefault((row["model_type"], row["mode"]), []).append(row)
    aggregate: dict[str, Any] = {"groups": {}, "persistence": persistence_rows[0] if persistence_rows else None}
    for (model_type, mode), rows in sorted(grouped.items()):
        group_key = f"{model_type}_{mode}"
        aggregate["groups"][group_key] = {"model_type": model_type, "mode": mode, "seed_count": len(rows), "thresholds": {}}
        for threshold in STEP_THRESHOLDS:
            key = str(threshold)
            values = [row["thresholds"][key]["all_joint_step_accuracy"] for row in rows]
            mae = [row["thresholds"][key]["joint_mae_deg_mean"] for row in rows]
            rmse = [row["thresholds"][key]["normalized_rmse"] for row in rows]
            successes = np.asarray([row["thresholds"][key]["all_joint_step_correct"] for row in rows], dtype=np.int64)
            totals = np.asarray([row["thresholds"][key]["all_joint_step_total"] for row in rows], dtype=np.int64)
            aggregate["groups"][group_key]["thresholds"][key] = {
                "per_seed_values": values,
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "episode_bootstrap_95ci": _bootstrap_ci(successes, totals, seed=20260622 + int(round(threshold * 100))),
                "normalized_rmse_mean": float(np.mean(rmse)),
                "normalized_rmse_sd": float(np.std(rmse, ddof=1)) if len(rmse) > 1 else 0.0,
                "joint_mae_deg_mean": float(np.mean(mae)),
                "joint_mae_deg_sd": float(np.std(mae, ddof=1)) if len(mae) > 1 else 0.0,
            }
    return aggregate


def write_tolerance_curve_csv(path: Path, per_seed_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_type",
        "mode",
        "seed",
        "threshold_deg",
        "all_joint_step_accuracy",
        "within_segment_step_accuracy",
        "boundary_transition_step_accuracy",
    ] + [f"joint{index + 1}_accuracy" for index in range(7)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_seed_rows:
            for threshold in TOLERANCE_CURVE_THRESHOLDS:
                metrics = row["tolerance_curve"][str(threshold)]
                writer.writerow(
                    {
                        "model_type": row["model_type"],
                        "mode": row["mode"],
                        "seed": row["seed"],
                        "threshold_deg": threshold,
                        "all_joint_step_accuracy": metrics["all_joint_step_accuracy"],
                        "within_segment_step_accuracy": metrics["within_segment_step_accuracy"],
                        "boundary_transition_step_accuracy": metrics["boundary_transition_step_accuracy"],
                        **{
                            f"joint{index + 1}_accuracy": metrics["per_joint_tolerance_accuracy"][index]
                            for index in range(7)
                        },
                    }
                )


class FKEvaluator:
    def __init__(self, manifest: dict[str, Any], split: str = "test") -> None:
        import robosuite as suite
        import corsi.envs.robosuite_corsi  # noqa: F401
        from corsi.envs.robosuite_corsi import (
            standard_corsi_robosuite_board_size,
            standard_corsi_robosuite_layout,
        )
        from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

        layout = standard_corsi_robosuite_layout()
        positions = [layout[index] for index in sorted(layout)]
        self.env = suite.make(
            env_name="CorsiSceneDemo",
            robots="PandaDexRH",
            controller_configs=load_composite_controller_config(controller="BASIC"),
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            ignore_done=True,
            block_xy_positions=positions,
            corsi_board_size_xy=standard_corsi_robosuite_board_size(),
            seed=0,
        )
        self.manifest = manifest
        self.split = split
        self.robot = self.env.robots[0]
        self.arm_qpos_indexes = np.asarray(list(self.robot._ref_arm_joint_pos_indexes), dtype=np.int64)
        all_qpos = np.arange(len(self.env.sim.data.qpos), dtype=np.int64)
        self.hand_qpos_indexes = np.asarray([index for index in all_qpos if index not in set(self.arm_qpos_indexes)], dtype=np.int64)
        self.site_name = "gripper0_right_index_tip_site"
        self.site_id = int(self.env.sim.model.site_name2id(self.site_name))
        self.fixed_hand_qpos = self._estimate_fixed_hand_qpos()
        self.block_geometry = self._read_block_geometry()

    def close(self) -> None:
        self.env.close()

    def _endpoint_samples(self) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
        split_ids = set(self.manifest["split"][self.split])
        for sample in self.manifest["samples"]:
            if sample["seq_id"] not in split_ids:
                continue
            segments = read_json(Path(str(sample["raw_segments_path"])))
            for segment in segments:
                yield sample, segment

    def _estimate_fixed_hand_qpos(self) -> np.ndarray:
        rows = []
        for sample, segment in self._endpoint_samples():
            with np.load(sample["raw_arrays_path"]) as arrays:
                rows.append(np.asarray(arrays["qpos"][int(segment["end_frame"])][self.hand_qpos_indexes], dtype=np.float64))
        if not rows:
            return np.asarray(self.env.sim.data.qpos[self.hand_qpos_indexes], dtype=np.float64)
        return np.median(np.stack(rows, axis=0), axis=0)

    def _read_block_geometry(self) -> dict[str, Any]:
        geometry = {}
        for block_id in range(9):
            geom_name = f"corsi_block_{block_id}_geom"
            body_name = f"corsi_block_{block_id}"
            geom_id = int(self.env.sim.model.geom_name2id(geom_name))
            body_id = int(self.env.sim.model.body_name2id(body_name))
            half_size = np.asarray(self.env.sim.model.geom_size[geom_id], dtype=np.float64)
            center = np.asarray(self.env.sim.data.body_xpos[body_id], dtype=np.float64)
            geometry[str(block_id)] = {
                "geom_name": geom_name,
                "body_name": body_name,
                "center_world": center.tolist(),
                "half_size": half_size.tolist(),
                "footprint_half_size_xy": half_size[:2].tolist(),
            }
        return geometry

    def mapping_metadata(self) -> dict[str, Any]:
        joint_names = []
        for joint_id in getattr(self.robot, "_ref_arm_joint_indexes", []):
            joint_names.append(str(self.env.sim.model.joint_id2name(int(joint_id))))
        qpos_ranges = []
        for joint_name in joint_names:
            joint_id = int(self.env.sim.model.joint_name2id(joint_name))
            qpos_ranges.append(self.env.sim.model.jnt_range[joint_id].tolist())
        return {
            "robot": self.robot.name,
            "site_name": self.site_name,
            "site_id": self.site_id,
            "arm_joint_names": joint_names,
            "arm_qpos_indexes": self.arm_qpos_indexes.tolist(),
            "hand_qpos_indexes": self.hand_qpos_indexes.tolist(),
            "fixed_hand_qpos": self.fixed_hand_qpos.tolist(),
            "joint_limits": qpos_ranges,
            "fk_qpos_policy": "set seven Panda arm qpos, preserve median fixed hand qpos, call sim.forward, read gripper0_right_index_tip_site",
        }

    def joint_limits(self) -> np.ndarray:
        ranges = []
        for joint_id in getattr(self.robot, "_ref_arm_joint_indexes", []):
            ranges.append(self.env.sim.model.jnt_range[int(joint_id)])
        return np.asarray(ranges, dtype=np.float64)

    def fk(self, q_arm: np.ndarray) -> np.ndarray:
        self.env.sim.data.qpos[self.arm_qpos_indexes] = np.asarray(q_arm, dtype=np.float64)
        self.env.sim.data.qpos[self.hand_qpos_indexes] = self.fixed_hand_qpos
        self.env.sim.data.qvel[:] = 0.0
        self.env.sim.forward()
        return np.asarray(self.env.sim.data.site_xpos[self.site_id], dtype=np.float64).copy()

    def validate(self) -> dict[str, Any]:
        errors = []
        for sample, segment in self._endpoint_samples():
            with np.load(sample["raw_arrays_path"]) as arrays:
                end = int(segment["end_frame"])
                q_arm = np.asarray(arrays["joint"][end], dtype=np.float64)
                recorded = np.asarray(arrays["ee_pose"][end, :3], dtype=np.float64)
            predicted = self.fk(q_arm)
            errors.append(float(np.linalg.norm(predicted - recorded)))
        median = float(np.median(errors)) if errors else 0.0
        maximum = float(np.max(errors)) if errors else 0.0
        return {
            "site_name": self.site_name,
            "sample_count": int(len(errors)),
            "median_position_error_m": median,
            "max_position_error_m": maximum,
            "median_position_error_mm": median * 1000.0,
            "max_position_error_mm": maximum * 1000.0,
            "passed": bool(median <= 0.002 and maximum <= 0.005),
        }


def nearest_block_id(xy: np.ndarray, block_geometry: dict[str, Any]) -> int:
    distances = {}
    for key, geom in block_geometry.items():
        center = np.asarray(geom["center_world"], dtype=np.float64)[:2]
        distances[int(key)] = float(np.linalg.norm(np.asarray(xy, dtype=np.float64) - center))
    return int(min(distances, key=distances.get))


def physical_hit(
    pos_world: np.ndarray,
    target_block_id: int,
    gt_endpoint_z: float,
    block_geometry: dict[str, Any],
    *,
    margin_m: float,
) -> bool:
    geom = block_geometry[str(int(target_block_id))]
    center = np.asarray(geom["center_world"], dtype=np.float64)
    half = np.asarray(geom["footprint_half_size_xy"], dtype=np.float64) + float(margin_m)
    pos = np.asarray(pos_world, dtype=np.float64)
    inside_xy = bool(np.all(np.abs(pos[:2] - center[:2]) <= half))
    z_ok = bool(abs(float(pos[2]) - float(gt_endpoint_z)) <= 0.020)
    return bool(inside_xy and z_ok)


def endpoint_transition_indices(prediction: dict[str, Any], k: int) -> list[tuple[int, int]]:
    """Returns `(rank, prediction_step)` for final within-segment transitions."""

    indices = []
    for rank in range(int(prediction["length"])):
        endpoint = (rank + 1) * int(k) - 1
        prediction_step = endpoint - 1
        if prediction_step < 0:
            continue
        if not bool(prediction["within_segment_transition"][prediction_step]):
            raise ValueError(
                f"{prediction['seq_id']} rank {rank}: endpoint prediction step {prediction_step} is not within-segment"
            )
        indices.append((rank, prediction_step))
    return indices


def token_rows_from_predictions(
    predictions: list[dict[str, Any]],
    *,
    condition_meta: dict[str, Any],
    manifest_samples: dict[str, dict[str, Any]],
    fk: FKEvaluator,
    k: int,
    autoregressive: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    joint_limits = fk.joint_limits()
    for prediction in predictions:
        sample = manifest_samples[prediction["seq_id"]]
        with np.load(sample["raw_arrays_path"]) as raw:
            raw_ee_pose = np.asarray(raw["ee_pose"], dtype=np.float64)
        for rank, pred_step in endpoint_transition_indices(prediction, k):
            endpoint = pred_step + 1
            target_block = int(prediction["block_id"][endpoint])
            pred_q = np.asarray(prediction["pred_phys"][pred_step], dtype=np.float64)
            target_q = np.asarray(prediction["joints_phys"][endpoint], dtype=np.float64)
            source_frame = int(prediction["source_frame_index"][endpoint])
            gt_ee = np.asarray(raw_ee_pose[source_frame, :3], dtype=np.float64)
            pred_ee = fk.fk(pred_q)
            nearest = nearest_block_id(pred_ee[:2], fk.block_geometry)
            limit_violation = bool(np.any((pred_q < joint_limits[:, 0]) | (pred_q > joint_limits[:, 1])))
            hit_0 = physical_hit(pred_ee, target_block, gt_ee[2], fk.block_geometry, margin_m=0.0)
            hit_5 = physical_hit(pred_ee, target_block, gt_ee[2], fk.block_geometry, margin_m=0.005)
            hit_10 = physical_hit(pred_ee, target_block, gt_ee[2], fk.block_geometry, margin_m=0.010)
            if autoregressive and limit_violation:
                hit_0 = hit_5 = hit_10 = False
            err = pred_ee - gt_ee
            rows.append(
                {
                    **condition_meta,
                    "seq_id": prediction["seq_id"],
                    "length": int(prediction["length"]),
                    "rank": int(rank),
                    "prediction_step": int(pred_step),
                    "endpoint_index": int(endpoint),
                    "target_block_id": target_block,
                    "predicted_block_id": int(nearest),
                    "nearest_block_correct": bool(nearest == target_block),
                    "physical_hit_margin_0mm": bool(hit_0),
                    "physical_hit_margin_5mm": bool(hit_5),
                    "physical_hit_margin_10mm": bool(hit_10),
                    "joint_limit_violation": bool(limit_violation),
                    "endpoint_ee_error_cm": float(np.linalg.norm(err) * 100.0),
                    "endpoint_xy_error_cm": float(np.linalg.norm(err[:2]) * 100.0),
                    "endpoint_z_error_cm": float(abs(err[2]) * 100.0),
                    "pred_ee_x": float(pred_ee[0]),
                    "pred_ee_y": float(pred_ee[1]),
                    "pred_ee_z": float(pred_ee[2]),
                    "gt_ee_x": float(gt_ee[0]),
                    "gt_ee_y": float(gt_ee[1]),
                    "gt_ee_z": float(gt_ee[2]),
                    "target_joint_rmse_deg": float(
                        math.sqrt(float(np.mean(((pred_q - target_q) * 180.0 / math.pi) ** 2)))
                    ),
                    "mode_type": "autoregressive" if autoregressive else "teacher_forced",
                }
            )
    return rows


def _metric_mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def token_accuracy_summary(rows: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    summaries = []
    for condition, condition_rows in sorted(by_condition.items()):
        by_length = {}
        by_rank_within_length = {}
        for row in condition_rows:
            length = str(row["length"])
            rank_key = f"{row['length']}:{row['rank']}"
            by_length.setdefault(length, []).append(row)
            by_rank_within_length.setdefault(rank_key, []).append(row)
        summaries.append(
            {
                "condition": condition,
                "model_type": condition_rows[0]["model_type"],
                "mode": condition_rows[0]["mode"],
                "seed": int(condition_rows[0]["seed"]),
                f"{prefix}_token_nearest_block_acc": _metric_mean(condition_rows, "nearest_block_correct"),
                f"{prefix}_token_physical_hit_acc": _metric_mean(condition_rows, "physical_hit_margin_5mm"),
                "physical_hit_margin_sensitivity": {
                    "0mm": _metric_mean(condition_rows, "physical_hit_margin_0mm"),
                    "5mm": _metric_mean(condition_rows, "physical_hit_margin_5mm"),
                    "10mm": _metric_mean(condition_rows, "physical_hit_margin_10mm"),
                },
                "endpoint_ee_error_cm_mean": _metric_mean(condition_rows, "endpoint_ee_error_cm"),
                "endpoint_xy_error_cm_mean": _metric_mean(condition_rows, "endpoint_xy_error_cm"),
                "endpoint_z_error_cm_mean": _metric_mean(condition_rows, "endpoint_z_error_cm"),
                "by_length": {
                    length: {
                        f"{prefix}_token_nearest_block_acc": _metric_mean(group, "nearest_block_correct"),
                        f"{prefix}_token_physical_hit_acc": _metric_mean(group, "physical_hit_margin_5mm"),
                    }
                    for length, group in sorted(by_length.items(), key=lambda item: int(item[0]))
                },
                "by_rank_within_length": {
                    key: {
                        f"{prefix}_token_nearest_block_acc": _metric_mean(group, "nearest_block_correct"),
                        f"{prefix}_token_physical_hit_acc": _metric_mean(group, "physical_hit_margin_5mm"),
                    }
                    for key, group in sorted(
                        by_rank_within_length.items(),
                        key=lambda item: tuple(int(part) for part in item[0].split(":")),
                    )
                },
            }
        )
    return {"per_condition": summaries}


def full_trial_rows_from_token_rows(rows: list[dict[str, Any]], *, prefix: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["condition"]), str(row["seq_id"])), []).append(row)
    full_rows = []
    for (condition, seq_id), group in sorted(grouped.items()):
        nearest_ok = bool(all(row["nearest_block_correct"] for row in group))
        hit_ok = bool(all(row["physical_hit_margin_5mm"] for row in group))
        first = group[0]
        full_rows.append(
            {
                "condition": condition,
                "model_type": first["model_type"],
                "mode": first["mode"],
                "seed": int(first["seed"]),
                "seq_id": seq_id,
                "length": int(first["length"]),
                f"{prefix}_full_nearest_block_correct": nearest_ok,
                f"{prefix}_full_physical_hit_correct": hit_ok,
                "mode_type": first["mode_type"],
            }
        )
    return full_rows


def full_accuracy_summary(rows: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    summaries = []
    for condition, group in sorted(by_condition.items()):
        by_length = {}
        for row in group:
            by_length.setdefault(str(row["length"]), []).append(row)
        summaries.append(
            {
                "condition": condition,
                "model_type": group[0]["model_type"],
                "mode": group[0]["mode"],
                "seed": int(group[0]["seed"]),
                f"{prefix}_full_nearest_block_acc": _metric_mean(group, f"{prefix}_full_nearest_block_correct"),
                f"{prefix}_full_physical_hit_acc": _metric_mean(group, f"{prefix}_full_physical_hit_correct"),
                "by_length": {
                    length: {
                        f"{prefix}_full_nearest_block_acc": _metric_mean(
                            length_rows, f"{prefix}_full_nearest_block_correct"
                        ),
                        f"{prefix}_full_physical_hit_acc": _metric_mean(
                            length_rows, f"{prefix}_full_physical_hit_correct"
                        ),
                    }
                    for length, length_rows in sorted(by_length.items(), key=lambda item: int(item[0]))
                },
            }
        )
    return {"per_condition": summaries}


def aggregate_seed_metric(per_condition: list[dict[str, Any]], metric_key: str) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = {}
    persistence = None
    for row in per_condition:
        if row["model_type"] == "persistence":
            persistence = float(row.get(metric_key, 0.0))
            continue
        grouped.setdefault((row["model_type"], row["mode"]), []).append(float(row.get(metric_key, 0.0)))
    return {
        "groups": {
            f"{model}_{mode}": {
                "model_type": model,
                "mode": mode,
                "values": values,
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
            for (model, mode), values in sorted(grouped.items())
        },
        "persistence": persistence,
    }


def autoregressive_growth_metrics(predictions: list[dict[str, Any]], fk: FKEvaluator) -> dict[str, Any]:
    joint_limits = fk.joint_limits()
    by_timestep: dict[int, list[float]] = {}
    by_rank: dict[int, list[float]] = {}
    violation_steps = 0
    total_steps = 0
    for prediction in predictions:
        step_indices = np.nonzero(prediction["loss_mask"].astype(bool))[0]
        pred = prediction["pred_phys"][step_indices]
        target = prediction["joints_phys"][step_indices + 1]
        rmse_deg = np.sqrt(np.mean(((pred - target) * 180.0 / math.pi) ** 2, axis=1))
        violation = np.any((pred < joint_limits[:, 0]) | (pred > joint_limits[:, 1]), axis=1)
        violation_steps += int(violation.sum())
        total_steps += int(violation.size)
        for step, value in zip(step_indices.tolist(), rmse_deg.tolist()):
            by_timestep.setdefault(int(step), []).append(float(value))
        for rank, value in zip(prediction["rank"][step_indices].tolist(), rmse_deg.tolist()):
            by_rank.setdefault(int(rank), []).append(float(value))
    return {
        "joint_limit_violation_rate": 0.0 if total_steps == 0 else float(violation_steps / total_steps),
        "joint_limit_violation_steps": int(violation_steps),
        "evaluated_steps": int(total_steps),
        "trajectory_rmse_growth_by_timestep": {
            str(key): float(np.mean(values)) for key, values in sorted(by_timestep.items())
        },
        "trajectory_rmse_growth_by_segment_rank": {
            str(key): float(np.mean(values)) for key, values in sorted(by_rank.items())
        },
    }


def run_accuracy_suite(
    config: dict[str, Any],
    *,
    device_name: str,
    posthoc_root: Path = POSTHOC_ROOT,
) -> dict[str, Any]:
    accuracy_dir = posthoc_root / "accuracy"
    fk_dir = posthoc_root / "fk"
    accuracy_dir.mkdir(parents=True, exist_ok=True)
    fk_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(config)
    manifest_path = Path(str(config["canonical_root"])) / "manifest.json"
    items = _load_test_items(manifest_path, load_images=True)
    shuffled_images = _shuffled_images_by_seq_id(items)
    samples = _sample_by_seq_id(manifest)
    selection = select_final_checkpoints(config, posthoc_root=posthoc_root)
    specs = _condition_specs(selection)
    device = resolve_device(device_name)
    k = int(manifest["k_samples_per_segment"])

    fk = FKEvaluator(manifest, split="test")
    try:
        fk_validation = fk.validate()
        write_json(fk_dir / "fk_validation.json", fk_validation)
        write_json(fk_dir / "identified_joint_and_site_mapping.json", fk.mapping_metadata())
        write_json(fk_dir / "block_geometry.json", fk.block_geometry)

        step_rows = []
        teacher_token_rows = []
        ar_token_rows = []
        ar_metrics = {"mode_name": "open_loop_joint_feedback_with_exogenous_images", "per_condition": []}
        failures = []
        for spec in specs:
            try:
                model = None
                if spec["model_type"] != "persistence":
                    model = _load_model_for_checkpoint(spec["model_type"], spec["checkpoint"], manifest, device)
                predictions = [
                    _teacher_forced_prediction(
                        item=item,
                        model=model,
                        model_type=spec["model_type"],
                        mode=spec["mode"],
                        manifest=manifest,
                        device=device,
                        shuffled_images=shuffled_images,
                    )
                    for item in items
                ]
                step_summary = _step_accuracy_summary_for_predictions(
                    predictions,
                    thresholds=STEP_THRESHOLDS,
                    condition_meta=spec,
                )
                curve_summary = _step_accuracy_summary_for_predictions(
                    predictions,
                    thresholds=TOLERANCE_CURVE_THRESHOLDS,
                    condition_meta=spec,
                )
                step_summary["tolerance_curve"] = curve_summary["thresholds"]
                step_rows.append(step_summary)
                if fk_validation["passed"]:
                    teacher_token_rows.extend(
                        token_rows_from_predictions(
                            predictions,
                            condition_meta=spec,
                            manifest_samples=samples,
                            fk=fk,
                            k=k,
                            autoregressive=False,
                        )
                    )

                ar_predictions = [
                    _autoregressive_prediction(
                        item=item,
                        model=model,
                        model_type=spec["model_type"],
                        mode=spec["mode"],
                        manifest=manifest,
                        device=device,
                        shuffled_images=shuffled_images,
                    )
                    for item in items
                ]
                ar_growth = autoregressive_growth_metrics(ar_predictions, fk)
                ar_condition_token_rows = []
                if fk_validation["passed"]:
                    ar_condition_token_rows = token_rows_from_predictions(
                        ar_predictions,
                        condition_meta=spec,
                        manifest_samples=samples,
                        fk=fk,
                        k=k,
                        autoregressive=True,
                    )
                    ar_token_rows.extend(ar_condition_token_rows)
                ar_full_rows = full_trial_rows_from_token_rows(ar_condition_token_rows, prefix="ar")
                ar_token_summary = token_accuracy_summary(ar_condition_token_rows, prefix="ar") if ar_condition_token_rows else {}
                ar_full_summary = full_accuracy_summary(ar_full_rows, prefix="ar") if ar_full_rows else {}
                ar_metrics["per_condition"].append(
                    {
                        **spec,
                        "growth": ar_growth,
                        "token_accuracy": ar_token_summary,
                        "full_accuracy": ar_full_summary,
                    }
                )
            except Exception as exc:  # preserve tracebacks and continue
                import traceback

                failures.append(
                    {
                        **spec,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
        write_json(accuracy_dir / "step_accuracy_per_seed.json", {"per_condition": step_rows, "failures": failures})
        write_json(accuracy_dir / "step_accuracy_aggregate.json", aggregate_step_accuracy(step_rows))
        write_tolerance_curve_csv(accuracy_dir / "tolerance_curves.csv", step_rows)

        if fk_validation["passed"]:
            write_parquet(accuracy_dir / "segment_endpoint_predictions.parquet", teacher_token_rows)
            token_summary = token_accuracy_summary(teacher_token_rows, prefix="tf")
            token_summary["aggregate_tf_token_nearest_block_acc"] = aggregate_seed_metric(
                token_summary["per_condition"], "tf_token_nearest_block_acc"
            )
            token_summary["aggregate_tf_token_physical_hit_acc"] = aggregate_seed_metric(
                token_summary["per_condition"], "tf_token_physical_hit_acc"
            )
            write_json(accuracy_dir / "token_accuracy_per_seed.json", token_summary)

            full_rows = full_trial_rows_from_token_rows(teacher_token_rows, prefix="tf")
            write_parquet(accuracy_dir / "full_trial_predictions.parquet", full_rows)
            full_summary = full_accuracy_summary(full_rows, prefix="tf")
            full_summary["aggregate_tf_full_nearest_block_acc"] = aggregate_seed_metric(
                full_summary["per_condition"], "tf_full_nearest_block_acc"
            )
            full_summary["aggregate_tf_full_physical_hit_acc"] = aggregate_seed_metric(
                full_summary["per_condition"], "tf_full_physical_hit_acc"
            )
            write_json(accuracy_dir / "full_accuracy_per_seed.json", full_summary)
        else:
            write_json(
                accuracy_dir / "token_accuracy_per_seed.json",
                {"skipped": True, "reason": "fk_validation_failed", "fk_validation": fk_validation},
            )
            write_json(
                accuracy_dir / "full_accuracy_per_seed.json",
                {"skipped": True, "reason": "fk_validation_failed", "fk_validation": fk_validation},
            )
            write_parquet(accuracy_dir / "segment_endpoint_predictions.parquet", [])
            write_parquet(accuracy_dir / "full_trial_predictions.parquet", [])

        if ar_token_rows:
            ar_metrics["token_summary"] = token_accuracy_summary(ar_token_rows, prefix="ar")
            ar_full_rows = full_trial_rows_from_token_rows(ar_token_rows, prefix="ar")
            ar_metrics["full_summary"] = full_accuracy_summary(ar_full_rows, prefix="ar")
            ar_metrics["aggregate_ar_token_nearest_block_acc"] = aggregate_seed_metric(
                ar_metrics["token_summary"]["per_condition"], "ar_token_nearest_block_acc"
            )
            ar_metrics["aggregate_ar_token_physical_hit_acc"] = aggregate_seed_metric(
                ar_metrics["token_summary"]["per_condition"], "ar_token_physical_hit_acc"
            )
        write_json(accuracy_dir / "autoregressive_metrics.json", ar_metrics)
        result = {
            "selection": selection,
            "fk_validation": fk_validation,
            "step_conditions": len(step_rows),
            "teacher_forced_token_rows": len(teacher_token_rows),
            "autoregressive_token_rows": len(ar_token_rows),
            "failures": failures,
        }
        write_json(posthoc_root / "accuracy_summary.json", result)
        return result
    finally:
        fk.close()


def _find_group_metric(aggregate: dict[str, Any], group_key: str, threshold: float, metric: str) -> float | None:
    group = aggregate.get("groups", {}).get(group_key)
    if not group:
        return None
    row = group.get("thresholds", {}).get(str(threshold))
    if not row:
        return None
    return float(row.get(metric, 0.0))


def _report_group_key(row: dict[str, Any]) -> str:
    if row.get("model_type") == "persistence":
        return "persistence"
    return f"{row.get('model_type')}_{row.get('mode')}"


def _append_endpoint_token_group_table(lines: list[str], rows: list[dict[str, Any]], *, prefix: str, title: str) -> None:
    lines.extend(["", title, ""])
    lines.append(
        f"| Group | Endpoint EE err cm | Endpoint XY err cm | Endpoint Z err cm | {prefix}_token_nearest_block_acc | {prefix}_token_physical_hit_acc | Hit 0/5/10mm |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_report_group_key(row), []).append(row)
    for group_key, group_rows in sorted(grouped.items()):
        nearest = [float(row[f"{prefix}_token_nearest_block_acc"]) for row in group_rows]
        hit = [float(row[f"{prefix}_token_physical_hit_acc"]) for row in group_rows]
        ee = [float(row["endpoint_ee_error_cm_mean"]) for row in group_rows]
        xy = [float(row["endpoint_xy_error_cm_mean"]) for row in group_rows]
        z = [float(row["endpoint_z_error_cm_mean"]) for row in group_rows]
        margins = {
            margin: [float(row["physical_hit_margin_sensitivity"][margin]) for row in group_rows]
            for margin in ["0mm", "5mm", "10mm"]
        }
        lines.append(
            f"| {group_key} | {np.mean(ee):.4f} | {np.mean(xy):.4f} | {np.mean(z):.4f} | "
            f"{np.mean(nearest):.6f} | {np.mean(hit):.6f} | "
            f"{np.mean(margins['0mm']):.6f}/{np.mean(margins['5mm']):.6f}/{np.mean(margins['10mm']):.6f} |"
        )


def _append_by_length_table(
    lines: list[str],
    rows: list[dict[str, Any]],
    *,
    prefix: str,
    metric_kind: str,
    title: str,
) -> None:
    nearest_key = f"{prefix}_{metric_kind}_nearest_block_acc"
    hit_key = f"{prefix}_{metric_kind}_physical_hit_acc"
    lines.extend(["", title, ""])
    lines.append(f"| Group | Length | {nearest_key} | {hit_key} |")
    lines.append("| --- | ---: | ---: | ---: |")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        for length, metrics in row.get("by_length", {}).items():
            grouped.setdefault((_report_group_key(row), str(length)), []).append(metrics)
    for (group_key, length), metric_rows in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        nearest = [float(row[nearest_key]) for row in metric_rows]
        hit = [float(row[hit_key]) for row in metric_rows]
        lines.append(f"| {group_key} | {length} | {np.mean(nearest):.6f} | {np.mean(hit):.6f} |")


def write_final_report(posthoc_root: Path = POSTHOC_ROOT, report_path: Path = FINAL_REPORT_PATH) -> None:
    convergence = read_json(posthoc_root / "convergence" / "convergence_per_run.json")
    selection = read_json(posthoc_root / "convergence" / "final_checkpoint_selection.json")
    lineage = read_json(posthoc_root / "convergence" / "continuation_lineage.json")
    step_per_seed = read_json(posthoc_root / "accuracy" / "step_accuracy_per_seed.json")
    step_aggregate = read_json(posthoc_root / "accuracy" / "step_accuracy_aggregate.json")
    token_summary = read_json(posthoc_root / "accuracy" / "token_accuracy_per_seed.json")
    full_summary = read_json(posthoc_root / "accuracy" / "full_accuracy_per_seed.json")
    ar_metrics = read_json(posthoc_root / "accuracy" / "autoregressive_metrics.json")
    fk_validation = read_json(posthoc_root / "fk" / "fk_validation.json")
    lines = [
        "# Corsi Motion Convergence And Accuracy",
        "",
        "Date: 2026-06-22",
        "",
        "## Interpretation Limits",
        "",
        "- timestep tolerance accuracy is a thresholded continuous-regression metric, not classification accuracy",
        "- token accuracy means segment-endpoint block identity/hit",
        "- teacher-forced full accuracy is not closed-loop robot success",
        "- autoregressive evaluation uses exogenous recorded images and is therefore still not a closed-loop visual-motor execution test",
        "- these metrics evaluate motion prediction, not Corsi working-memory recall",
        "",
        "## Loss And Resume Audit",
        "",
        f"Training loss mask: `{convergence['loss_mask_audit']['conclusion']}`.",
        "The original checkpoints lack scheduler, AMP scaler, and RNG states, so required extensions are warm-started",
        "from `best.pt` in separate `*_extended_continuation_from_best` run directories.",
        "",
        "## Final Checkpoints",
        "",
        "| Model | Seed | Original checkpoint | Final checkpoint | Final source | Original best val | Final best val |",
        "| --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for row in selection["selections"]:
        lines.append(
            f"| {row['model_type']} | {row['seed']} | `{row['original_checkpoint']}` | `{row['final_checkpoint']}` | {row['final_source']} | {row['original_best_val_rmse']:.6f} | {row['final_best_val_rmse']:.6f} |"
        )
    lines.extend(["", "## Convergence Decisions", ""])
    lines.append("| Model | Seed | Class | Best epoch | Since best | Best val | Final val | Extend reason |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |")
    for row in convergence["per_run"]:
        stats = row["stats_50"]
        lines.append(
            f"| {row['model_type']} | {row['seed']} | {row['final_classification']} | {stats.get('best_epoch', 0)} | {stats.get('epochs_since_best', 0)} | {float(stats.get('best_val_rmse', 0.0)):.6f} | {float(stats.get('final_val_rmse', 0.0)):.6f} | {row['continuation_decision']['reason']} |"
        )
    lines.extend(["", "### Convergence Statistics Over 30 And 50 Epochs", ""])
    lines.append(
        "| Model | Seed | Window | Class | Confidence | OLS slope | Theil-Sen slope | Normalized OLS | Normalized Theil-Sen | Relative change | Pred OLS decrease | Pred Theil-Sen decrease | Train-val gap | Val worse while train decreases |"
    )
    lines.append("| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in convergence["per_run"]:
        for key in ["stats_30", "stats_50"]:
            stats = row[key]
            lines.append(
                f"| {row['model_type']} | {row['seed']} | {stats.get('window', 0)} | {stats.get('classification', '')} | {stats.get('confidence', '')} | "
                f"{float(stats.get('ols_slope', 0.0)):.8f} | {float(stats.get('theil_sen_slope', 0.0)):.8f} | "
                f"{float(stats.get('normalized_ols_slope', 0.0)):.8f} | {float(stats.get('normalized_theil_sen_slope', 0.0)):.8f} | "
                f"{float(stats.get('relative_change', 0.0)):.6f} | {float(stats.get('predicted_relative_decrease_ols', 0.0)):.6f} | "
                f"{float(stats.get('predicted_relative_decrease_theil_sen', 0.0)):.6f} | "
                f"{float(stats.get('train_validation_gap_train_loss_minus_val_mse', 0.0)):.8f} | "
                f"{stats.get('validation_worsens_while_train_decreases', False)} |"
            )
    lines.extend(["", "Extended runs:", ""])
    if lineage.get("runs"):
        for row in lineage["runs"]:
            final_summary = read_json(Path(row["extension_run_dir"]) / "summary.json")
            source_best = float(row.get("source_checkpoint_epoch", 0))
            lines.append(
                f"- `{row['original_run']}` -> `{row['extension_run_name']}` warm-start from best epoch {source_best:.0f}; final epoch {final_summary['epochs_completed']}, best val {final_summary['best_val_normalized_rmse']:.6f}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Final Test Metrics", ""])
    lines.append("Validation-only checkpoint selection was completed before reading final test accuracy outputs.")
    lines.append("")
    lines.append("| Group | Normalized RMSE mean | Step acc @1deg mean | Step acc @5deg mean |")
    lines.append("| --- | ---: | ---: | ---: |")
    for group_key in sorted(step_aggregate.get("groups", {})):
        rmse = _find_group_metric(step_aggregate, group_key, 5.0, "normalized_rmse_mean")
        acc1 = _find_group_metric(step_aggregate, group_key, 1.0, "mean")
        acc5 = _find_group_metric(step_aggregate, group_key, 5.0, "mean")
        lines.append(f"| {group_key} | {rmse if rmse is not None else 0.0:.6f} | {acc1 if acc1 is not None else 0.0:.6f} | {acc5 if acc5 is not None else 0.0:.6f} |")
    if step_aggregate.get("persistence"):
        p = step_aggregate["persistence"]["thresholds"]
        lines.append(
            f"| persistence_normal | {p['5.0']['normalized_rmse']:.6f} | {p['1.0']['all_joint_step_accuracy']:.6f} | {p['5.0']['all_joint_step_accuracy']:.6f} |"
        )
    lines.extend(["", "## Step Accuracy Thresholds", ""])
    for group_key, group in sorted(step_aggregate.get("groups", {}).items()):
        lines.append(f"### {group_key}")
        lines.append("")
        lines.append("| Threshold deg | Mean | SD | Bootstrap 95% CI |")
        lines.append("| ---: | ---: | ---: | --- |")
        for threshold in STEP_THRESHOLDS:
            row = group["thresholds"][str(threshold)]
            lines.append(
                f"| {threshold} | {row['mean']:.6f} | {row['sd']:.6f} | [{row['episode_bootstrap_95ci'][0]:.6f}, {row['episode_bootstrap_95ci'][1]:.6f}] |"
            )
        lines.append("")
    lines.extend(["## Token Accuracy", ""])
    lines.append("Teacher-forced metrics use endpoint prediction from `e-1` and never the boundary transition from `e`.")
    for key in ["aggregate_tf_token_nearest_block_acc", "aggregate_tf_token_physical_hit_acc"]:
        if key in token_summary:
            lines.append(f"- `{key}`: `{json.dumps(token_summary[key], sort_keys=True)}`")
    _append_endpoint_token_group_table(
        lines,
        token_summary.get("per_condition", []),
        prefix="tf",
        title="### Teacher-Forced Endpoint Error And Token Accuracy",
    )
    _append_by_length_table(
        lines,
        token_summary.get("per_condition", []),
        prefix="tf",
        metric_kind="token",
        title="### Teacher-Forced Token Accuracy By Length",
    )
    lines.extend(["", "## Full-Trial Accuracy", ""])
    for key in ["aggregate_tf_full_nearest_block_acc", "aggregate_tf_full_physical_hit_acc"]:
        if key in full_summary:
            lines.append(f"- `{key}`: `{json.dumps(full_summary[key], sort_keys=True)}`")
    _append_by_length_table(
        lines,
        full_summary.get("per_condition", []),
        prefix="tf",
        metric_kind="full",
        title="### Teacher-Forced Full-Trial Accuracy By Length",
    )
    lines.extend(["", "## Autoregressive Diagnostic", ""])
    lines.append("Mode name: `open_loop_joint_feedback_with_exogenous_images`.")
    for key in ["aggregate_ar_token_nearest_block_acc", "aggregate_ar_token_physical_hit_acc"]:
        if key in ar_metrics:
            lines.append(f"- `{key}`: `{json.dumps(ar_metrics[key], sort_keys=True)}`")
    _append_endpoint_token_group_table(
        lines,
        ar_metrics.get("token_summary", {}).get("per_condition", []),
        prefix="ar",
        title="### Autoregressive Endpoint Error And Token Accuracy",
    )
    _append_by_length_table(
        lines,
        ar_metrics.get("token_summary", {}).get("per_condition", []),
        prefix="ar",
        metric_kind="token",
        title="### Autoregressive Token Accuracy By Length",
    )
    _append_by_length_table(
        lines,
        ar_metrics.get("full_summary", {}).get("per_condition", []),
        prefix="ar",
        metric_kind="full",
        title="### Autoregressive Full-Trial Accuracy By Length",
    )
    lines.extend(["", "### Autoregressive Joint-Limit And Growth Diagnostics", ""])
    lines.append("| Group | Joint-limit violation rate | Evaluated steps | RMSE growth timestep entries | RMSE growth rank entries |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    ar_grouped: dict[str, list[dict[str, Any]]] = {}
    for row in ar_metrics.get("per_condition", []):
        ar_grouped.setdefault(_report_group_key(row), []).append(row)
    for group_key, rows in sorted(ar_grouped.items()):
        growth_rows = [row.get("growth", {}) for row in rows]
        violation = [float(row.get("joint_limit_violation_rate", 0.0)) for row in growth_rows]
        steps = [int(row.get("evaluated_steps", 0)) for row in growth_rows]
        timestep_entries = [len(row.get("trajectory_rmse_growth_by_timestep", {})) for row in growth_rows]
        rank_entries = [len(row.get("trajectory_rmse_growth_by_segment_rank", {})) for row in growth_rows]
        lines.append(
            f"| {group_key} | {np.mean(violation):.6f} | {int(np.sum(steps))} | {int(np.mean(timestep_entries))} | {int(np.mean(rank_entries))} |"
        )
    lines.extend(["", "## FK Validation", ""])
    lines.append(
        f"FK validation passed: `{fk_validation['passed']}`; median error {fk_validation['median_position_error_mm']:.3f} mm, max error {fk_validation['max_position_error_mm']:.3f} mm."
    )
    lines.extend(["", "## Failed Seeds Or Unavailable Histories", ""])
    failures = step_per_seed.get("failures", [])
    if failures:
        for failure in failures:
            lines.append(f"- {failure['condition']}: {failure['error_type']} {failure['error']}")
    else:
        lines.append("- no accuracy seed failures recorded")
    continuation_notes = []
    for row in lineage.get("runs", []):
        summaries = [row.get("extension_summary") or {}, row.get("second_extension_summary") or {}]
        for summary in summaries:
            reason = summary.get("stopped_reason")
            if reason and reason not in {"early_stopping", "max_epochs_reached"}:
                continuation_notes.append(
                    f"{row.get('extension_run_name')}: stopped at epoch {summary.get('epochs_completed')} because `{reason}`"
                )
    if continuation_notes:
        lines.append("- continuation stopped reasons:")
        for note in continuation_notes:
            lines.append(f"  - {note}")
    unavailable = [
        row
        for row in convergence["per_run"]
        if not row["history_available"] or not row["best_warm_start_possible"]
    ]
    if unavailable:
        for row in unavailable:
            lines.append(f"- unavailable history/checkpoint: {row['run_name']}")
    lines.extend(
        [
            "",
            "## Commands And Runtime",
            "",
            "Primary resumable command:",
            "",
            "```bash",
            "conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.posthoc_suite --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json --resume --devices auto",
            "```",
            "",
            f"Continuation runtime seconds recorded in lineage: {lineage.get('runtime_seconds', 0.0):.1f}",
            "",
            "## Output Paths",
            "",
            f"- convergence: `{posthoc_root / 'convergence'}`",
            f"- accuracy: `{posthoc_root / 'accuracy'}`",
            f"- fk: `{posthoc_root / 'fk'}`",
            f"- final visual states: `{posthoc_root / 'states'}`",
            f"- audit report: `{AUDIT_REPORT_PATH}`",
            f"- final report: `{FINAL_REPORT_PATH}`",
            "",
            "## Finish Checklist",
            "",
            "- files changed: see git status and this report's output paths",
            "- tests passed: recorded in final assistant response after test execution",
            "- runs extended: see `continuation_lineage.json`",
            "- final convergence decisions: see `convergence_per_run.json`",
            "- final checkpoints: see `final_checkpoint_selection.json`",
            "- exact resume command: listed above",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_devices(value: str) -> list[str]:
    if value == "auto":
        return ["cuda:0" if torch.cuda.is_available() else "cpu"]
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Corsi posthoc convergence and accuracy suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--skip-continuation", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    posthoc_root = POSTHOC_ROOT
    convergence = audit_and_classify(config, posthoc_root=posthoc_root)
    devices = parse_devices(args.devices)
    if not args.skip_continuation:
        run_continuations(config, convergence, device=devices[0], posthoc_root=posthoc_root)
        convergence = audit_and_classify(config, posthoc_root=posthoc_root)
    selection = select_final_checkpoints(config, posthoc_root=posthoc_root)
    export_final_visual_states(config, selection, device_name=devices[0], posthoc_root=posthoc_root)
    if not args.skip_accuracy:
        run_accuracy_suite(config, device_name=devices[0], posthoc_root=posthoc_root)
        write_final_report(posthoc_root=posthoc_root, report_path=FINAL_REPORT_PATH)
    print(
        json.dumps(
            {
                "posthoc_root": str(posthoc_root),
                "audit_report": str(AUDIT_REPORT_PATH),
                "final_report": str(FINAL_REPORT_PATH),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
