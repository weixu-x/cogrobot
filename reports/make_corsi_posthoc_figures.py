"""Generate compact figures for the Corsi posthoc convergence/accuracy report."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1")
OUT = Path("reports/figures")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def group_key(row: dict) -> str:
    if row.get("model_type") == "persistence":
        return "persistence"
    return f"{row.get('model_type')}_{row.get('mode')}"


def metric_by_group(rows: list[dict], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(float(row[metric]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def final_validation_rmse() -> Path:
    selection = read_json(ROOT / "convergence" / "final_checkpoint_selection.json")
    rows = selection["selections"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=160, sharey=True)
    colors = {"original": "#9aa3ad", "final": "#2f7f73"}
    for axis, model_type in zip(axes, ["visual_joint", "joint_only"]):
        model_rows = [row for row in rows if row["model_type"] == model_type]
        seeds = [str(row["seed"]) for row in model_rows]
        x = np.arange(len(seeds))
        original = [float(row["original_best_val_rmse"]) for row in model_rows]
        final = [float(row["final_best_val_rmse"]) for row in model_rows]
        axis.bar(x - 0.18, original, width=0.36, color=colors["original"], label="original")
        axis.bar(x + 0.18, final, width=0.36, color=colors["final"], label="final selected")
        for index, (orig, fin) in enumerate(zip(original, final)):
            delta = orig - fin
            if delta > 1e-8:
                axis.text(index, fin + 0.0007, f"-{delta:.4f}", ha="center", va="bottom", fontsize=8, color="#20584f")
        axis.set_title(model_type)
        axis.set_xlabel("seed")
        axis.set_xticks(x, seeds)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("validation normalized RMSE")
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Final checkpoint selection by validation RMSE", y=0.98)
    fig.tight_layout()
    path = OUT / "corsi_validation_rmse_original_vs_final.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def accuracy_overview() -> Path:
    step = read_json(ROOT / "accuracy" / "step_accuracy_aggregate.json")
    token = read_json(ROOT / "accuracy" / "token_accuracy_per_seed.json")
    full = read_json(ROOT / "accuracy" / "full_accuracy_per_seed.json")
    ar = read_json(ROOT / "accuracy" / "autoregressive_metrics.json")

    groups = [
        "visual_joint_normal",
        "joint_only_normal",
        "persistence",
        "visual_joint_shuffled_vision",
        "visual_joint_zero_vision",
    ]
    display_groups = ["visual", "joint", "persistence", "visual shuffled", "visual zero"]
    metrics = [
        ("Step @1 deg", {}),
        ("TF token hit", metric_by_group(token["per_condition"], "tf_token_physical_hit_acc")),
        ("TF full hit", metric_by_group(full["per_condition"], "tf_full_physical_hit_acc")),
        ("AR token hit", metric_by_group(ar["token_summary"]["per_condition"], "ar_token_physical_hit_acc")),
        ("AR full hit", metric_by_group(ar["full_summary"]["per_condition"], "ar_full_physical_hit_acc")),
    ]

    step_values = {}
    for key, row in step.get("groups", {}).items():
        step_values[key] = float(row["thresholds"]["1.0"]["mean"])
    if step.get("persistence"):
        step_values["persistence"] = float(step["persistence"]["thresholds"]["1.0"]["all_joint_step_accuracy"])
    metrics[0][1].update(step_values)

    data = np.asarray([[values.get(group, np.nan) for group in groups] for _, values in metrics], dtype=np.float64)
    fig, axis = plt.subplots(figsize=(10.8, 4.7), dpi=160)
    image = axis.imshow(data, vmin=0.0, vmax=1.0, cmap="YlGnBu")
    axis.set_xticks(np.arange(len(groups)), display_groups, rotation=22, ha="right")
    axis.set_yticks(np.arange(len(metrics)), [name for name, _ in metrics])
    axis.set_title("Motion prediction accuracy overview")
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            if np.isfinite(value):
                axis.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=9, color="#111")
    cbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    cbar.set_label("accuracy")
    fig.tight_layout()
    path = OUT / "corsi_motion_accuracy_overview.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in [final_validation_rmse(), accuracy_overview()]:
        print(path)


if __name__ == "__main__":
    main()
