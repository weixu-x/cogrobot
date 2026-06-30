"""Generate visualizations for the V2 LSTM memory onset run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN = "stage2_seed0_lstm_memory_dmem64_onset_20260629"
RUN_ROOT = Path("corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12") / RUN
EVAL_ROOT = Path("corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12")
FIG_ROOT = Path("reports/figures/v2_lstm_onset_20260629")
MILESTONES = [10, 20, 40, 80, 120, 160, 200]
MILESTONE_LABELS = [f"epoch{value:03d}" for value in MILESTONES]
BLOCK_COLORS = {
    -1: "#f1f5f9",
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
    6: "#e377c2",
    7: "#7f7f7f",
    8: "#bcbd22",
    9: "#17becf",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def savefig(name: str) -> Path:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    path = FIG_ROOT / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def metric(history: list[dict], split: str, key: str) -> np.ndarray:
    return np.array([row.get(split, {}).get(key, np.nan) for row in history], dtype=float)


def load_merged_history() -> list[dict]:
    """Merge segmented resume summaries into a single epoch-indexed history."""

    rows_by_epoch: dict[int, dict] = {}
    summary_paths = [RUN_ROOT / "summary.json"]
    summary_paths.extend(sorted((RUN_ROOT / "milestones").glob("epoch_*/summary.json")))
    for path in summary_paths:
        if not path.exists():
            continue
        summary = load_json(path)
        for row in summary.get("history", []):
            rows_by_epoch[int(row["epoch"])] = row
    return [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]


def plot_training_curves(history: list[dict]) -> None:
    epochs = np.array([row["epoch"] + 1 for row in history], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.ravel()

    axes[0].plot(epochs, metric(history, "train", "loss"), label="train loss", color="#0f766e")
    axes[0].plot(epochs, metric(history, "val", "loss"), label="val loss", color="#b91c1c")
    axes[0].set_title("Total Loss")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(epochs, metric(history, "train", "seq_loss"), label="train seq", color="#2563eb")
    axes[1].plot(epochs, metric(history, "val", "seq_loss"), label="val seq", color="#60a5fa")
    axes[1].plot(epochs, metric(history, "train", "memory_order_loss"), label="train memory-order", color="#7c3aed")
    axes[1].plot(epochs, metric(history, "val", "memory_order_loss"), label="val memory-order", color="#c084fc")
    axes[1].set_title("Main Loss Components")
    axes[1].legend(fontsize=8)

    axes[2].plot(epochs, metric(history, "train", "joint_loss"), label="train joint", color="#16a34a")
    axes[2].plot(epochs, metric(history, "val", "joint_loss"), label="val joint", color="#86efac")
    axes[2].plot(epochs, metric(history, "train", "ee_pose_loss"), label="train ee pose", color="#ea580c")
    axes[2].plot(epochs, metric(history, "val", "ee_pose_loss"), label="val ee pose", color="#fdba74")
    axes[2].set_title("Grounding Aux Losses")
    axes[2].set_xlabel("epoch")
    axes[2].legend(fontsize=8)

    axes[3].plot(epochs, metric(history, "train", "coord_loss"), label="train coord", color="#0e7490")
    axes[3].plot(epochs, metric(history, "val", "coord_loss"), label="val coord", color="#67e8f9")
    axes[3].plot(epochs, metric(history, "train", "ee_xy_loss"), label="train ee xy", color="#be123c")
    axes[3].plot(epochs, metric(history, "val", "ee_xy_loss"), label="val ee xy", color="#fda4af")
    axes[3].set_title("Coordinate Aux Losses")
    axes[3].set_xlabel("epoch")
    axes[3].legend(fontsize=8)

    for axis in axes:
        axis.grid(True, alpha=0.25)
        for mark in MILESTONES:
            axis.axvline(mark, color="#94a3b8", linewidth=0.8, alpha=0.45)

    fig.suptitle("LSTM Memory Onset Training Curves", fontsize=15, y=1.02)
    savefig("training_loss_curves.png")


def plot_validation_curves(history: list[dict]) -> None:
    epochs = np.array([row["epoch"] + 1 for row in history], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.ravel()

    axes[0].plot(epochs, metric(history, "val", "full_sequence_accuracy"), label="full sequence", color="#1d4ed8")
    axes[0].plot(epochs, metric(history, "val", "token_accuracy"), label="token", color="#15803d")
    axes[0].axhline(0.325, linestyle="--", color="#1d4ed8", alpha=0.45, label="slot full val")
    axes[0].axhline(0.627, linestyle="--", color="#15803d", alpha=0.45, label="slot token val")
    axes[0].set_title("Autonomous Recall Accuracy")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=8)

    axes[1].plot(epochs, metric(history, "val", "eos_accuracy"), label="EOS", color="#9333ea")
    axes[1].plot(epochs, metric(history, "val", "predicted_length_accuracy"), label="predicted length", color="#f97316")
    axes[1].set_title("Stop/Length Behavior")
    axes[1].legend(fontsize=8)

    axes[2].plot(epochs, metric(history, "val", "duplicate_sequence_rate"), label="duplicate sequence", color="#dc2626")
    axes[2].plot(epochs, metric(history, "val", "mean_duplicate_count"), label="mean duplicate count", color="#991b1b")
    axes[2].axhline(0.525, linestyle="--", color="#dc2626", alpha=0.45, label="slot dup val")
    axes[2].set_title("Duplicate Collapse")
    axes[2].set_xlabel("epoch")
    axes[2].legend(fontsize=8)

    axes[3].plot(epochs, metric(history, "val", "mean_unique_predicted_blocks"), label="unique predicted blocks", color="#0891b2")
    axes[3].plot(epochs, metric(history, "val", "mean_set_overlap_jaccard"), label="set overlap", color="#4d7c0f")
    axes[3].set_title("Set Recovery")
    axes[3].set_xlabel("epoch")
    axes[3].legend(fontsize=8)

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.set_ylim(bottom=0)
        for mark in MILESTONES:
            axis.axvline(mark, color="#94a3b8", linewidth=0.8, alpha=0.45)

    fig.suptitle("Validation Behavior Across Training", fontsize=15, y=1.02)
    savefig("validation_metric_curves.png")


def load_eval(milestone: int, checkpoint: str, split: str) -> dict:
    return load_json(EVAL_ROOT / f"{RUN}_epoch{milestone:03d}_{checkpoint}_{split}_eval.json")


def plot_milestone_comparison() -> None:
    checkpoints = ["latest", "best_full_sequence", "best_token", "best_val_loss"]
    metrics = [
        ("full_sequence_accuracy", "Full Sequence"),
        ("token_accuracy", "Token"),
        ("duplicate_sequence_rate", "Duplicate Rate"),
        ("loss", "Loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.ravel()

    for axis, (key, title) in zip(axes, metrics):
        for checkpoint in checkpoints:
            values = [load_eval(m, checkpoint, "val")[key] for m in MILESTONES]
            axis.plot(MILESTONES, values, marker="o", label=checkpoint.replace("_", " "))
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        if key != "loss":
            axis.set_ylim(0, 1.02)
        axis.set_xlabel("milestone epoch")
    axes[0].legend(fontsize=8)
    fig.suptitle("Validation Metrics by Retained Checkpoint Choice", fontsize=15, y=1.02)
    savefig("milestone_checkpoint_comparison.png")


def matrix_for(metric_key: str, split: str, checkpoint: str = "best_full_sequence") -> np.ndarray:
    rows = []
    for milestone in MILESTONES:
        data = load_eval(milestone, checkpoint, split)["per_length_metrics"]
        rows.append([data[str(length)][metric_key] for length in range(2, 10)])
    return np.array(rows, dtype=float)


def plot_heatmap(matrix: np.ndarray, title: str, filename: str, vmin: float = 0.0, vmax: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6))
    image = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks(np.arange(8), labels=[str(i) for i in range(2, 10)])
    ax.set_yticks(np.arange(len(MILESTONES)), labels=[str(i) for i in MILESTONES])
    ax.set_xlabel("sequence length")
    ax.set_ylabel("milestone epoch")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if value < 0.45 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.026, pad=0.03)
    savefig(filename)


def plot_per_length_heatmaps() -> None:
    plot_heatmap(
        matrix_for("full_sequence_accuracy", "val"),
        "Val Exact Accuracy by Length, Best-Full Checkpoint",
        "per_length_val_exact_heatmap.png",
    )
    plot_heatmap(
        matrix_for("token_accuracy", "val"),
        "Val Token Accuracy by Length, Best-Full Checkpoint",
        "per_length_val_token_heatmap.png",
    )
    plot_heatmap(
        matrix_for("duplicate_sequence_rate", "val"),
        "Val Duplicate Sequence Rate by Length, Best-Full Checkpoint",
        "per_length_val_duplicate_heatmap.png",
    )
    plot_heatmap(
        matrix_for("full_sequence_accuracy", "test"),
        "Test Exact Accuracy by Length, Best-Full Checkpoint",
        "per_length_test_exact_heatmap.png",
    )


def token_color(token: int) -> str:
    return BLOCK_COLORS.get(int(token), "#cbd5e1")


def draw_sequence_cells(ax, tokens: list[int], y: int, label: str, max_len: int = 10) -> None:
    ax.text(-0.35, y, label, ha="right", va="center", fontsize=8)
    padded = [int(t) for t in tokens[:max_len]]
    padded.extend([-1] * (max_len - len(padded)))
    for x, token in enumerate(padded):
        rect = plt.Rectangle((x, y - 0.4), 0.9, 0.8, facecolor=token_color(token), edgecolor="white", linewidth=1)
        ax.add_patch(rect)
        text = "" if token == -1 else ("EOS" if token == 9 else str(token))
        ax.text(x + 0.45, y, text, ha="center", va="center", fontsize=7, color="white" if token in {0, 3, 4, 5, 7} else "black")


def plot_prediction_progression(split: str = "val") -> None:
    sample_indices = [0, 5, 10, 15, 20, 25, 30, 35]
    max_len = 10
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(12.5, 13), sharex=True)
    for axis, index in zip(axes, sample_indices):
        base = load_eval(80, "best_full_sequence", split)["rows"][index]
        axis.set_xlim(-1.4, max_len + 0.2)
        axis.set_ylim(-0.8, len(MILESTONES) + 1.1)
        axis.axis("off")
        axis.text(-1.35, len(MILESTONES) + 0.65, f"{split} row {index}, length {base['target_length']}", ha="left", va="center", fontsize=9, weight="bold")
        draw_sequence_cells(axis, base["target_tokens"], len(MILESTONES), "target", max_len=max_len)
        for row_index, milestone in enumerate(reversed(MILESTONES)):
            row = load_eval(milestone, "best_full_sequence", split)["rows"][index]
            draw_sequence_cells(axis, row["predicted_tokens"], row_index, f"e{milestone}", max_len=max_len)
            exact = "ok" if row["exact_correct"] else "miss"
            axis.text(max_len + 0.05, row_index, f"{exact} tok={row['token_accuracy']:.2f} dup={row['duplicate_count']}", ha="left", va="center", fontsize=7)
    fig.suptitle(f"Prediction Progression Across Milestones ({split}, Best-Full)", fontsize=15, y=0.995)
    savefig(f"prediction_progression_{split}.png")


def plot_slot_comparison() -> None:
    lstm = load_eval(160, "best_full_sequence", "val")
    lstm_test = load_eval(160, "best_full_sequence", "test")
    labels = ["Val full", "Val token", "Val dup", "Test full", "Test token", "Test dup"]
    slot = [0.325, 0.627, 0.525, 0.300, 0.631, 0.575]
    lstm_values = [
        lstm["full_sequence_accuracy"],
        lstm["token_accuracy"],
        lstm["duplicate_sequence_rate"],
        lstm_test["full_sequence_accuracy"],
        lstm_test["token_accuracy"],
        lstm_test["duplicate_sequence_rate"],
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar(x - width / 2, slot, width, label="slot-compress D_mem=64", color="#64748b")
    ax.bar(x + width / 2, lstm_values, width, label="LSTM D_mem=64 onset best", color="#0f766e")
    ax.set_xticks(x, labels=labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_title("Slot-Compress vs LSTM Memory")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    for xpos, value in zip(x - width / 2, slot):
        ax.text(xpos, value + 0.025, f"{value:.2f}", ha="center", fontsize=8)
    for xpos, value in zip(x + width / 2, lstm_values):
        ax.text(xpos, value + 0.025, f"{value:.2f}", ha="center", fontsize=8)
    savefig("slot_vs_lstm_comparison.png")


def main() -> int:
    history = load_merged_history()
    plot_training_curves(history)
    plot_validation_curves(history)
    plot_milestone_comparison()
    plot_per_length_heatmaps()
    plot_prediction_progression("val")
    plot_prediction_progression("test")
    plot_slot_comparison()

    generated = sorted(FIG_ROOT.glob("*.png"))
    index = {
        "figure_root": str(FIG_ROOT),
        "figures": [str(path) for path in generated],
    }
    (FIG_ROOT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
