"""Plot attention diagnostics saved by evaluate_visual.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Attention")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = np.load(args.attention_npz)
    weights = payload["attention_weights"]
    lengths = payload["target_lengths"]

    mean_attention = weights.mean(axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mean_attention, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("encoder time")
    ax.set_ylabel("decoder output step")
    ax.set_title(f"{args.title}: mean attention")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "attention_heatmap_all.png", dpi=180)
    plt.close(fig)

    entropy_by_step = []
    for step in range(weights.shape[1]):
        valid = []
        for trial_index, length in enumerate(lengths):
            if step >= int(length):
                continue
            row = weights[trial_index, step]
            total = row.sum()
            if total > 0:
                row = row / total
            valid.append(float(-(row * np.log(np.clip(row, 1e-8, None))).sum()))
        entropy_by_step.append(np.nan if not valid else float(np.mean(valid)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(entropy_by_step)), entropy_by_step, marker="o")
    ax.set_xlabel("decoder output step")
    ax.set_ylabel("attention entropy")
    ax.set_title(f"{args.title}: entropy by step")
    fig.tight_layout()
    fig.savefig(output_dir / "attention_entropy_curve.png", dpi=180)
    plt.close(fig)

    displacements = []
    for trial_index, length in enumerate(lengths):
        for step in range(int(length)):
            displacements.append(int(weights[trial_index, step].argmax() - step))
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(min(displacements, default=0) - 0.5, max(displacements, default=0) + 1.5)
    ax.hist(displacements, bins=bins)
    ax.set_xlabel("peak index - output step")
    ax.set_ylabel("count")
    ax.set_title(f"{args.title}: peak displacement")
    fig.tight_layout()
    fig.savefig(output_dir / "peak_displacement_distribution.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
