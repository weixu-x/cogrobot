"""Plot error taxonomy results saved by evaluate_visual.py."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--error-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Error taxonomy")
    return parser.parse_args()


def _read_rows(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    rows = _read_rows(Path(args.error_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "wrong_block_count",
        "order_error_count",
        "repeat_error_count",
        "omission_count",
        "adjacent_transposition_count",
    ]
    by_length = defaultdict(list)
    for row in rows:
        by_length[int(row["length"])].append(row)
    lengths = sorted(by_length)
    values = {
        metric: [
            float(np.mean([float(row[metric]) for row in by_length[length]]))
            for length in lengths
        ]
        for metric in metrics
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottoms = np.zeros(len(lengths))
    for metric in metrics:
        ax.bar(lengths, values[metric], bottom=bottoms, label=metric.replace("_count", ""))
        bottoms += np.asarray(values[metric])
    ax.set_xlabel("sequence length")
    ax.set_ylabel("mean count per trial")
    ax.set_title(args.title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "error_taxonomy_by_length.png", dpi=180)
    plt.close(fig)

    first_errors = [
        int(float(row["first_error_position"]))
        for row in rows
        if row.get("first_error_position") not in {"", "None", None}
    ]
    if first_errors:
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.arange(min(first_errors) - 0.5, max(first_errors) + 1.5)
        ax.hist(first_errors, bins=bins)
        ax.set_xlabel("first error position")
        ax.set_ylabel("count")
        ax.set_title(f"{args.title}: first error position")
        fig.tight_layout()
        fig.savefig(output_dir / "first_error_position_histogram.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
