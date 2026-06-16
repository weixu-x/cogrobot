"""Aggregate ablation metrics into CSV and Markdown summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_metrics(metrics: dict) -> dict:
    full_by_length = metrics.get("full_sequence_accuracy_by_length", {})
    transposition = metrics.get("transposition_like_rate_by_length", {})
    return {
        "experiment_name": metrics.get("experiment_name"),
        "seed": metrics.get("seed"),
        "use_attention": metrics.get("use_attention"),
        "use_step_embedding": metrics.get("use_step_embedding"),
        "use_scheduled_sampling": metrics.get("use_scheduled_sampling"),
        "best_epoch": metrics.get("best_epoch"),
        "best_full_sequence_accuracy": metrics.get("best_full_sequence_accuracy"),
        "best_token_accuracy": metrics.get("best_token_accuracy"),
        "estimated_span": metrics.get("estimated_span"),
        "length_2_acc": full_by_length.get("2"),
        "length_3_acc": full_by_length.get("3"),
        "length_4_acc": full_by_length.get("4"),
        "length_5_acc": full_by_length.get("5"),
        "length_6_acc": full_by_length.get("6"),
        "order_error": metrics.get("order_error"),
        "wrong_block": metrics.get("wrong_block"),
        "repeat_error": metrics.get("repeat_error"),
        "transposition_like_rate": transposition.get("6", 0.0),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    metrics_paths = sorted(args.output_root.glob("*/metrics_best.json"))
    rows = [flatten_metrics(load_json(path)) for path in metrics_paths]
    summary_dir = args.output_root / "summary"
    write_csv(summary_dir / "ablation_summary.csv", rows)
    write_markdown(summary_dir / "ablation_summary.md", rows)
    print(f"Wrote {len(rows)} rows to {summary_dir}")


if __name__ == "__main__":
    main()
