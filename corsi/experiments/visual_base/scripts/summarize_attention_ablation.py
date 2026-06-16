"""Collect evaluate_visual.py summaries into one attention ablation CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="corsi_artifacts/visual_base/training/attention_ablation")
    parser.add_argument("--output-csv", default="corsi_artifacts/visual_base/results/attention_ablation_summary.csv")
    return parser.parse_args()


def flatten_summary(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload.get("metrics", {})
    attention = metrics.get("attention", {})
    taxonomy = metrics.get("error_taxonomy", {})
    row: Dict[str, Any] = {
        "run_id": path.parents[1].name,
        "model_name": path.parents[1].name,
        "eval_id": path.parent.name,
        "attention_type": payload.get("attention_type", ""),
        "data_root": payload.get("data_root", ""),
        "eval_mode": payload.get("eval_mode", ""),
        "token_accuracy": metrics.get("token_accuracy", 0.0),
        "full_sequence_accuracy": metrics.get("full_sequence_accuracy", 0.0),
        "mean_first_error_position": metrics.get("error_analysis", {}).get("mean_first_error_pos"),
        "checkpoint_path": payload.get("checkpoint_path", ""),
    }
    for length, stats in metrics.get("per_length", {}).items():
        row[f"token_accuracy_len_{length}"] = stats.get("token_acc", 0.0)
        row[f"full_sequence_accuracy_len_{length}"] = stats.get("full_seq_acc", 0.0)
    row.update(attention)
    row.update(taxonomy)
    return row


def main() -> None:
    args = parse_args()
    summaries = sorted(Path(args.result_root).glob("*/eval_*/summary_metrics.json"))
    rows: List[Dict[str, Any]] = [flatten_summary(path) for path in summaries]
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
