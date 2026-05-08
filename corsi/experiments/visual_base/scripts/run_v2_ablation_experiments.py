"""Run the freecam_index_v2 visual ablation experiments sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENT_ORDER = [
    "freecam_index_v2_ablate_baseline_lstm",
    "freecam_index_v2_ablate_attention_only",
    "freecam_index_v2_ablate_step_only",
    "freecam_index_v2_ablate_ss_only",
    "freecam_index_v2_ablate_attention_step",
    "freecam_index_v2_ablate_attention_ss",
    "freecam_index_v2_ablate_step_ss",
    "freecam_index_v2_ablate_attention_step_ss",
]

REQUIRED_OUTPUTS = [
    "config.json",
    "best_model.pt",
    "last_checkpoint.pt",
    "train_log.jsonl",
    "metrics_best.json",
    "metrics_last.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def metrics_status(metrics: dict) -> dict:
    full_by_length = metrics.get("full_sequence_accuracy_by_length", {})
    return {
        "experiment_name": metrics.get("experiment_name"),
        "best_epoch": metrics.get("best_epoch"),
        "best_full_sequence_accuracy": metrics.get("best_full_sequence_accuracy"),
        "best_token_accuracy": metrics.get("best_token_accuracy"),
        "estimated_span": metrics.get("estimated_span"),
        "length_5_accuracy": full_by_length.get("5"),
        "length_6_accuracy": full_by_length.get("6"),
        "order_error": metrics.get("order_error"),
        "repeat_error": metrics.get("repeat_error"),
        "wrong_block": metrics.get("wrong_block"),
    }


def resolve_output_dir(config: dict, output_root: Path, seed: int) -> Path:
    experiment_name = Path(config["output_dir"]).name
    if "_seed" in experiment_name:
        experiment_name = experiment_name.rsplit("_seed", 1)[0]
    return output_root / f"{experiment_name}_seed{seed}"


def verify_outputs(output_dir: Path) -> None:
    missing = [filename for filename in REQUIRED_OUTPUTS if not (output_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing outputs in {output_dir}: {', '.join(missing)}")


def run_one(config_path: Path, output_root: Path, seed: int) -> dict:
    config = load_json(config_path)
    output_dir = resolve_output_dir(config, output_root, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "train.py"),
        "--config",
        str(config_path),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    log_path = output_dir / "run.log"
    error_log_path = output_dir / "error.log"

    with open(log_path, "w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if completed.returncode != 0:
        write_json(
            error_log_path,
            {
                "command": command,
                "returncode": completed.returncode,
                "log_path": str(log_path),
            },
        )
        raise RuntimeError(f"Experiment failed: {config_path.stem} (see {error_log_path})")

    verify_outputs(output_dir)
    metrics = load_json(output_dir / "metrics_best.json")
    print(json.dumps(metrics_status(metrics), ensure_ascii=True))
    return metrics


def main() -> None:
    args = parse_args()
    config_paths = {path.stem: path for path in args.config_dir.glob("*.json")}

    for experiment_name in EXPERIMENT_ORDER:
        if experiment_name not in config_paths:
            raise FileNotFoundError(f"Missing config: {experiment_name}.json")

    for seed in args.seeds:
        for experiment_name in EXPERIMENT_ORDER:
            run_one(config_paths[experiment_name], args.output_root, seed)


if __name__ == "__main__":
    main()
