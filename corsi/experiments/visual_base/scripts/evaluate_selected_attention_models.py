from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = [
    ("eval_v2_free_running", "corsi_artifacts/visual_base/datasets/freecam_index_v2_val"),
    ("eval_v3_free_running", "corsi_artifacts/visual_base/datasets/freecam_index_v3_val"),
    ("eval_v3_len7_free_running", "corsi_artifacts/visual_base/datasets/freecam_index_v3_val_len7"),
    ("eval_v3_len8_free_running", "corsi_artifacts/visual_base/datasets/freecam_index_v3_val_len8"),
    ("eval_v3_len9_free_running", "corsi_artifacts/visual_base/datasets/freecam_index_v3_val_len9"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="corsi_artifacts/visual_base/training/attention_ablation")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--mode", default="free_running", choices=["free_running", "teacher_forced"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root)

    for model_name in args.models:
        run_dir = result_root / model_name
        checkpoint = run_dir / "best_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        for eval_id, data_root in DATASETS:
            output_dir = run_dir / eval_id
            command = [
                sys.executable,
                "evaluate_visual.py",
                "--checkpoint",
                str(checkpoint),
                "--data-root",
                data_root,
                "--mode",
                args.mode,
                "--save-attention",
                "--output-dir",
                str(output_dir),
            ]
            print("running", model_name, eval_id, flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
