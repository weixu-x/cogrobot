"""Run the mandatory multi-seed Corsi motion baseline suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from corsi.experiments.corsi_motion_baseline.canonicalize import load_config
from corsi.experiments.corsi_motion_baseline.train import train_one_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run multi-seed Corsi motion training suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-type", choices=["visual_joint", "joint_only"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config["device"] = args.device
    seeds = list(args.seeds) if args.seeds is not None else list(config.get("seeds", [0, 5, 10, 15, 20]))
    summaries = []
    for seed in seeds:
        run_name = f"{args.model_type}_seed{seed}{args.run_suffix}"
        run_dir = Path(str(config["output_root"])) / run_name
        summary = train_one_run(
            config,
            model_type=args.model_type,
            seed=int(seed),
            run_dir=run_dir,
            max_epochs_override=args.max_epochs,
            resume=bool(args.resume),
        )
        summaries.append(summary)
        suite_path = Path(str(config["output_root"])) / f"{args.model_type}_suite_summary.json"
        suite_path.parent.mkdir(parents=True, exist_ok=True)
        suite_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
