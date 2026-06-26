"""Run small Corsi memory-recall V2 training suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from corsi.experiments.corsi_memory_recall_v2.train import load_config, train_one_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Corsi memory-recall V2 Stage 1/2 training suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--overfit-episodes", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-full-training", action="store_true")
    parser.add_argument("--warm-start-stage1-checkpoint", default="")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config["device"] = args.device
    seeds = list(args.seeds) if args.seeds is not None else list(config.get("seeds", [0]))
    output_root = Path(args.output_root or config.get("output_root", "corsi_artifacts/memory_recall_v2/runs"))
    summaries = []
    for seed in seeds:
        run_name = f"stage{args.stage}_seed{seed}{args.run_suffix}"
        summary = train_one_run(
            config,
            stage=int(args.stage),
            seed=int(seed),
            run_dir=output_root / run_name,
            max_epochs=args.max_epochs,
            overfit_episodes=int(args.overfit_episodes),
            resume=bool(args.resume),
            allow_full_training=bool(args.allow_full_training),
            dry_run=bool(args.dry_run),
            warm_start_stage1_checkpoint=args.warm_start_stage1_checkpoint,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    suite_path = output_root / f"stage{args.stage}_suite_summary.json"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
