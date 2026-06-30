"""Append entries to reports/baseline_progress_log.md."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append a baseline progress-log entry.")
    parser.add_argument("--log", default="reports/baseline_progress_log.md")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--config-diff", default="none")
    parser.add_argument("--key-results", default="pending")
    parser.add_argument("--checkpoint", default="n/a")
    args = parser.parse_args(argv)

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    path = Path(args.log)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = "\n".join(
        [
            f"## {timestamp} {args.phase}",
            f"- Command: `{args.command}`",
            f"- Config changes: {args.config_diff}",
            f"- Key results: {args.key_results}",
            f"- Checkpoint: `{args.checkpoint}`",
            "",
        ]
    )
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as handle:
        if mode == "w":
            handle.write("# Corsi V2 Baseline Progress Log\n\n")
        handle.write(entry)
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
