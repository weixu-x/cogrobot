"""Build the Phase 3 D_mem sweep report from split diagnostic summaries."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


RUN_RE = re.compile(r"dmem(?P<dmem>\d+)_seed(?P<seed>\d+)")


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{fmt(mean(values))} +/- {fmt(pstdev(values))}"


def breakpoint(per_length: dict[str, float]) -> str:
    for length in sorted(per_length, key=int):
        if float(per_length[length]) < 0.80:
            return str(length)
    return "DMEM_OVERCAPACITY"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def load_rows(pattern: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path().glob(pattern)):
        match = RUN_RE.search(path.name)
        if not match:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_length_exact = {
            str(length): float(item["full_sequence_accuracy"])
            for length, item in payload["per_length_metrics"].items()
        }
        rows.append(
            {
                "path": str(path),
                "dmem": int(match.group("dmem")),
                "seed": int(match.group("seed")),
                "full": float(payload["full_sequence_accuracy"]),
                "token": float(payload["token_accuracy"]),
                "duplicate": float(payload["duplicate_sequence_rate"]),
                "u_score": float(payload["u_shape_score"]),
                "final_order": float(payload["final_order_token_accuracy"]),
                "final_order_known_length_exact": float(payload["final_order_known_length_exact"]),
                "per_length_exact": per_length_exact,
                "breakpoint": breakpoint(per_length_exact),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dmem: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dmem[int(row["dmem"])].append(row)
    groups = []
    for dmem, group in sorted(by_dmem.items()):
        lengths = sorted({length for row in group for length in row["per_length_exact"]}, key=int)
        mean_curve = {
            length: mean([row["per_length_exact"][length] for row in group if length in row["per_length_exact"]])
            for length in lengths
        }
        bp = breakpoint(mean_curve)
        bp_distance = 99.0 if bp == "DMEM_OVERCAPACITY" else abs(float(bp) - 4.5)
        groups.append(
            {
                "dmem": dmem,
                "seeds": sorted(int(row["seed"]) for row in group),
                "mean_curve": mean_curve,
                "breakpoint": bp,
                "breakpoint_sort_distance": bp_distance,
                "full_mean_std": mean_std([row["full"] for row in group]),
                "token_mean_std": mean_std([row["token"] for row in group]),
                "duplicate_mean_std": mean_std([row["duplicate"] for row in group]),
                "u_score_mean_std": mean_std([row["u_score"] for row in group]),
                "final_order_mean_std": mean_std([row["final_order"] for row in group]),
            }
        )
    groups.sort(key=lambda item: (item["breakpoint_sort_distance"], int(item["dmem"])))
    return {"runs": rows, "groups": groups}


def write_md(summary: dict[str, Any], output_md: Path) -> None:
    group_rows = [
        [
            group["dmem"],
            ",".join(str(seed) for seed in group["seeds"]),
            group["breakpoint"],
            group["u_score_mean_std"],
            group["final_order_mean_std"],
            group["token_mean_std"],
            group["duplicate_mean_std"],
        ]
        for group in summary["groups"]
    ]
    run_rows = [
        [
            row["dmem"],
            row["seed"],
            row["breakpoint"],
            fmt(row["full"]),
            fmt(row["token"]),
            fmt(row["duplicate"]),
            fmt(row["u_score"]),
            fmt(row["final_order"]),
        ]
        for row in sorted(summary["runs"], key=lambda item: (item["dmem"], item["seed"]))
    ]
    lines = [
        "# Corsi V2 Expanded800 D_mem Sweep Summary",
        "",
        "Sorted by breakpoint closeness to length 4/5, not by maximum accuracy.",
        "",
        "## D_mem Ranking",
        "",
        table(
            ["D_mem", "seeds", "mean breakpoint", "U-score mean+/-std", "final-order mean+/-std", "token mean+/-std", "duplicate mean+/-std"],
            group_rows,
        ),
        "",
        "## Runs",
        "",
        table(["D_mem", "seed", "breakpoint", "full", "token", "duplicate", "U-score", "final-order"], run_rows),
        "",
        "## Mean Per-Length Exact Curves",
        "",
    ]
    for group in summary["groups"]:
        curve_rows = [[length, fmt(value)] for length, value in sorted(group["mean_curve"].items(), key=lambda item: int(item[0]))]
        lines.extend([f"### D_mem {group['dmem']}", "", table(["length", "mean exact"], curve_rows), ""])
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write D_mem sweep summary from diagnostic split summaries.")
    parser.add_argument("--pattern", default="reports/expanded800_dmem*_seed*_test_diag.json")
    parser.add_argument("--output-json", default="reports/dmem_sweep_summary.json")
    parser.add_argument("--output-md", default="reports/dmem_sweep_summary.md")
    args = parser.parse_args(argv)
    rows = load_rows(args.pattern)
    if not rows:
        raise SystemExit(f"no diagnostic rows matched pattern {args.pattern!r}")
    summary = summarize(rows)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_md(summary, Path(args.output_md))
    print(json.dumps({"runs": len(rows), "output_md": args.output_md}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
