"""Write compact split-specific summaries from full binding diagnostic JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def extract(result: Mapping[str, Any], split: str) -> dict[str, Any]:
    behavior = result["original"]["behavior"][split]
    final_hc = result["original"]["final_probes"]["hc"][split]
    serial = behavior.get("behavior_sanity", {}).get("serial_position_shape", {})
    return {
        "source_diagnostic": result.get("checkpoint"),
        "config": result.get("config"),
        "checkpoint": result.get("checkpoint"),
        "checkpoint_epoch": result.get("checkpoint_epoch"),
        "split": split,
        "full_sequence_accuracy": behavior["full_sequence_accuracy"],
        "token_accuracy": behavior["token_accuracy"],
        "duplicate_sequence_rate": behavior["duplicate_sequence_rate"],
        "u_shape_score": serial.get("u_shape_score", 0.0),
        "final_order_token_accuracy": final_hc["order"]["block_token_order_accuracy"],
        "final_order_known_length_exact": final_hc["order"]["known_length_exact_sequence_accuracy"],
        "final_length_accuracy": final_hc["length"]["accuracy"],
        "per_length_metrics": behavior["per_length_metrics"],
        "behavior_sanity": behavior.get("behavior_sanity", {}),
    }


def write_md(summary: Mapping[str, Any], output_md: Path) -> None:
    rows = []
    for length, item in sorted(summary["per_length_metrics"].items(), key=lambda kv: int(kv[0])):
        rows.append(
            [
                length,
                item["count"],
                fmt(item["full_sequence_accuracy"]),
                fmt(item["token_accuracy"]),
                fmt(item["duplicate_sequence_rate"]),
            ]
        )
    lines = [
        f"# Corsi V2 Binding Diagnostics - {summary['split']}",
        "",
        f"- Config: `{summary['config']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Epoch: `{summary['checkpoint_epoch']}`",
        f"- Full exact: `{fmt(summary['full_sequence_accuracy'])}`",
        f"- Token: `{fmt(summary['token_accuracy'])}`",
        f"- Duplicate sequence rate: `{fmt(summary['duplicate_sequence_rate'])}`",
        f"- U-score: `{fmt(summary['u_shape_score'])}`",
        f"- Final-order probe token: `{fmt(summary['final_order_token_accuracy'])}`",
        f"- Final-order known-length exact: `{fmt(summary['final_order_known_length_exact'])}`",
        "",
        "## Per-Length",
        "",
        table(["length", "count", "exact", "token", "duplicate"], rows),
        "",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize one split from full Corsi V2 binding diagnostics.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    result = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    summary = extract(result, args.split)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_md(summary, Path(args.output_md))
    print(json.dumps({"split": args.split, "output_json": str(output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
