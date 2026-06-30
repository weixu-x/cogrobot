"""Phase 0 dataset self-checks for the expanded Corsi V2 baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import permutations
from pathlib import Path
from typing import Any, Mapping, Sequence


SPLITS = ("train", "val", "test")
TARGET_COUNTS = {
    "train": {"2": 72, "3": 300, "4": 600, "5": 800, "6": 800, "7": 800, "8": 800, "9": 800},
    "val": {"2": 10, "3": 30, "4": 30, "5": 30, "6": 30, "7": 30, "8": 30, "9": 30},
    "test": {"2": 20, "3": 50, "4": 50, "5": 50, "6": 50, "7": 50, "8": 50, "9": 50},
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_counts_from_config(config: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    raw_counts = config.get("split_length_counts") or config.get("expected_split_length_counts") or TARGET_COUNTS
    return {
        split: {str(length): int(count) for length, count in dict(raw_counts[split]).items()}
        for split in SPLITS
    }


def count_by_split_length(samples: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for sample in samples:
        split = str(sample.get("split", ""))
        if split in counters:
            counters[split][str(int(sample["length"]))] += 1
    return {
        split: {length: int(counters[split][length]) for length in sorted(counters[split], key=int)}
        for split in SPLITS
    }


def order_sets_by_length_split(samples: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, set[tuple[int, ...]]]]:
    result: dict[int, dict[str, set[tuple[int, ...]]]] = {}
    for sample in samples:
        split = str(sample.get("split", ""))
        length = int(sample["length"])
        result.setdefault(length, {name: set() for name in SPLITS})
        if split in SPLITS:
            result[length][split].add(tuple(int(value) for value in sample["block_order"]))
    return result


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def run_checks(
    *,
    config_path: Path,
    raw_manifest_path: Path,
    canonical_manifest_path: Path | None,
) -> dict[str, Any]:
    config = load_json(config_path)
    manifest = load_json(raw_manifest_path)
    samples = list(manifest.get("samples") or [])
    expected = split_counts_from_config(config)
    actual = count_by_split_length(samples)
    failures: list[str] = []
    if actual != expected:
        failures.append(f"split x length counts mismatch: actual={actual} expected={expected}")

    failed_episodes = list(manifest.get("failed_episodes") or [])
    skipped_episodes = list(manifest.get("skipped_episodes") or [])
    if failed_episodes or skipped_episodes:
        failures.append(f"raw manifest has failed/skipped episodes: failed={len(failed_episodes)} skipped={len(skipped_episodes)}")

    internal_repeat_count = 0
    for sample in samples:
        order = [int(value) for value in sample["block_order"]]
        if len(order) != len(set(order)):
            internal_repeat_count += 1
    if internal_repeat_count:
        failures.append(f"{internal_repeat_count} samples contain repeated blocks inside the sequence")

    order_sets = order_sets_by_length_split(samples)
    all_len2 = set(permutations(range(9), 2))
    train_len2 = order_sets.get(2, {}).get("train", set())
    len2_train_missing = sorted(all_len2 - train_len2)
    len2_train_extra = sorted(train_len2 - all_len2)
    if len(train_len2) != 72 or len2_train_missing or len2_train_extra:
        failures.append(
            f"length-2 train is not the complete 72 ordered pairs: count={len(train_len2)} missing={len(len2_train_missing)} extra={len(len2_train_extra)}"
        )

    overlap_rows: list[dict[str, Any]] = []
    for length in sorted(order_sets):
        split_sets = order_sets[length]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = split_sets[left] & split_sets[right]
            row = {"length": length, "pair": f"{left}/{right}", "overlap_count": len(overlap)}
            overlap_rows.append(row)
            if length >= 4 and overlap:
                failures.append(f"length {length} has {len(overlap)} overlapping orders for {left}/{right}")

    canonical_summary: dict[str, Any] | None = None
    if canonical_manifest_path is not None and canonical_manifest_path.exists():
        canonical = load_json(canonical_manifest_path)
        canonical_summary = {
            "manifest_path": str(canonical_manifest_path),
            "canonical_fingerprint": canonical.get("canonical_fingerprint"),
            "split_length_counts": canonical.get("split_length_counts"),
            "sample_count": len(canonical.get("samples", [])),
        }
        if canonical.get("split_length_counts") != expected:
            failures.append("canonical split_length_counts do not match Phase 0 targets")

    return {
        "status": "ok" if not failures else "failed",
        "config": str(config_path),
        "raw_manifest_path": str(raw_manifest_path),
        "raw_sample_count": len(samples),
        "expected_split_length_counts": expected,
        "actual_split_length_counts": actual,
        "length2_train_complete_count": len(train_len2),
        "length2_train_missing_count": len(len2_train_missing),
        "internal_repeat_count": internal_repeat_count,
        "cross_split_overlaps": overlap_rows,
        "baseline_dataset_design_record_found": Path("baseline_dataset_design_record.md").exists(),
        "canonical": canonical_summary,
        "failures": failures,
    }


def write_markdown(result: Mapping[str, Any], output_md: Path) -> None:
    expected = result["expected_split_length_counts"]
    actual = result["actual_split_length_counts"]
    count_rows = []
    for split in SPLITS:
        for length in sorted(expected[split], key=int):
            count_rows.append([split, length, actual.get(split, {}).get(length, 0), expected[split][length]])
    overlap_rows = [
        [row["length"], row["pair"], row["overlap_count"]]
        for row in result["cross_split_overlaps"]
    ]
    checks = [
        ["split x length counts", "PASS" if actual == expected else "FAIL"],
        ["length-2 train complete 72", "PASS" if result["length2_train_complete_count"] == 72 and result["length2_train_missing_count"] == 0 else "FAIL"],
        ["L>=4 cross-split overlap zero", "PASS" if not any(row["length"] >= 4 and row["overlap_count"] for row in result["cross_split_overlaps"]) else "FAIL"],
        ["internal block repeats", "PASS" if result["internal_repeat_count"] == 0 else "FAIL"],
        ["raw failed/skipped episodes", "PASS" if not result["failures"] or not any("failed/skipped" in item for item in result["failures"]) else "FAIL"],
    ]
    lines = [
        "# Corsi V2 Expanded800 Dataset Manifest",
        "",
        f"- Status: `{result['status']}`",
        f"- Raw manifest: `{result['raw_manifest_path']}`",
        f"- Raw samples: `{result['raw_sample_count']}`",
        f"- `baseline_dataset_design_record.md` found in checkout: `{result['baseline_dataset_design_record_found']}`",
        "",
        "## Self-Checks",
        "",
        table(["check", "result"], checks),
        "",
        "## Split x Length Counts",
        "",
        table(["split", "length", "actual", "target"], count_rows),
        "",
        "## Cross-Split Block-Order Overlap",
        "",
        table(["length", "split pair", "overlap count"], overlap_rows),
    ]
    if result.get("canonical"):
        lines.extend(
            [
                "",
                "## Canonical Dataset",
                "",
                f"- Manifest: `{result['canonical']['manifest_path']}`",
                f"- Samples: `{result['canonical']['sample_count']}`",
                f"- Fingerprint: `{result['canonical']['canonical_fingerprint']}`",
            ]
        )
    if result["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in result["failures"])
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 0 self-checks for the expanded Corsi V2 dataset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--canonical-manifest", default="")
    parser.add_argument("--output-json", default="reports/dataset_manifest.json")
    parser.add_argument("--output-md", default="reports/dataset_manifest.md")
    args = parser.parse_args(argv)
    result = run_checks(
        config_path=Path(args.config),
        raw_manifest_path=Path(args.raw_manifest),
        canonical_manifest_path=Path(args.canonical_manifest) if args.canonical_manifest else None,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result, Path(args.output_md))
    print(json.dumps({"status": result["status"], "failures": result["failures"]}, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
