"""Merges multiple robosuite visual dataset shards into one manifest-driven dataset root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--split-name", type=str, required=True)
    parser.add_argument("--shard-dirs", type=str, nargs="+", required=True)
    return parser.parse_args()


def load_manifest(shard_dir: Path) -> dict[str, Any]:
    manifest_path = shard_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset_manifest.json in shard dir: {shard_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = [Path(path) for path in args.shard_dirs]
    shard_manifests = [load_manifest(path) for path in shard_dirs]
    if not shard_manifests:
        raise ValueError("At least one shard dir is required")

    first_manifest = shard_manifests[0]
    camera_names = list(first_manifest.get("camera_names", []))
    sample_structure = list(first_manifest.get("sample_structure", []))
    export_params = dict(first_manifest.get("export_params", {}))

    all_samples: list[dict[str, Any]] = []
    seq_min = None
    seq_max = None
    shard_payloads: list[dict[str, Any]] = []

    for shard_dir, manifest in zip(shard_dirs, shard_manifests):
        if list(manifest.get("camera_names", [])) != camera_names:
            raise ValueError(f"Camera mismatch in shard {shard_dir}")
        shard_payloads.append(
            {
                "shard_dir": str(shard_dir),
                "num_samples": int(manifest.get("num_samples", 0)),
                "dataset_name": manifest.get("dataset_name", ""),
                "split_name": manifest.get("split_name", ""),
                "shard_info": dict(manifest.get("shard_info", {})),
            }
        )
        all_samples.extend(list(manifest.get("samples", [])))
        shard_range = manifest.get("sequence_length_range", {})
        shard_min = shard_range.get("min")
        shard_max = shard_range.get("max")
        seq_min = shard_min if seq_min is None else min(seq_min, shard_min)
        seq_max = shard_max if seq_max is None else max(seq_max, shard_max)

    merged_manifest = {
        "dataset_name": args.dataset_name,
        "split_name": args.split_name,
        "camera_names": camera_names,
        "num_samples": len(all_samples),
        "samples_dir": "manifest_only_merged_dataset",
        "sequence_length_range": {
            "min": seq_min,
            "max": seq_max,
        },
        "sample_structure": sample_structure,
        "export_params": export_params,
        "merged_from_shards": shard_payloads,
        "samples": all_samples,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(merged_manifest, indent=2), encoding="utf-8")
    print(json.dumps(merged_manifest, indent=2))
    print(f"Merged dataset manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
