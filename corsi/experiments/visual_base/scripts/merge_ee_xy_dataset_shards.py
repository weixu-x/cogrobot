"""Merges EE-XY robosuite dataset shards into one dataset manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.experiments.visual_base.scripts.export_robosuite_ee_xy_dataset import (  # noqa: E402
    validate_dataset,
    validate_sequence_overlap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--shards-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--overwrite-existing-samples", action="store_true")
    parser.add_argument("--expected-num-samples", type=int, default=0)
    parser.add_argument("--validate-against-dir", type=str, default="")
    return parser.parse_args()


def rewrite_paths(payload: Any, *, old_sample_dir: str, new_sample_dir: str) -> Any:
    if isinstance(payload, str):
        if payload.startswith(old_sample_dir):
            return new_sample_dir + payload[len(old_sample_dir) :]
        return payload
    if isinstance(payload, list):
        return [rewrite_paths(item, old_sample_dir=old_sample_dir, new_sample_dir=new_sample_dir) for item in payload]
    if isinstance(payload, dict):
        return {
            key: rewrite_paths(value, old_sample_dir=old_sample_dir, new_sample_dir=new_sample_dir)
            for key, value in payload.items()
        }
    return payload


def load_shard_manifests(shards_dir: Path) -> list[dict[str, Any]]:
    manifest_paths = sorted(shards_dir.glob("shard_*_of_*/dataset_manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"No shard dataset_manifest.json files found under {shards_dir}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]


def assert_compatible_shards(shard_manifests: list[dict[str, Any]]) -> None:
    first = shard_manifests[0]
    keys = [
        "dataset_name",
        "split_name",
        "camera_names",
        "sequence_mode",
        "split_seed",
        "xy_normalization",
    ]
    for shard in shard_manifests[1:]:
        for key in keys:
            if shard.get(key) != first.get(key):
                raise ValueError(f"Incompatible shard manifest key '{key}': {shard.get(key)} != {first.get(key)}")


def merge_shards(
    *,
    dataset_root: Path,
    shards_dir: Path,
    output_dir: Path,
    overwrite_existing_samples: bool,
) -> dict[str, Any]:
    shard_manifests = load_shard_manifests(shards_dir)
    assert_compatible_shards(shard_manifests)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    merged_samples: list[dict[str, Any]] = []
    seen_trial_ids: set[str] = set()
    seen_sequences: set[tuple[int, ...]] = set()
    merged_shard_dirs: list[str] = []

    for shard_manifest in shard_manifests:
        shard_output_dir = Path(shard_manifest.get("shard_output_dir") or Path(shard_manifest["samples_dir"]).parent)
        merged_shard_dirs.append(str(shard_output_dir))
        for sample in shard_manifest.get("samples", []):
            trial_id = str(sample["trial_id"])
            if trial_id in seen_trial_ids:
                raise ValueError(f"Duplicate trial id across shards: {trial_id}")
            sequence_key = tuple(int(block_id) for block_id in sample.get("sequence", []))
            if sequence_key in seen_sequences:
                raise ValueError(f"Duplicate sequence across shards: {list(sequence_key)}")
            seen_trial_ids.add(trial_id)
            seen_sequences.add(sequence_key)

            old_sample_dir = Path(sample["sample_dir"])
            new_sample_dir = samples_dir / trial_id
            if new_sample_dir.exists():
                if not overwrite_existing_samples:
                    raise FileExistsError(
                        f"{new_sample_dir} already exists. Re-run with --overwrite-existing-samples to replace it."
                    )
                shutil.rmtree(new_sample_dir)
            shutil.copytree(old_sample_dir, new_sample_dir)

            rewritten_sample = rewrite_paths(
                sample,
                old_sample_dir=str(old_sample_dir),
                new_sample_dir=str(new_sample_dir),
            )
            rewritten_sample["sample_dir"] = str(new_sample_dir)
            rewritten_sample["manifest_path"] = str(new_sample_dir / "manifest.json")

            trial_manifest_path = new_sample_dir / "manifest.json"
            trial_manifest = json.loads(trial_manifest_path.read_text(encoding="utf-8"))
            rewritten_trial_manifest = rewrite_paths(
                trial_manifest,
                old_sample_dir=str(old_sample_dir),
                new_sample_dir=str(new_sample_dir),
            )
            trial_manifest_path.write_text(json.dumps(rewritten_trial_manifest, indent=2), encoding="utf-8")
            merged_samples.append(rewritten_sample)

    merged_samples.sort(key=lambda sample: sample["trial_id"])
    length_counts: dict[str, int] = {}
    for sample in merged_samples:
        length_key = str(int(sample["length"]))
        length_counts[length_key] = length_counts.get(length_key, 0) + 1

    merged_manifest = dict(shard_manifests[0])
    merged_manifest.update(
        {
            "num_samples": len(merged_samples),
            "num_sequences": len(merged_samples),
            "length_counts": length_counts,
            "samples_dir": str(samples_dir),
            "samples": merged_samples,
            "is_shard": False,
            "base_output_dir": str(dataset_root),
            "shard_output_dir": "",
            "merged_from_shards": merged_shard_dirs,
            "shard_info": {
                "merged": True,
                "num_shards": len(shard_manifests),
                "shard_dirs": merged_shard_dirs,
            },
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(merged_manifest, indent=2), encoding="utf-8")
    return merged_manifest


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    shards_dir = Path(args.shards_dir) if args.shards_dir else dataset_root / "shards"
    output_dir = Path(args.output_dir) if args.output_dir else dataset_root
    merged_manifest = merge_shards(
        dataset_root=dataset_root,
        shards_dir=shards_dir,
        output_dir=output_dir,
        overwrite_existing_samples=bool(args.overwrite_existing_samples),
    )
    if args.expected_num_samples and int(merged_manifest["num_samples"]) != int(args.expected_num_samples):
        raise RuntimeError(
            f"Expected {args.expected_num_samples} samples, got {merged_manifest['num_samples']}"
        )
    validation = validate_dataset(output_dir)
    if args.validate_against_dir:
        validation["sequence_overlap"] = validate_sequence_overlap(output_dir, Path(args.validate_against_dir))
        if not validation["sequence_overlap"]["passed"]:
            validation["errors"].append(
                f"Sequence overlap with {args.validate_against_dir}: "
                f"{validation['sequence_overlap']['overlap_examples']}"
            )
            validation["passed"] = False
    summary = {
        "dataset_root": str(output_dir),
        "num_samples": merged_manifest["num_samples"],
        "length_counts": merged_manifest.get("length_counts", {}),
        "validation": validation,
    }
    summary_path = output_dir / "merge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not validation["passed"]:
        raise RuntimeError(f"Merged dataset validation failed: {validation['errors']}")


if __name__ == "__main__":
    main()
