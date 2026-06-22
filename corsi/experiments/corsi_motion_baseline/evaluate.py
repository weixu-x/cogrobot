"""Evaluation entry point for Corsi 7-joint motion baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from corsi.experiments.corsi_motion_baseline.canonicalize import load_config
from corsi.experiments.corsi_motion_baseline.dataset import (
    CorsiMotionCanonicalDataset,
    collate_motion_prediction_batch,
)
from corsi.experiments.corsi_motion_baseline.model import build_model
from corsi.experiments.corsi_motion_baseline.train import evaluate_model, resolve_device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Corsi 7-joint motion baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model-type", choices=["visual_joint", "joint_only", "persistence"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--mode", choices=["normal", "shuffled_vision", "zero_vision"], default="normal")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    manifest_path = Path(str(config["canonical_root"])) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = CorsiMotionCanonicalDataset(
        manifest_path,
        split=args.split,
        load_images=(args.model_type == "visual_joint"),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_motion_prediction_batch,
    )
    device = resolve_device(args.device)
    model = None
    if args.model_type != "persistence":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for learned models")
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = build_model(args.model_type, joint_dim=int(manifest["joint_dim"]), hidden_dim=128).to(device)
        model.load_state_dict(payload["model_state_dict"])
    metrics = evaluate_model(
        model,
        loader,
        device=device,
        manifest=manifest,
        mode=args.mode,
        model_type="visual_joint" if args.model_type == "visual_joint" else "joint_only",
    )
    metrics.update(
        {
            "model_type": args.model_type,
            "split": args.split,
            "checkpoint": args.checkpoint,
            "canonical_fingerprint": manifest["canonical_fingerprint"],
        }
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
