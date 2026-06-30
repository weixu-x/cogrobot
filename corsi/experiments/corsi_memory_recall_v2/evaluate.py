"""Evaluation entry point for Corsi memory-recall V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from corsi.experiments.corsi_memory_recall_v2.analysis import (
    compute_sequence_metrics,
    run_causal_sanity_checks,
)
from corsi.experiments.corsi_memory_recall_v2.train import (
    canonical_manifest_path,
    compute_stage2_loss,
    load_checkpoint,
    load_config,
    make_loader,
    move_to_device,
    pad_sequence_metric_batches,
    resolve_device,
    _build_model,
    _call_model,
)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: Any,
    *,
    device: torch.device,
    eos_token_id: int = 9,
    ignore_index: int = -100,
    loss_weights: Mapping[str, float] | None = None,
    sanity_checks: bool = False,
    split: str | None = None,
) -> dict[str, Any]:
    model.eval()
    predictions = []
    targets = []
    masks = []
    loss_component_totals: dict[str, float] = {}
    loss_component_count = 0
    row_metadata: list[dict[str, Any]] = []
    predicted_xy_rows: list[list[list[float]]] = []
    target_xy_rows: list[list[list[float]]] = []
    sanity: dict[str, Any] | None = None
    for batch in loader:
        batch = move_to_device(batch, device)
        metadata = batch.get("metadata", {})
        eos_values = metadata.get("eos_token_id")
        if torch.is_tensor(eos_values):
            eos_token_id = int(eos_values.flatten()[0].item())
        ignore_index = int(batch.get("ignore_index", ignore_index))
        outputs = _call_model(model, batch, stage=2)
        logits = outputs.get("token_logits", outputs.get("logits"))
        if logits is None:
            raise RuntimeError("V2 evaluator requires model outputs with 'token_logits' or 'logits'.")
        predictions.append(logits.detach().cpu())
        targets.append(batch["targets"]["tokens"].detach().cpu())
        masks.append(batch["targets"]["token_mask"].detach().cpu())
        loss_components = compute_stage2_loss(
            outputs,
            batch,
            weights=loss_weights,
            return_components=True,
        )
        for key, value in loss_components.items():
            if torch.is_tensor(value) and value.ndim == 0:
                loss_component_totals[key] = loss_component_totals.get(key, 0.0) + float(
                    value.detach().cpu().item()
                )
        loss_component_count += 1

        batch_size = int(batch["targets"]["tokens"].shape[0])
        seq_ids = metadata.get("seq_id")
        lengths = metadata.get("length")
        for row_index in range(batch_size):
            row: dict[str, Any] = {}
            if split is not None:
                row["split"] = split
            if isinstance(seq_ids, list) and row_index < len(seq_ids):
                row["episode_id"] = str(seq_ids[row_index])
            if torch.is_tensor(lengths):
                row["length"] = int(lengths.detach().cpu().flatten()[row_index].item())
            row_metadata.append(row)
        coord = outputs.get("coord")
        if torch.is_tensor(coord):
            predicted_xy_rows.extend(coord.detach().cpu().tolist())
        target_xy = batch["targets"].get("target_xy", batch["targets"].get("block_xy"))
        if torch.is_tensor(target_xy):
            target_xy_rows.extend(target_xy.detach().cpu().tolist())
        if sanity_checks and sanity is None:
            sanity = run_causal_sanity_checks(
                model,
                batch,
                eos_token_id=eos_token_id,
                ignore_index=ignore_index,
            )
    padded_predictions, padded_targets, padded_masks = pad_sequence_metric_batches(
        predictions,
        targets,
        masks,
        ignore_index=ignore_index,
        eos_token_id=eos_token_id,
    )
    metrics = compute_sequence_metrics(
        padded_predictions,
        padded_targets,
        padded_masks,
        eos_token_id=eos_token_id,
        ignore_index=ignore_index,
    )
    if loss_component_count:
        metrics.update(
            {
                key: float(value / loss_component_count)
                for key, value in sorted(loss_component_totals.items())
            }
        )
    for row_index, row in enumerate(metrics.get("rows", [])):
        if row_index < len(row_metadata):
            row.update(row_metadata[row_index])
        if row_index < len(predicted_xy_rows):
            row["predicted_xy"] = predicted_xy_rows[row_index][: len(row.get("predicted_tokens", []))]
        if row_index < len(target_xy_rows):
            row["target_xy"] = target_xy_rows[row_index][: int(row.get("target_length", 0))]
    if sanity is not None:
        metrics["causal_sanity_checks"] = sanity
    return metrics


def evaluate_checkpoint(
    *,
    config: Mapping[str, Any],
    checkpoint: str | Path,
    split: str,
    batch_size: int,
    device_name: str,
    sanity_checks: bool = False,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    manifest_path = canonical_manifest_path(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loader = make_loader(manifest_path, split=split, batch_size=batch_size, shuffle=False)
    model = _build_model(config, stage=2, manifest=manifest).to(device)
    payload = load_checkpoint(checkpoint, model=model, map_location=device)
    metrics = evaluate_model(
        model,
        loader,
        device=device,
        eos_token_id=int(manifest.get("eos_token_id", 9)),
        ignore_index=int(manifest.get("ignore_index", -100)),
        loss_weights=dict(config.get("loss_weights", {})) if isinstance(config.get("loss_weights", {}), Mapping) else {},
        sanity_checks=sanity_checks,
        split=split,
    )
    metrics.update(
        {
            "split": split,
            "checkpoint": str(checkpoint),
            "epoch": int(payload.get("epoch", -1)),
            "canonical_fingerprint": manifest.get("canonical_fingerprint"),
        }
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate autonomous Corsi memory-recall V2 checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sanity-checks", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    metrics = evaluate_checkpoint(
        config=load_config(args.config),
        checkpoint=args.checkpoint,
        split=args.split,
        batch_size=int(args.batch_size),
        device_name=args.device,
        sanity_checks=bool(args.sanity_checks),
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
