from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.data import RobosuiteVisualCorsiDataset, collate_visual_batch
from corsi.models.lstm_visual import VisualSeq2SeqLSTM
from corsi.training.device import resolve_torch_device
from corsi.training.train_visual import (
    build_model_config,
    heatmap_entropy,
    load_checkpoint,
    move_batch_to_device,
    normalize_heatmap_image,
    TrainVisualConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-per-category", type=int, default=8)
    return parser.parse_args()


def config_from_payload(payload: Dict[str, object]) -> TrainVisualConfig:
    args = payload.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    data = {**TrainVisualConfig().__dict__, **args}
    return TrainVisualConfig(**data)


def save_step(
    *,
    category_dir: Path,
    count: int,
    batch,
    batch_index: int,
    step_index: int,
    target_index: int,
    pred_index: int,
    spatial_error: float,
    entropy: float,
    target_heatmap,
    pred_heatmap,
) -> None:
    sample_dir = category_dir / f"sample_{count:03d}_{batch['trial_ids'][batch_index]}_step_{step_index + 1:02d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    frame = batch["frames"][batch_index, step_index].detach().cpu().permute(1, 2, 0).numpy()
    imageio.imwrite(sample_dir / "input_frame.png", (frame * 255.0).clip(0, 255).astype(np.uint8))
    imageio.imwrite(sample_dir / "target_heatmap.png", normalize_heatmap_image(target_heatmap))
    imageio.imwrite(sample_dir / "predicted_heatmap.png", normalize_heatmap_image(pred_heatmap))
    with open(sample_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "trial_id": batch["trial_ids"][batch_index],
                "recall_position": step_index + 1,
                "target_index": int(target_index),
                "predicted_nearest_index": int(pred_index),
                "spatial_error": float(spatial_error),
                "entropy": float(entropy),
            },
            handle,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    payload = load_checkpoint(checkpoint_path)
    config = config_from_payload(payload)
    data_root = args.data_root or config.val_dataset_root or config.dataset_root
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "best_heatmap_visualizations"
    device, _ = resolve_torch_device(args.device)

    model = VisualSeq2SeqLSTM(build_model_config(config)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    dataset = RobosuiteVisualCorsiDataset(
        data_root,
        camera_name=config.camera_name,
        include_reset_frame=config.include_reset_frame,
        heatmap_size=config.heatmap_size,
        heatmap_sigma=config.heatmap_sigma,
        heatmap_normalize=config.heatmap_normalize,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_visual_batch)

    categories = {
        "correct_prediction": 0,
        "wrong_near_target": 0,
        "wrong_far_from_target": 0,
    }
    for category in categories:
        (output_dir / category).mkdir(parents=True, exist_ok=True)

    block_xy = model.block_heatmap_xy.detach().cpu()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model.greedy_decode(
                frames=batch["frames"],
                frame_lengths=batch["frame_lengths"],
                target_lengths=batch["target_lengths"],
                max_steps=batch["targets"].size(1),
            )
            entropy = heatmap_entropy(outputs["heatmap_logits"]).detach().cpu()
            predictions = outputs["predictions"].detach().cpu()
            pred_xy = outputs["heatmap_xy"].detach().cpu()
            targets = batch["targets"].detach().cpu()
            target_heatmaps = batch["target_heatmaps"].detach().cpu()
            pred_heatmaps = F.softmax(outputs["heatmap_logits"].detach().cpu().squeeze(2).flatten(start_dim=-2), dim=-1)
            pred_heatmaps = pred_heatmaps.reshape_as(target_heatmaps)
            target_xy = block_xy[targets.clamp(0, block_xy.size(0) - 1)]
            spatial_errors = torch.linalg.norm(pred_xy.float() - target_xy.float(), dim=-1)

            for batch_index in range(targets.size(0)):
                length = int(batch["target_lengths"][batch_index].item())
                for step_index in range(length):
                    target_index = int(targets[batch_index, step_index].item())
                    pred_index = int(predictions[batch_index, step_index].item())
                    spatial_error = float(spatial_errors[batch_index, step_index].item())
                    if pred_index == target_index:
                        category = "correct_prediction"
                    elif spatial_error <= 4.0:
                        category = "wrong_near_target"
                    elif spatial_error >= 8.0:
                        category = "wrong_far_from_target"
                    else:
                        continue
                    if categories[category] >= args.max_per_category:
                        continue
                    save_step(
                        category_dir=output_dir / category,
                        count=categories[category],
                        batch=batch,
                        batch_index=batch_index,
                        step_index=step_index,
                        target_index=target_index,
                        pred_index=pred_index,
                        spatial_error=spatial_error,
                        entropy=float(entropy[batch_index, step_index].item()),
                        target_heatmap=target_heatmaps[batch_index, step_index].numpy(),
                        pred_heatmap=pred_heatmaps[batch_index, step_index].numpy(),
                    )
                    categories[category] += 1
            if all(count >= args.max_per_category for count in categories.values()):
                break

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump({"checkpoint": str(checkpoint_path), "data_root": data_root, "counts": categories}, handle, indent=2)
    print(json.dumps({"output_dir": str(output_dir), "counts": categories}))


if __name__ == "__main__":
    main()
