from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import imageio.v2 as imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.heatmaps import nearest_block_decode, standard_block_heatmap_xy
from corsi.training.train_visual import TrainVisualConfig, build_datasets, load_config_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--samples-per-split", type=int, default=20)
    return parser.parse_args()


def load_config(config_path: str) -> TrainVisualConfig:
    overrides = load_config_overrides(config_path)
    data = {**TrainVisualConfig().__dict__, **overrides}
    return TrainVisualConfig(**data)


def normalize_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.copy()
    values = np.asarray(image, dtype=np.float32)
    if values.max() <= 1.0:
        values = values * 255.0
    return values.clip(0, 255).astype(np.uint8)


def draw_square(image: np.ndarray, x: int, y: int, color: tuple[int, int, int], radius: int = 2) -> None:
    height, width = image.shape[:2]
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    image[y0:y1, x0:x1] = np.asarray(color, dtype=np.uint8)


def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray, block_xy: np.ndarray) -> np.ndarray:
    image = normalize_image(frame)
    height, width = image.shape[:2]
    scale_y = max(1, int(np.ceil(height / heatmap.shape[0])))
    scale_x = max(1, int(np.ceil(width / heatmap.shape[1])))
    heatmap_up = np.repeat(np.repeat(heatmap, scale_y, axis=0), scale_x, axis=1)[:height, :width]
    if float(heatmap_up.max()) > float(heatmap_up.min()):
        heatmap_up = (heatmap_up - heatmap_up.min()) / (heatmap_up.max() - heatmap_up.min())
    color = np.zeros_like(image)
    color[..., 0] = (heatmap_up * 255.0).astype(np.uint8)
    overlay = (0.65 * image.astype(np.float32) + 0.35 * color.astype(np.float32)).clip(0, 255).astype(np.uint8)
    for center_x, center_y in block_xy:
        draw_square(
            overlay,
            int(round(float(center_x) / (heatmap.shape[1] - 1) * (width - 1))),
            int(round(float(center_y) / (heatmap.shape[0] - 1) * (height - 1))),
            (0, 255, 0),
            radius=2,
        )
    return overlay


def decode_heatmap_target(heatmap: np.ndarray, block_xy: np.ndarray) -> tuple[list[int], int]:
    height, width = heatmap.shape
    flat_index = int(np.asarray(heatmap).reshape(-1).argmax())
    y, x = divmod(flat_index, width)
    xy = torch.tensor([[[x, y]]], dtype=torch.float32)
    decoded, _ = nearest_block_decode(xy, block_xy=torch.tensor(block_xy, dtype=torch.float32))
    return [int(x), int(y)], int(decoded[0, 0].item())


def inspect_split(name: str, dataset, *, output_dir: Path, max_samples: int) -> Dict[str, Any]:
    split_dir = output_dir / name
    split_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    correct = 0
    rows = []
    heatmap_size = int(getattr(dataset.dataset if hasattr(dataset, "dataset") else dataset, "heatmap_size", 32))
    block_xy = standard_block_heatmap_xy(heatmap_size)

    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        frames = sample["frames"]
        targets = sample["targets"]
        target_heatmaps = sample["target_heatmaps"]
        for step_index, target_index in enumerate(targets):
            argmax_xy, decoded_index = decode_heatmap_target(target_heatmaps[step_index], block_xy)
            is_correct = decoded_index == int(target_index)
            total += 1
            correct += int(is_correct)
            row = {
                "split": name,
                "sample_index": sample_index,
                "trial_id": sample["trial_id"],
                "recall_position": step_index + 1,
                "target_index": int(target_index),
                "target_heatmap_argmax": argmax_xy,
                "nearest_block_from_target_argmax": decoded_index,
                "correct": is_correct,
            }
            rows.append(row)
            if sample_index < max_samples:
                image = overlay_heatmap(frames[step_index], target_heatmaps[step_index], block_xy)
                imageio.imwrite(split_dir / f"sample_{sample_index:03d}_step_{step_index + 1:02d}.png", image)
        if sample_index < max_samples:
            with open(split_dir / f"sample_{sample_index:03d}_metadata.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "trial_id": sample["trial_id"],
                        "target_sequence": [int(value) for value in targets],
                        "rows": [row for row in rows if row["sample_index"] == sample_index],
                    },
                    handle,
                    indent=2,
                )

    return {
        "split": name,
        "num_tokens": total,
        "target_decode_acc": float(correct / total) if total else 0.0,
        "rows": rows,
        "visualization_dir": str(split_dir),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.output_dir) / "target_sanity"
    train_dataset, val_dataset = build_datasets(config)
    train_summary = inspect_split("train", train_dataset, output_dir=output_dir, max_samples=args.samples_per_split)
    val_summary = inspect_split("val", val_dataset, output_dir=output_dir, max_samples=args.samples_per_split)
    summary = {
        "config": args.config,
        "target_decode_acc": min(train_summary["target_decode_acc"], val_summary["target_decode_acc"]),
        "train": {key: value for key, value in train_summary.items() if key != "rows"},
        "val": {key: value for key, value in val_summary.items() if key != "rows"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
