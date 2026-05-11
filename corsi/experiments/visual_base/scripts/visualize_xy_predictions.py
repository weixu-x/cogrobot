from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.data import RobosuiteVisualCorsiDataset, collate_visual_batch
from corsi.heatmaps import nearest_block_decode
from corsi.models.lstm_visual import VisualSeq2SeqLSTM
from corsi.training.device import resolve_torch_device
from corsi.training.train_visual import (
    TrainVisualConfig,
    block_norm_xy_from_mapping,
    build_model_config,
    load_checkpoint,
    move_batch_to_device,
    table_delta_scale_from_xy_normalization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--camera-name", default="")
    return parser.parse_args()


def config_from_payload(payload: Dict[str, object]) -> TrainVisualConfig:
    args = payload.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    data = {**TrainVisualConfig().__dict__, **args}
    return TrainVisualConfig(**data)


def resolve_manifest_path(path_text: str, dataset_root: Path) -> Path:
    path = Path(path_text)
    if path.exists() or path.is_absolute():
        return path
    candidate = dataset_root / path_text
    if candidate.exists():
        return candidate
    if "corsi_artifacts" in path.parts:
        artifact_index = path.parts.index("corsi_artifacts")
        candidate = REPO_ROOT.joinpath(*path.parts[artifact_index:])
        if candidate.exists():
            return candidate
    return path


def collect_block_pixel_map(dataset: RobosuiteVisualCorsiDataset, camera_name: str) -> Dict[int, tuple[float, float]]:
    points: dict[int, list[tuple[float, float]]] = {}
    for sample in dataset.samples:
        step_metadata = sample.get("step_metadata")
        if not isinstance(step_metadata, list):
            manifest_path = resolve_manifest_path(str(sample.get("manifest_path", "")), dataset.dataset_root)
            if not manifest_path.exists():
                continue
            step_metadata = json.loads(manifest_path.read_text(encoding="utf-8")).get("step_metadata", [])
        for metadata in step_metadata:
            image_points = metadata.get("image_points", {}).get(camera_name, {})
            if "target_block_pixel" not in image_points:
                continue
            block_index = int(metadata["block_index"])
            pixel = image_points["target_block_pixel"]
            points.setdefault(block_index, []).append((float(pixel[0]), float(pixel[1])))
    return {
        block_index: (
            float(np.mean([point[0] for point in block_points])),
            float(np.mean([point[1] for point in block_points])),
        )
        for block_index, block_points in points.items()
    }


def fit_norm_to_pixel_affine(
    block_xy_norm: torch.Tensor,
    block_pixel_map: Dict[int, tuple[float, float]],
) -> np.ndarray:
    rows = []
    targets = []
    for block_index, pixel in sorted(block_pixel_map.items()):
        if block_index < 0 or block_index >= block_xy_norm.size(0):
            continue
        x, y = block_xy_norm[block_index].tolist()
        rows.append([float(x), float(y), 1.0])
        targets.append([float(pixel[0]), float(pixel[1])])
    if len(rows) < 3:
        raise ValueError("Need at least three block pixel correspondences to fit XY overlay affine")
    affine, _, _, _ = np.linalg.lstsq(np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64), rcond=None)
    return affine


def xy_to_pixel(norm_xy: Any, affine: np.ndarray) -> tuple[int, int]:
    vector = np.asarray([float(norm_xy[0]), float(norm_xy[1]), 1.0], dtype=np.float64)
    pixel = vector @ affine
    return int(round(float(pixel[0]))), int(round(float(pixel[1])))


def draw_point(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    label: str,
) -> None:
    x, y = point
    radius = 7
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=(255, 255, 255), width=2)
    draw.line((x - 11, y, x + 11, y), fill=(255, 255, 255), width=1)
    draw.line((x, y - 11, x, y + 11), fill=(255, 255, 255), width=1)
    font = ImageFont.load_default()
    bbox = draw.textbbox((x + 10, y - 14), label, font=font)
    draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=(0, 0, 0))
    draw.text((x + 10, y - 14), label, fill=(255, 255, 255), font=font)


def metadata_pixel(
    metadata: dict,
    camera_name: str,
    key: str,
    fallback_xy: Any,
    affine: np.ndarray,
) -> tuple[int, int]:
    image_points = metadata.get("image_points", {}).get(camera_name, {})
    if key in image_points:
        return tuple(int(value) for value in image_points[key])
    return xy_to_pixel(fallback_xy, affine)


def save_overlay(
    *,
    output_path: Path,
    frame: torch.Tensor,
    metadata: dict,
    camera_name: str,
    target_xy: torch.Tensor,
    pred_xy: torch.Tensor,
    target_block_xy: torch.Tensor,
    target_block_index: int,
    nearest_block_index: int,
    block_pixel_map: Dict[int, tuple[float, float]],
    affine: np.ndarray,
    xy_error_norm: float,
    xy_error_table: float,
) -> None:
    image_array = (frame.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image_array).convert("RGB")
    draw = ImageDraw.Draw(image)
    target_ee_pixel = metadata_pixel(metadata, camera_name, "ee_pixel", target_xy.tolist(), affine)
    target_block_pixel = metadata_pixel(
        metadata,
        camera_name,
        "target_block_pixel",
        target_block_xy.tolist(),
        affine,
    )
    pred_pixel = xy_to_pixel(pred_xy.tolist(), affine)
    nearest_block_pixel = (
        tuple(int(round(value)) for value in block_pixel_map[nearest_block_index])
        if nearest_block_index in block_pixel_map
        else xy_to_pixel(target_block_xy.tolist(), affine)
    )

    draw.line((target_ee_pixel[0], target_ee_pixel[1], pred_pixel[0], pred_pixel[1]), fill=(255, 255, 255), width=1)
    draw_point(draw, target_block_pixel, fill=(30, 180, 90), label=f"block {target_block_index}")
    draw_point(draw, target_ee_pixel, fill=(230, 80, 180), label="target ee")
    draw_point(draw, pred_pixel, fill=(255, 190, 40), label="pred xy")
    draw_point(draw, nearest_block_pixel, fill=(70, 140, 255), label=f"nearest {nearest_block_index}")

    footer = (
        f"target block {target_block_index}  nearest {nearest_block_index}  "
        f"xy_error_norm={xy_error_norm:.4f}  xy_error_table={xy_error_table:.2f}"
    )
    font = ImageFont.load_default()
    width, height = image.size
    bbox = draw.textbbox((8, height - 22), footer, font=font)
    draw.rectangle((bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3), fill=(0, 0, 0))
    draw.text((8, height - 22), footer, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    payload = load_checkpoint(checkpoint_path)
    config = config_from_payload(payload)
    data_root = args.data_root or config.val_dataset_root or config.dataset_root
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "xy_prediction_overlays"
    camera_name = args.camera_name or config.camera_name
    device, _ = resolve_torch_device(args.device)

    if config.output_type != "xy":
        raise ValueError(f"Checkpoint output_type is {config.output_type!r}; expected 'xy'")

    dataset = RobosuiteVisualCorsiDataset(
        data_root,
        camera_name=camera_name,
        include_reset_frame=config.include_reset_frame,
        heatmap_size=config.heatmap_size,
        heatmap_sigma=config.heatmap_sigma,
        heatmap_normalize=config.heatmap_normalize,
        target_type=config.target_type,
    )
    block_xy_norm = block_norm_xy_from_mapping(dataset.block_xy_norm)
    table_delta_scale = table_delta_scale_from_xy_normalization(dataset.xy_normalization)
    block_pixel_map = collect_block_pixel_map(dataset, camera_name)
    affine = fit_norm_to_pixel_affine(block_xy_norm, block_pixel_map)

    model = VisualSeq2SeqLSTM(build_model_config(config)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_visual_batch)

    saved = 0
    overlays: list[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model.greedy_decode(
                frames=batch["frames"],
                frame_lengths=batch["frame_lengths"],
                target_lengths=batch["target_lengths"],
                max_steps=batch["targets"].size(1),
                delay_mode=config.delay_mode,
                delay_steps=config.delay_steps,
                blank_feature_mode=config.blank_feature_mode,
            )
            pred_xy = outputs["pred_xy"].detach().cpu()
            target_xy = batch["target_xy"].detach().cpu()
            target_block_xy = batch["target_block_xy_norm"].detach().cpu()
            target_block_indices = batch["target_block_indices"].detach().cpu()
            nearest_predictions, _ = nearest_block_decode(pred_xy, block_xy=block_xy_norm)
            errors_norm = torch.linalg.norm(pred_xy.float() - target_xy.float(), dim=-1)
            errors_table = torch.linalg.norm((pred_xy.float() - target_xy.float()) * table_delta_scale, dim=-1)
            frames = batch["frames"].detach().cpu()

            for batch_index in range(frames.size(0)):
                length = int(batch["target_lengths"][batch_index].item())
                for step_index in range(length):
                    if saved >= args.max_samples:
                        break
                    output_path = (
                        output_dir
                        / str(batch["trial_ids"][batch_index])
                        / f"step_{step_index:02d}_xy_prediction.png"
                    )
                    save_overlay(
                        output_path=output_path,
                        frame=frames[batch_index, step_index],
                        metadata=batch["step_metadata"][batch_index][step_index],
                        camera_name=camera_name,
                        target_xy=target_xy[batch_index, step_index],
                        pred_xy=pred_xy[batch_index, step_index],
                        target_block_xy=target_block_xy[batch_index, step_index],
                        target_block_index=int(target_block_indices[batch_index, step_index].item()),
                        nearest_block_index=int(nearest_predictions[batch_index, step_index].item()),
                        block_pixel_map=block_pixel_map,
                        affine=affine,
                        xy_error_norm=float(errors_norm[batch_index, step_index].item()),
                        xy_error_table=float(errors_table[batch_index, step_index].item()),
                    )
                    overlays.append(str(output_path))
                    saved += 1
                if saved >= args.max_samples:
                    break
            if saved >= args.max_samples:
                break

    summary = {
        "checkpoint": str(checkpoint_path),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "num_overlays": len(overlays),
        "overlay_paths": overlays,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
