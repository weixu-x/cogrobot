"""Creates overlay previews for the embodied visual Corsi EE XY dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.experiments.visual_base.scripts.ee_xy_dataset_utils import norm_xy_to_table  # noqa: E402


DEFAULT_DATASET_ROOT = "corsi_artifacts/visual_base/datasets/freecam_ee_xy_v1_preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--camera", type=str, default="freecam")
    parser.add_argument("--max-trials", type=int, default=5)
    parser.add_argument("--max-steps-per-trial", type=int, default=8)
    parser.add_argument("--random-subset", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--axis-map", choices=["freecam_default", "table_xy"], default="freecam_default")
    parser.add_argument("--padding", type=int, default=56)
    return parser.parse_args()


def resolve_path(path_text: str, *, dataset_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute() or path.exists():
        return path
    candidate = dataset_root / path_text
    if candidate.exists():
        return candidate
    return path


def norm_to_pixel(
    norm_xy: Sequence[float],
    *,
    width: int,
    height: int,
    padding: int,
    axis_map: str,
) -> tuple[int, int]:
    nx = float(norm_xy[0])
    ny = float(norm_xy[1])
    left = float(padding)
    right = float(width - padding)
    top = float(padding)
    bottom = float(height - padding)
    if axis_map == "freecam_default":
        px = left + (ny + 1.0) * 0.5 * (right - left)
        py = top + (1.0 - nx) * 0.5 * (bottom - top)
    else:
        px = left + (nx + 1.0) * 0.5 * (right - left)
        py = top + (1.0 - ny) * 0.5 * (bottom - top)
    return int(round(px)), int(round(py))


def draw_point(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    label: str,
) -> None:
    x, y = point
    radius = 7
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
    draw.line((x - 12, y, x + 12, y), fill=outline, width=2)
    draw.line((x, y - 12, x, y + 12), fill=outline, width=2)
    font = ImageFont.load_default()
    text_x = x + 10
    text_y = max(2, y - 18)
    bbox = draw.textbbox((text_x, text_y), label, font=font)
    draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=(0, 0, 0))
    draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)


def overlay_step(
    *,
    image_path: Path,
    output_path: Path,
    step_metadata: dict,
    xy_bounds: dict,
    camera: str,
    axis_map: str,
    padding: int,
) -> None:
    image = Image.fromarray(imageio.imread(image_path)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    image_points = step_metadata.get("image_points", {}).get(camera, {})
    if "target_block_pixel" in image_points and "ee_pixel" in image_points:
        target_point = tuple(int(v) for v in image_points["target_block_pixel"])
        ee_point = tuple(int(v) for v in image_points["ee_pixel"])
    else:
        target_point = norm_to_pixel(
            step_metadata["target_block_xy_norm"],
            width=width,
            height=height,
            padding=padding,
            axis_map=axis_map,
        )
        ee_point = norm_to_pixel(
            step_metadata["ee_xy_norm"],
            width=width,
            height=height,
            padding=padding,
            axis_map=axis_map,
        )
    block_index = int(step_metadata["block_index"])
    target_xy_table = norm_xy_to_table(step_metadata["target_block_xy_norm"], xy_bounds)
    ee_xy_table = norm_xy_to_table(step_metadata["ee_xy_norm"], xy_bounds)
    draw.line((target_point[0], target_point[1], ee_point[0], ee_point[1]), fill=(255, 255, 255), width=1)
    draw_point(
        draw,
        target_point,
        fill=(30, 180, 90),
        outline=(255, 255, 255),
        label=f"block {block_index}",
    )
    draw_point(
        draw,
        ee_point,
        fill=(220, 60, 180),
        outline=(255, 255, 255),
        label="ee",
    )
    footer = (
        f"step {step_metadata['step']}  block {block_index}  "
        f"target_xy=({target_xy_table[0]:.3f},{target_xy_table[1]:.3f})  "
        f"ee_xy=({ee_xy_table[0]:.3f},{ee_xy_table[1]:.3f})"
    )
    font = ImageFont.load_default()
    bbox = draw.textbbox((8, height - 22), footer, font=font)
    draw.rectangle((bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3), fill=(0, 0, 0))
    draw.text((8, height - 22), footer, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "preview_overlays"
    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    xy_bounds = dataset_manifest["xy_normalization"]["bounds"]

    overlay_paths: list[str] = []
    samples = list(dataset_manifest.get("samples", []))
    if args.random_subset:
        rng = random.Random(int(args.sample_seed))
        rng.shuffle(samples)
    samples = samples[: args.max_trials]
    selected_trial_ids = [str(sample.get("trial_id", "")) for sample in samples]
    for sample in samples:
        manifest_path = resolve_path(sample["manifest_path"], dataset_root=dataset_root)
        trial_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        trial_id = trial_manifest["trial_id"]
        keyframes = trial_manifest["keyframe_paths"][args.camera]
        step_metadata = trial_manifest["step_metadata"]
        for step_index, metadata in enumerate(step_metadata[: args.max_steps_per_trial]):
            image_path = resolve_path(keyframes[step_index], dataset_root=dataset_root)
            output_path = output_dir / trial_id / f"step_{step_index:02d}_block_{metadata['block_index']}_{args.camera}_overlay.png"
            overlay_step(
                image_path=image_path,
                output_path=output_path,
                step_metadata=metadata,
                xy_bounds=xy_bounds,
                camera=args.camera,
                axis_map=args.axis_map,
                padding=args.padding,
            )
            overlay_paths.append(str(output_path))

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "camera": args.camera,
        "num_overlays": len(overlay_paths),
        "overlay_paths": overlay_paths,
        "axis_map": args.axis_map,
        "random_subset": bool(args.random_subset),
        "sample_seed": int(args.sample_seed),
        "trial_ids": selected_trial_ids,
    }
    summary_path = output_dir / "preview_overlay_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Preview overlays saved to {output_dir}")


if __name__ == "__main__":
    main()
