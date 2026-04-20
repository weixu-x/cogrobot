"""Builds a single summary figure from exported robosuite Corsi frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise ImportError("make_robosuite_sequence_figure.py requires Pillow") from exc


BACKGROUND = "#f7f4ec"
PANEL_BG = "#ffffff"
PANEL_OUTLINE = "#3d352d"
TEXT = "#1f1a17"
SUBTEXT = "#4a4038"
CHIP_BG = "#1b6ca8"
CHIP_TEXT = "#ffffff"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--camera", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    return parser.parse_args()


def _load_font(size: int):
    for name in ["Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _human_block_ids(sequence: list[int]) -> list[int]:
    return [int(index) + 1 for index in sequence]


def _load_manifest(input_dir: Path) -> dict:
    manifest_path = input_dir / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _pick_camera(manifest: dict, explicit_camera: str) -> str:
    if explicit_camera:
        return explicit_camera
    camera_names = manifest.get("camera_names", [])
    if not camera_names:
        raise ValueError("No camera names found in manifest")
    return str(camera_names[0])


def _load_keyframes(input_dir: Path, camera: str, sequence: list[int]) -> list[Image.Image]:
    keyframe_dir = input_dir / "rollout_keyframes" / camera
    frames = []
    for step_index, block_index in enumerate(sequence):
        frame_path = keyframe_dir / f"keyframe_step_{step_index:02d}_block_{block_index}_{camera}.png"
        frames.append(Image.open(frame_path).convert("RGB"))
    return frames


def build_summary_figure(input_dir: Path, camera: str) -> Path:
    manifest = _load_manifest(input_dir)
    sequence = [int(item) for item in manifest["sequence"]]
    human_sequence = _human_block_ids(sequence)

    reset_path = Path(manifest["reset_paths"][camera])
    reset_image = Image.open(reset_path).convert("RGB")
    keyframes = _load_keyframes(input_dir, camera, sequence)

    card_width = 280
    gap = 22
    margin = 24
    title_h = 78
    chip_h = 56
    caption_h = 34

    thumb_width = card_width
    thumb_height = int(keyframes[0].height * (thumb_width / keyframes[0].width))
    left_panel_w = 360
    right_panel_w = max(card_width * 2 + gap, left_panel_w)
    canvas_w = margin * 2 + left_panel_w + gap + right_panel_w
    grid_rows = (len(keyframes) + 1) // 2
    grid_h = grid_rows * (thumb_height + caption_h) + (grid_rows - 1) * gap
    reset_h = int(reset_image.height * ((left_panel_w - 2 * margin) / reset_image.width))
    canvas_h = margin * 2 + title_h + chip_h + gap + max(reset_h, grid_h) + 24

    image = Image.new("RGB", (canvas_w, canvas_h), BACKGROUND)
    draw = ImageDraw.Draw(image)

    title_font = _load_font(30)
    body_font = _load_font(18)
    chip_font = _load_font(18)
    caption_font = _load_font(17)

    draw.text((margin, margin), f"Robosuite Corsi Sample ({camera})", fill=TEXT, font=title_font)
    draw.text(
        (margin, margin + 40),
        f"index sequence: {sequence}    human block ids: {human_sequence}",
        fill=SUBTEXT,
        font=body_font,
    )

    chip_x = margin
    chip_y = margin + title_h
    for step_index, (block_index, human_block) in enumerate(zip(sequence, human_sequence), start=1):
        chip_text = f"{step_index}. idx {block_index} / block {human_block}"
        text_w, text_h = _text_size(draw, chip_text, chip_font)
        chip_w = text_w + 22
        chip_box = (chip_x, chip_y, chip_x + chip_w, chip_y + 34)
        draw.rounded_rectangle(chip_box, radius=14, fill=CHIP_BG)
        draw.text((chip_x + 11, chip_y + 8), chip_text, fill=CHIP_TEXT, font=chip_font)
        chip_x += chip_w + 10

    left_x = margin
    left_y = margin + title_h + chip_h + gap
    reset_box = (left_x, left_y, left_x + left_panel_w, left_y + reset_h + 44)
    draw.rounded_rectangle(reset_box, radius=18, fill=PANEL_BG, outline=PANEL_OUTLINE, width=2)
    draw.text((left_x + 16, left_y + 12), "Reset View", fill=TEXT, font=body_font)
    reset_resized = reset_image.resize((left_panel_w - 2 * margin, reset_h), Image.Resampling.LANCZOS)
    image.paste(reset_resized, (left_x + margin, left_y + 40))

    right_x = left_x + left_panel_w + gap
    right_y = left_y
    for frame_index, frame in enumerate(keyframes):
        row = frame_index // 2
        col = frame_index % 2
        cell_x = right_x + col * (card_width + gap)
        cell_y = right_y + row * (thumb_height + caption_h + gap)
        box = (cell_x, cell_y, cell_x + card_width, cell_y + thumb_height + caption_h)
        draw.rounded_rectangle(box, radius=18, fill=PANEL_BG, outline=PANEL_OUTLINE, width=2)
        thumb = frame.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        image.paste(thumb, (cell_x, cell_y))
        caption = f"step {frame_index + 1}  idx {sequence[frame_index]}  block {human_sequence[frame_index]}"
        draw.text((cell_x + 14, cell_y + thumb_height + 8), caption, fill=TEXT, font=caption_font)

    output_path = input_dir / f"sequence_summary_{camera}.png"
    image.save(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    manifest = _load_manifest(input_dir)
    camera = _pick_camera(manifest, args.camera)
    output_path = Path(args.output_path) if args.output_path else build_summary_figure(input_dir, camera)
    if args.output_path:
        built_path = build_summary_figure(input_dir, camera)
        Image.open(built_path).save(output_path)
        if built_path != output_path and built_path.exists():
            built_path.unlink()
    print(output_path)


if __name__ == "__main__":
    main()
