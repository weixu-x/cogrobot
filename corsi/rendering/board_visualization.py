"""Board visualization utilities for coordinate-based Corsi inspection."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


Layout = Mapping[int, Sequence[float]]

BACKGROUND = "#f7f4ec"
BOARD_FILL = "#efe8db"
BOARD_OUTLINE = "#3d352d"
BLOCK_FILL = "#fdf9f0"
BLOCK_ACTIVE_FILL = "#f3c55c"
BLOCK_TEXT = "#1f1a17"
ORDER_BG = "#1b6ca8"
ORDER_TEXT = "#ffffff"
TITLE_TEXT = "#2d241f"


def board_rows_from_layout(layout: Layout) -> List[List[int]]:
    grouped: Dict[float, List[Tuple[float, int]]] = {}
    for block_id, (x, y) in layout.items():
        grouped.setdefault(float(y), []).append((float(x), int(block_id)))

    rows: List[List[int]] = []
    for y in sorted(grouped.keys(), reverse=True):
        rows.append([block_id for _, block_id in sorted(grouped[y], key=lambda item: item[0])])
    return rows


def sequence_order_map(sequence: Sequence[int]) -> Dict[int, List[int]]:
    order_map: Dict[int, List[int]] = {}
    for step_index, block_id in enumerate(sequence, start=1):
        order_map.setdefault(int(block_id), []).append(step_index)
    return order_map


def _load_font(size: int):
    for name in ["Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_board_png(
    layout: Layout,
    sequence: Sequence[int],
    title: str,
    output_path: Path,
    *,
    cell_size: int = 160,
    margin: int = 50,
) -> Path:
    rows = board_rows_from_layout(layout)
    order_map = sequence_order_map(sequence)

    width = margin * 2 + cell_size * 3
    height = margin * 2 + cell_size * 3 + 90
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    title_font = _load_font(34)
    label_font = _load_font(24)
    order_font = _load_font(22)

    draw.rounded_rectangle(
        [(24, 24), (width - 24, height - 24)],
        radius=28,
        fill=BOARD_FILL,
        outline=BOARD_OUTLINE,
        width=3,
    )
    draw.text((margin, 34), title, fill=TITLE_TEXT, font=title_font)

    board_top = margin + 70
    for row_index, row in enumerate(rows):
        for col_index, block_id in enumerate(row):
            left = margin + col_index * cell_size
            top = board_top + row_index * cell_size
            right = left + cell_size - 18
            bottom = top + cell_size - 18
            is_active = block_id in order_map
            fill = BLOCK_ACTIVE_FILL if is_active else BLOCK_FILL
            draw.rounded_rectangle(
                [(left, top), (right, bottom)],
                radius=26,
                fill=fill,
                outline=BOARD_OUTLINE,
                width=3,
            )
            draw.text((left + 16, top + 14), f"Block {block_id}", fill=BLOCK_TEXT, font=label_font)

            orders = order_map.get(block_id, [])
            if orders:
                chip_left = left + 16
                chip_top = top + 68
                for order in orders:
                    chip = [chip_left, chip_top, chip_left + 42, chip_top + 34]
                    draw.rounded_rectangle(chip, radius=14, fill=ORDER_BG)
                    draw.text((chip_left + 12, chip_top + 6), str(order), fill=ORDER_TEXT, font=order_font)
                    chip_left += 52

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def render_board_svg(
    layout: Layout,
    sequence: Sequence[int],
    title: str,
    output_path: Path,
    *,
    cell_size: int = 160,
    margin: int = 50,
) -> Path:
    rows = board_rows_from_layout(layout)
    order_map = sequence_order_map(sequence)

    width = margin * 2 + cell_size * 3
    height = margin * 2 + cell_size * 3 + 90
    board_top = margin + 70

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{BACKGROUND}"/>',
        f'<rect x="24" y="24" width="{width - 48}" height="{height - 48}" rx="28" fill="{BOARD_FILL}" stroke="{BOARD_OUTLINE}" stroke-width="3"/>',
        f'<text x="{margin}" y="58" font-size="34" font-family="Helvetica, Arial, sans-serif" fill="{TITLE_TEXT}">{html.escape(title)}</text>',
    ]

    for row_index, row in enumerate(rows):
        for col_index, block_id in enumerate(row):
            left = margin + col_index * cell_size
            top = board_top + row_index * cell_size
            width_box = cell_size - 18
            height_box = cell_size - 18
            is_active = block_id in order_map
            fill = BLOCK_ACTIVE_FILL if is_active else BLOCK_FILL

            parts.append(
                f'<rect x="{left}" y="{top}" width="{width_box}" height="{height_box}" rx="26" fill="{fill}" stroke="{BOARD_OUTLINE}" stroke-width="3"/>'
            )
            parts.append(
                f'<text x="{left + 16}" y="{top + 40}" font-size="24" font-family="Helvetica, Arial, sans-serif" fill="{BLOCK_TEXT}">Block {block_id}</text>'
            )

            chip_left = left + 16
            chip_top = top + 68
            for order in order_map.get(block_id, []):
                parts.append(
                    f'<rect x="{chip_left}" y="{chip_top}" width="42" height="34" rx="14" fill="{ORDER_BG}"/>'
                )
                parts.append(
                    f'<text x="{chip_left + 12}" y="{chip_top + 24}" font-size="22" font-family="Helvetica, Arial, sans-serif" fill="{ORDER_TEXT}">{order}</text>'
                )
                chip_left += 52

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path
