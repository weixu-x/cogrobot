"""Shared helpers for embodied visual Corsi XY preview datasets."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from corsi.envs.sequence_generator import (
    STANDARD_CORSI_BOARD_SIZE,
    STANDARD_CORSI_ROBOSUITE_TARGET_WIDTH,
)


XY_BOUNDS_KEYS = ("x_min", "x_max", "y_min", "y_max")


def table_xy_to_norm(table_xy: Sequence[float], bounds: Mapping[str, float]) -> list[float]:
    """Maps table-frame XY coordinates to [-1, 1] using explicit bounds."""

    x, y = float(table_xy[0]), float(table_xy[1])
    x_min = float(bounds["x_min"])
    x_max = float(bounds["x_max"])
    y_min = float(bounds["y_min"])
    y_max = float(bounds["y_max"])
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid XY bounds: {dict(bounds)}")
    return [
        2.0 * (x - x_min) / (x_max - x_min) - 1.0,
        2.0 * (y - y_min) / (y_max - y_min) - 1.0,
    ]


def norm_xy_to_table(norm_xy: Sequence[float], bounds: Mapping[str, float]) -> list[float]:
    """Maps normalized [-1, 1] XY coordinates back to table-frame XY."""

    nx, ny = float(norm_xy[0]), float(norm_xy[1])
    x_min = float(bounds["x_min"])
    x_max = float(bounds["x_max"])
    y_min = float(bounds["y_min"])
    y_max = float(bounds["y_max"])
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid XY bounds: {dict(bounds)}")
    return [
        x_min + (nx + 1.0) * 0.5 * (x_max - x_min),
        y_min + (ny + 1.0) * 0.5 * (y_max - y_min),
    ]


def xy_bounds_from_board_size(board_size_xy: Sequence[float]) -> dict[str, float]:
    """Returns centered table-frame normalization bounds for a Corsi board."""

    board_width = float(board_size_xy[0])
    board_height = float(board_size_xy[1])
    if board_width <= 0.0 or board_height <= 0.0:
        raise ValueError(f"Invalid board size: {board_size_xy}")
    return {
        "x_min": -board_width / 2.0,
        "x_max": board_width / 2.0,
        "y_min": -board_height / 2.0,
        "y_max": board_height / 2.0,
    }


def corsi_lower_left_xy_bounds(
    board_size: Sequence[float] = STANDARD_CORSI_BOARD_SIZE,
) -> dict[str, float]:
    """Returns normalization bounds for the canonical lower-left Corsi frame."""

    board_width = float(board_size[0])
    board_height = float(board_size[1])
    if board_width <= 0.0 or board_height <= 0.0:
        raise ValueError(f"Invalid board size: {board_size}")
    return {
        "x_min": 0.0,
        "x_max": board_width,
        "y_min": 0.0,
        "y_max": board_height,
    }


def robosuite_table_xy_to_corsi_xy(
    table_xy: Sequence[float],
    *,
    board_size: Sequence[float] = STANDARD_CORSI_BOARD_SIZE,
    target_width: float = STANDARD_CORSI_ROBOSUITE_TARGET_WIDTH,
) -> list[float]:
    """Inverts `standard_corsi_robosuite_layout` into lower-left Corsi XY."""

    table_x, table_y = float(table_xy[0]), float(table_xy[1])
    board_width, board_height = float(board_size[0]), float(board_size[1])
    scale = float(target_width) / board_width
    if scale <= 0.0:
        raise ValueError(f"Invalid robosuite target width: {target_width}")

    centered_corsi_x = table_y / scale
    centered_corsi_y = -table_x / scale
    return [
        centered_corsi_x + board_width / 2.0,
        centered_corsi_y + board_height / 2.0,
    ]


def corsi_xy_to_robosuite_table_xy(
    corsi_xy: Sequence[float],
    *,
    board_size: Sequence[float] = STANDARD_CORSI_BOARD_SIZE,
    target_width: float = STANDARD_CORSI_ROBOSUITE_TARGET_WIDTH,
) -> list[float]:
    """Maps lower-left Corsi XY into the robosuite table XY frame."""

    corsi_x, corsi_y = float(corsi_xy[0]), float(corsi_xy[1])
    board_width, board_height = float(board_size[0]), float(board_size[1])
    scale = float(target_width) / board_width
    if scale <= 0.0:
        raise ValueError(f"Invalid robosuite target width: {target_width}")
    centered_corsi_x = corsi_x - board_width / 2.0
    centered_corsi_y = corsi_y - board_height / 2.0
    return [
        -centered_corsi_y * scale,
        centered_corsi_x * scale,
    ]


def finite_xy(value: Sequence[float]) -> bool:
    """Returns True when a two-element coordinate contains finite values."""

    arr = np.asarray(value, dtype=np.float64)
    return arr.shape == (2,) and bool(np.isfinite(arr).all())


def norm_xy_in_bounds(value: Sequence[float], *, tolerance: float = 1.0e-6) -> bool:
    """Returns True when normalized XY lies inside [-1, 1] within tolerance."""

    arr = np.asarray(value, dtype=np.float64)
    return arr.shape == (2,) and bool((arr >= -1.0 - tolerance).all() and (arr <= 1.0 + tolerance).all())
