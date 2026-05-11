"""Heatmap target, loss, and decode helpers for visual Corsi models."""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None
    Tensor = object  # type: ignore[assignment,misc]

from corsi.envs.sequence_generator import (
    STANDARD_CORSI_BOARD_SIZE,
    standard_corsi_layout,
)


HeatmapLossName = Literal["spatial_ce", "mse"]


def standard_block_heatmap_xy(size: int, *, num_blocks: int = 9) -> np.ndarray:
    """Returns block center coordinates in heatmap pixel space as `[block, x, y]`."""

    if size < 2:
        raise ValueError("heatmap size must be >= 2")
    layout = standard_corsi_layout(origin="image", use_block_centers=True)
    board_width, board_height = STANDARD_CORSI_BOARD_SIZE
    coords = []
    for block_id in range(num_blocks):
        center_x, center_y = layout[block_id]
        x = float(center_x) / float(board_width) * float(size - 1)
        y = (float(board_height) - float(center_y)) / float(board_height) * float(size - 1)
        coords.append((x, y))
    return np.asarray(coords, dtype=np.float32)


def gaussian_heatmap(
    center_xy: Tuple[float, float],
    *,
    size: int,
    sigma: float,
    normalize: bool = True,
) -> np.ndarray:
    """Builds a 2D Gaussian heatmap centered at `(x, y)` in heatmap coordinates."""

    if size < 2:
        raise ValueError("heatmap size must be >= 2")
    if sigma <= 0:
        raise ValueError("heatmap sigma must be > 0")
    center_x, center_y = center_xy
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    dist_sq = (xs - float(center_x)) ** 2 + (ys - float(center_y)) ** 2
    heatmap = np.exp(-dist_sq / (2.0 * float(sigma) ** 2)).astype(np.float32)
    if normalize:
        total = float(heatmap.sum())
        if total > 0.0:
            heatmap = heatmap / total
    return heatmap


def target_heatmap_for_block(
    block_index: int,
    *,
    size: int,
    sigma: float,
    normalize: bool = True,
    num_blocks: int = 9,
) -> np.ndarray:
    """Converts a target block index into a Gaussian heatmap."""

    if block_index < 0 or block_index >= num_blocks:
        raise ValueError(f"block index {block_index} is outside [0, {num_blocks - 1}]")
    centers = standard_block_heatmap_xy(size, num_blocks=num_blocks)
    return gaussian_heatmap(
        tuple(float(value) for value in centers[int(block_index)]),
        size=size,
        sigma=sigma,
        normalize=normalize,
    )


def sequence_to_target_heatmaps(
    sequence,
    *,
    size: int,
    sigma: float,
    normalize: bool = True,
    num_blocks: int = 9,
) -> np.ndarray:
    """Converts an index target sequence into `[T, H, W]` Gaussian heatmaps."""

    return np.stack(
        [
            target_heatmap_for_block(
                int(block_index),
                size=size,
                sigma=sigma,
                normalize=normalize,
                num_blocks=num_blocks,
            )
            for block_index in sequence
        ],
        axis=0,
    ).astype(np.float32)


def spatial_heatmap_loss(
    heatmap_logits: Tensor,
    target_heatmaps: Tensor,
    mask: Tensor,
    *,
    loss_type: HeatmapLossName = "spatial_ce",
    eps: float = 1e-8,
) -> Tensor:
    """Masked heatmap loss over active sequence positions."""

    if torch is None or F is None:  # pragma: no cover
        raise ImportError("spatial_heatmap_loss requires PyTorch")
    if heatmap_logits.dim() != 5:
        raise ValueError("heatmap_logits must have shape [B, T, 1, H, W]")
    if target_heatmaps.dim() != 4:
        raise ValueError("target_heatmaps must have shape [B, T, H, W]")
    logits = heatmap_logits.squeeze(2)
    if logits.shape != target_heatmaps.shape:
        raise ValueError(
            f"heatmap shape mismatch: logits={tuple(logits.shape)}, "
            f"targets={tuple(target_heatmaps.shape)}"
        )
    active_mask = mask.to(dtype=logits.dtype)
    active_count = active_mask.sum().clamp_min(1.0)

    if loss_type == "spatial_ce":
        batch_size, steps, height, width = logits.shape
        flat_logits = logits.reshape(batch_size, steps, height * width)
        flat_targets = target_heatmaps.reshape(batch_size, steps, height * width)
        target_dist = flat_targets / flat_targets.sum(dim=-1, keepdim=True).clamp_min(eps)
        token_loss = -(target_dist * F.log_softmax(flat_logits, dim=-1)).sum(dim=-1)
    elif loss_type == "mse":
        token_loss = F.mse_loss(logits, target_heatmaps, reduction="none").mean(dim=(-1, -2))
    else:
        raise ValueError(f"Unsupported heatmap loss: {loss_type}")

    return (token_loss * active_mask).sum() / active_count


def decode_heatmap_argmax(heatmap_logits: Tensor) -> Tensor:
    """Decodes heatmap logits to integer `[x, y]` argmax coordinates."""

    if torch is None:  # pragma: no cover
        raise ImportError("decode_heatmap_argmax requires PyTorch")
    if heatmap_logits.dim() != 5:
        raise ValueError("heatmap_logits must have shape [B, T, 1, H, W]")
    logits = heatmap_logits.squeeze(2)
    _, _, _, width = logits.shape
    flat_indices = logits.flatten(start_dim=-2).argmax(dim=-1)
    y = torch.div(flat_indices, width, rounding_mode="floor")
    x = flat_indices.remainder(width)
    return torch.stack([x, y], dim=-1)


def nearest_block_decode(
    xy: Tensor,
    *,
    block_xy: Tensor | None = None,
    size: int | None = None,
    num_blocks: int = 9,
) -> tuple[Tensor, Tensor]:
    """Maps heatmap coordinates to nearest block indices and distances."""

    if torch is None:  # pragma: no cover
        raise ImportError("nearest_block_decode requires PyTorch")
    xy_float = xy.float()
    if block_xy is None:
        if size is None:
            raise ValueError("size is required when block_xy is not provided")
        block_xy = torch.as_tensor(
            standard_block_heatmap_xy(size, num_blocks=num_blocks),
            dtype=xy_float.dtype,
            device=xy.device,
        )
    else:
        block_xy = block_xy.to(dtype=xy_float.dtype, device=xy.device)
    distances = torch.cdist(xy_float.reshape(-1, 2), block_xy).reshape(*xy.shape[:-1], block_xy.size(0))
    min_distances, indices = distances.min(dim=-1)
    return indices, min_distances
