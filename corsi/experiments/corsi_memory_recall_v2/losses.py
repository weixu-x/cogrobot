"""Loss helpers for Corsi memory-recall V2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

DEFAULT_LOSS_WEIGHTS = {
    "seq": 1.0,
    "memory_order": 0.3,
    "memory_order_final": 0.0,
    "memory_length": 0.0,
    "coord": 0.05,
    "joint": 0.1,
    "ee_pose": 0.05,
    "ee_xy": 0.05,
}


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    masked = values * mask
    denom = mask.expand_as(values).sum()
    if denom.item() == 0:
        return values.sum() * 0.0
    return masked.sum() / denom.clamp_min(1.0)


def sequence_cross_entropy_loss(logits: Tensor, targets: Tensor, *, ignore_index: int = -100) -> Tensor:
    """Cross entropy over block+EOS logits, ignoring padded target positions."""

    if logits.ndim != 3:
        raise ValueError(f"expected logits [B,T,C], got shape={tuple(logits.shape)}")
    if targets.shape != logits.shape[:2]:
        raise ValueError(f"targets shape {tuple(targets.shape)} does not match logits [B,T]={tuple(logits.shape[:2])}")
    valid = targets.ne(int(ignore_index))
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=int(ignore_index),
    )


def weak_coord_loss(
    pred_coord: Tensor,
    target_xy: Tensor,
    *,
    target_tokens: Tensor | None = None,
    token_mask: Tensor | None = None,
    eos_token_id: int = 9,
    ignore_index: int = -100,
) -> Tensor:
    """MSE for weak block-coordinate supervision on non-EOS recall steps."""

    if pred_coord.ndim != 3 or pred_coord.shape[-1] != 2:
        raise ValueError(f"expected pred_coord [B,T,2], got shape={tuple(pred_coord.shape)}")
    if target_xy.ndim != 3 or target_xy.shape[-1] != 2:
        raise ValueError(f"expected target_xy [B,T,2] or [B,L,2], got shape={tuple(target_xy.shape)}")
    steps = min(pred_coord.shape[1], target_xy.shape[1])
    pred = pred_coord[:, :steps]
    target = target_xy[:, :steps].to(device=pred.device, dtype=pred.dtype)
    valid = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
    if token_mask is not None:
        valid = valid & token_mask[:, :steps].to(device=pred.device, dtype=torch.bool)
    if target_tokens is not None:
        tokens = target_tokens[:, :steps].to(device=pred.device)
        valid = valid & tokens.ne(int(ignore_index)) & tokens.ne(int(eos_token_id))
    per_step = (pred - target).pow(2).mean(dim=-1)
    return _masked_mean(per_step, valid)


def memory_order_prefix_loss(
    logits: Tensor,
    target_tokens: Tensor,
    *,
    segment_mask: Tensor,
    token_mask: Tensor | None = None,
    eos_token_id: int = 9,
    ignore_index: int = -100,
) -> Tensor:
    """Prefix CE from each memory update state to all block identities seen so far."""

    if logits.ndim != 4:
        raise ValueError(f"expected memory order logits [B,L,P,C], got shape={tuple(logits.shape)}")
    if segment_mask.shape != logits.shape[:2]:
        raise ValueError(
            f"segment_mask shape {tuple(segment_mask.shape)} does not match logits [B,L]={tuple(logits.shape[:2])}"
        )
    batch_size, updates, positions, classes = logits.shape
    del batch_size
    del classes
    targets = target_tokens[:, :positions].to(device=logits.device)
    if targets.shape[1] < positions:
        pad = torch.full(
            (targets.shape[0], positions - targets.shape[1]),
            int(ignore_index),
            dtype=targets.dtype,
            device=targets.device,
        )
        targets = torch.cat([targets, pad], dim=1)
    valid_position = targets.ne(int(ignore_index)) & targets.ne(int(eos_token_id))
    if token_mask is not None:
        mask = token_mask[:, :positions].to(device=logits.device, dtype=torch.bool)
        if mask.shape[1] < positions:
            pad = torch.zeros(
                (mask.shape[0], positions - mask.shape[1]),
                dtype=torch.bool,
                device=mask.device,
            )
            mask = torch.cat([mask, pad], dim=1)
        valid_position = valid_position & mask
    update_valid = segment_mask[:, :updates].to(device=logits.device, dtype=torch.bool)
    update_index = torch.arange(updates, device=logits.device).view(1, updates, 1)
    position_index = torch.arange(positions, device=logits.device).view(1, 1, positions)
    prefix_valid = position_index <= update_index
    valid = update_valid.unsqueeze(-1) & valid_position.unsqueeze(1) & prefix_valid
    if not bool(valid.any()):
        return logits.sum() * 0.0
    expanded_targets = targets.unsqueeze(1).expand(-1, updates, -1)
    safe_targets = expanded_targets.masked_fill(~valid, 0)
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        safe_targets.reshape(-1),
        reduction="none",
    ).reshape_as(valid.float())
    exposure = (update_valid.unsqueeze(-1) & prefix_valid).sum(dim=1).clamp_min(1).to(dtype=logits.dtype)
    weights = valid.to(dtype=logits.dtype) / exposure.unsqueeze(1)
    return (per_token * weights).sum() / weights.sum().clamp_min(1.0)


def final_memory_order_loss(
    logits: Tensor,
    target_tokens: Tensor,
    *,
    token_mask: Tensor | None = None,
    eos_token_id: int = 9,
    ignore_index: int = -100,
) -> Tensor:
    """CE from final memory state to the full ordered block sequence."""

    if logits.ndim != 3:
        raise ValueError(f"expected final memory order logits [B,P,C], got shape={tuple(logits.shape)}")
    targets = target_tokens[:, : logits.shape[1]].to(device=logits.device)
    if targets.shape[1] < logits.shape[1]:
        pad = torch.full(
            (targets.shape[0], logits.shape[1] - targets.shape[1]),
            int(ignore_index),
            dtype=targets.dtype,
            device=targets.device,
        )
        targets = torch.cat([targets, pad], dim=1)
    valid = targets.ne(int(ignore_index)) & targets.ne(int(eos_token_id))
    if token_mask is not None:
        mask = token_mask[:, : logits.shape[1]].to(device=logits.device, dtype=torch.bool)
        if mask.shape[1] < logits.shape[1]:
            pad = torch.zeros(
                (mask.shape[0], logits.shape[1] - mask.shape[1]),
                dtype=torch.bool,
                device=mask.device,
            )
            mask = torch.cat([mask, pad], dim=1)
        valid = valid & mask
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], targets[valid])


def final_memory_length_loss(
    logits: Tensor,
    target_tokens: Tensor,
    *,
    token_mask: Tensor | None = None,
    eos_token_id: int = 9,
    ignore_index: int = -100,
) -> Tensor:
    """CE from final memory state to the number of recalled blocks."""

    if logits.ndim != 2:
        raise ValueError(f"expected final memory length logits [B,C], got shape={tuple(logits.shape)}")
    tokens = target_tokens.to(device=logits.device)
    valid = tokens.ne(int(ignore_index)) & tokens.ne(int(eos_token_id))
    if token_mask is not None:
        valid = valid & token_mask.to(device=logits.device, dtype=torch.bool)
    lengths = valid.sum(dim=1).clamp(min=1, max=logits.shape[-1]).long()
    labels = lengths - 1
    return F.cross_entropy(logits, labels)


def pose_auxiliary_loss(pred: Tensor, target: Tensor, *, mask: Tensor | None = None) -> Tensor:
    """Masked MSE for per-frame pose-like auxiliary targets."""

    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
    per_item = (pred - target.to(device=pred.device, dtype=pred.dtype)).pow(2)
    return _masked_mean(per_item, mask)


def ee_auxiliary_loss(pred: Tensor, target: Tensor, *, mask: Tensor | None = None) -> Tensor:
    return pose_auxiliary_loss(pred, target, mask=mask)


def combined_v2_loss(
    outputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    *,
    frame_mask: Tensor | None = None,
    weights: Mapping[str, float] | None = None,
    ignore_index: int = -100,
    eos_token_id: int = 9,
) -> dict[str, Tensor]:
    """Compute the weighted V2 objective from model outputs and collated targets."""

    loss_weights = {**DEFAULT_LOSS_WEIGHTS, **dict(weights or {})}
    device = outputs["logits"].device
    total = outputs["logits"].sum() * 0.0
    result: dict[str, Tensor] = {}

    seq = sequence_cross_entropy_loss(outputs["logits"], targets["tokens"].to(device=device), ignore_index=ignore_index)
    result["seq_loss"] = seq
    total = total + float(loss_weights["seq"]) * seq

    if "memory_order_logits" in outputs and float(loss_weights["memory_order"]) != 0.0:
        memory_order = memory_order_prefix_loss(
            outputs["memory_order_logits"],
            targets["tokens"],
            segment_mask=frame_mask.any(dim=-1) if frame_mask is not None and frame_mask.ndim == 3 else targets["block_xy"].new_ones(outputs["memory_order_logits"].shape[:2], dtype=torch.bool),
            token_mask=targets.get("token_mask"),
            eos_token_id=eos_token_id,
            ignore_index=ignore_index,
        )
        result["memory_order_loss"] = memory_order
        total = total + float(loss_weights["memory_order"]) * memory_order

    if "final_memory_order_logits" in outputs and float(loss_weights["memory_order_final"]) != 0.0:
        final_memory_order = final_memory_order_loss(
            outputs["final_memory_order_logits"],
            targets["tokens"],
            token_mask=targets.get("token_mask"),
            eos_token_id=eos_token_id,
            ignore_index=ignore_index,
        )
        result["final_memory_order_loss"] = final_memory_order
        total = total + float(loss_weights["memory_order_final"]) * final_memory_order

    if "final_memory_length_logits" in outputs and float(loss_weights["memory_length"]) != 0.0:
        final_memory_length = final_memory_length_loss(
            outputs["final_memory_length_logits"],
            targets["tokens"],
            token_mask=targets.get("token_mask"),
            eos_token_id=eos_token_id,
            ignore_index=ignore_index,
        )
        result["final_memory_length_loss"] = final_memory_length
        total = total + float(loss_weights["memory_length"]) * final_memory_length

    if "coord" in outputs and ("target_xy" in targets or "block_xy" in targets) and float(loss_weights["coord"]) != 0.0:
        target_xy = targets.get("target_xy", targets.get("block_xy"))
        assert target_xy is not None
        coord = weak_coord_loss(
            outputs["coord"],
            target_xy.to(device=device),
            target_tokens=targets.get("tokens"),
            token_mask=targets.get("token_mask"),
            eos_token_id=eos_token_id,
            ignore_index=ignore_index,
        )
        result["coord_loss"] = coord
        total = total + float(loss_weights["coord"]) * coord

    aux_specs: list[tuple[str, str, str]] = [
        ("joint_loss", "pred_joint", "joint"),
        ("ee_pose_loss", "pred_ee_pose", "ee_pose"),
        ("ee_xy_loss", "pred_ee_xy", "ee_xy"),
    ]
    for loss_key, output_key, target_key in aux_specs:
        weight_key = loss_key.removesuffix("_loss")
        if output_key not in outputs or target_key not in targets or float(loss_weights[weight_key]) == 0.0:
            continue
        aux = pose_auxiliary_loss(outputs[output_key], targets[target_key].to(device=device), mask=frame_mask)
        result[loss_key] = aux
        total = total + float(loss_weights[weight_key]) * aux

    result["loss"] = total
    return result


def _batch_scalar_int(batch: Mapping[str, Any], key: str, default: int) -> int:
    value: Any = batch.get(key, default)
    metadata = batch.get("metadata")
    if isinstance(metadata, Mapping) and key in metadata:
        value = metadata[key]
    if torch.is_tensor(value):
        return int(value.flatten()[0].item())
    return int(value)


def compute_stage1_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
) -> Tensor:
    """Stage 1 auxiliary reconstruction objective over presentation frames."""

    targets = batch["targets"]
    frame_mask = batch["model_inputs"]["frame_mask"]
    loss_weights = {"joint": 1.0, "ee_pose": 0.5, "ee_xy": 0.5, **dict(weights or {})}
    losses: list[Tensor] = []
    for output_key, target_key in (
        ("pred_joint", "joint"),
        ("joint", "joint"),
        ("pred_ee_pose", "ee_pose"),
        ("ee_pose", "ee_pose"),
        ("pred_ee_xy", "ee_xy"),
        ("ee_xy", "ee_xy"),
    ):
        if output_key in outputs and target_key in targets:
            loss = pose_auxiliary_loss(outputs[output_key], targets[target_key], mask=frame_mask)
            losses.append(float(loss_weights[target_key]) * loss)
    if not losses:
        raise RuntimeError("Stage 1 loss requires joint, ee_pose, or ee_xy auxiliary outputs.")
    return sum(losses)


def compute_stage2_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
) -> Tensor:
    """Stage 2 autonomous recall objective, including weak coordinate and auxiliary terms."""

    loss_outputs = dict(outputs)
    if "logits" not in loss_outputs and "token_logits" in loss_outputs:
        loss_outputs["logits"] = loss_outputs["token_logits"]
    losses = combined_v2_loss(
        loss_outputs,
        batch["targets"],
        frame_mask=batch["model_inputs"].get("frame_mask"),
        weights=weights,
        ignore_index=_batch_scalar_int(batch, "ignore_index", -100),
        eos_token_id=_batch_scalar_int(batch, "eos_token_id", 9),
    )
    return losses["loss"]
