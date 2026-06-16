"""Batch collation for freecam_motion_v1 samples."""

from __future__ import annotations

from typing import Dict, Sequence

from corsi.data.collate_visual import collate_visual_batch


def collate_motion_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not batch:
        raise ValueError("batch must not be empty")

    try:
        import torch
    except ImportError as exc:
        raise ImportError("collate_motion_batch requires PyTorch") from exc

    collated = collate_visual_batch(batch)
    batch_size = len(batch)
    max_target_length = int(collated["target_lengths"].max().item())

    def pad_float_field(name: str, width: int):
        tensor = torch.zeros((batch_size, max_target_length, width), dtype=torch.float32)
        for batch_index, sample in enumerate(batch):
            values = torch.tensor(sample[name], dtype=torch.float32)
            tensor[batch_index, : values.shape[0], : values.shape[1]] = values
        return tensor

    collated["motion_target_xy"] = pad_float_field("motion_target_xy", 2)
    collated["motion_delta_xy"] = pad_float_field("motion_delta_xy", 2)

    joint_width = int(batch[0]["keyframe_arm_joint_qpos"].shape[1])
    action_width = int(batch[0]["keyframe_arm_action"].shape[1])
    if joint_width:
        collated["keyframe_arm_joint_qpos"] = pad_float_field("keyframe_arm_joint_qpos", joint_width)
    if action_width:
        collated["keyframe_arm_action"] = pad_float_field("keyframe_arm_action", action_width)
    collated["motion_schema_version"] = [sample["motion_schema_version"] for sample in batch]
    collated["joint_position_source"] = [sample["joint_position_source"] for sample in batch]
    return collated
