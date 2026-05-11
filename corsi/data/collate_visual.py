"""Batch collation for variable-length robosuite visual Corsi data."""

from __future__ import annotations

from typing import Dict, List, Sequence


def collate_visual_batch(
    batch: Sequence[Dict[str, object]],
    *,
    pad_value: float = 0.0,
    target_pad_value: int = -100,
) -> Dict[str, object]:
    """Pads a visual batch into `[B, T, C, H, W]` tensors."""

    if not batch:
        raise ValueError("batch must not be empty")

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "collate_visual_batch requires PyTorch. Install torch before training."
        ) from exc

    frame_lengths = torch.tensor(
        [int(sample.get("frame_length", sample["frames"].shape[0])) for sample in batch],
        dtype=torch.long,
    )
    original_frame_lengths = torch.tensor(
        [int(sample.get("original_frame_length", sample["frames"].shape[0])) for sample in batch],
        dtype=torch.long,
    )
    target_lengths = torch.tensor(
        [int(sample.get("target_length", sample["length"])) for sample in batch],
        dtype=torch.long,
    )

    max_frame_length = int(frame_lengths.max().item())
    max_target_length = int(target_lengths.max().item())

    sample_frames = batch[0]["frames"]
    if len(sample_frames.shape) != 4:
        raise ValueError("Expected frames with shape [T, H, W, C]")
    _, height, width, channels = sample_frames.shape

    frames_pad = torch.full(
        (len(batch), max_frame_length, channels, height, width),
        float(pad_value),
        dtype=torch.float32,
    )
    targets_pad = torch.full((len(batch), max_target_length), target_pad_value, dtype=torch.long)
    sample_heatmaps = batch[0].get("target_heatmaps")
    target_heatmaps_pad = None
    if sample_heatmaps is not None:
        heatmap_height, heatmap_width = sample_heatmaps.shape[-2:]
        target_heatmaps_pad = torch.zeros(
            (len(batch), max_target_length, heatmap_height, heatmap_width),
            dtype=torch.float32,
        )
    has_target_xy = batch[0].get("target_xy") is not None
    has_target_block_xy = batch[0].get("target_block_xy_norm") is not None
    has_ee_xy = batch[0].get("ee_xy_norm") is not None
    has_ee_xyz = batch[0].get("ee_xyz_world") is not None
    has_target_block_indices = batch[0].get("target_block_indices") is not None
    target_xy_pad = (
        torch.zeros((len(batch), max_target_length, 2), dtype=torch.float32)
        if has_target_xy
        else None
    )
    target_block_xy_pad = (
        torch.zeros((len(batch), max_target_length, 2), dtype=torch.float32)
        if has_target_block_xy
        else None
    )
    ee_xy_pad = (
        torch.zeros((len(batch), max_target_length, 2), dtype=torch.float32)
        if has_ee_xy
        else None
    )
    ee_xyz_pad = (
        torch.zeros((len(batch), max_target_length, 3), dtype=torch.float32)
        if has_ee_xyz
        else None
    )
    target_block_indices_pad = (
        torch.full((len(batch), max_target_length), target_pad_value, dtype=torch.long)
        if has_target_block_indices
        else None
    )
    mask = torch.zeros((len(batch), max_target_length), dtype=torch.bool)

    trial_ids: List[str] = []
    camera_names: List[str] = []
    frame_paths: List[object] = []
    reset_paths: List[str] = []
    metadata: List[object] = []
    step_metadata: List[object] = []
    xy_normalization: List[object] = []
    block_xy_norm: List[object] = []

    for batch_index, sample in enumerate(batch):
        sample_frame_length = int(frame_lengths[batch_index].item())
        sample_target_length = int(target_lengths[batch_index].item())
        frames = torch.tensor(sample["frames"], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        targets = torch.tensor(sample["targets"], dtype=torch.long)

        frames_pad[batch_index, :sample_frame_length] = frames[:sample_frame_length]
        targets_pad[batch_index, :sample_target_length] = targets[:sample_target_length]
        if target_heatmaps_pad is not None:
            heatmaps = torch.tensor(sample["target_heatmaps"], dtype=torch.float32)
            target_heatmaps_pad[batch_index, :sample_target_length] = heatmaps[:sample_target_length]
        if target_xy_pad is not None:
            target_xy = torch.tensor(sample["target_xy"], dtype=torch.float32)
            target_xy_pad[batch_index, :sample_target_length] = target_xy[:sample_target_length]
        if target_block_xy_pad is not None:
            target_block_xy = torch.tensor(sample["target_block_xy_norm"], dtype=torch.float32)
            target_block_xy_pad[batch_index, :sample_target_length] = target_block_xy[:sample_target_length]
        if ee_xy_pad is not None:
            ee_xy = torch.tensor(sample["ee_xy_norm"], dtype=torch.float32)
            ee_xy_pad[batch_index, :sample_target_length] = ee_xy[:sample_target_length]
        if ee_xyz_pad is not None:
            ee_xyz = torch.tensor(sample["ee_xyz_world"], dtype=torch.float32)
            ee_xyz_pad[batch_index, :sample_target_length] = ee_xyz[:sample_target_length]
        if target_block_indices_pad is not None:
            indices = torch.tensor(sample["target_block_indices"], dtype=torch.long)
            target_block_indices_pad[batch_index, :sample_target_length] = indices[:sample_target_length]
        mask[batch_index, :sample_target_length] = True

        trial_ids.append(str(sample["trial_id"]))
        camera_names.append(str(sample["camera_name"]))
        frame_paths.append(sample["frame_paths"])
        reset_paths.append(str(sample["reset_path"]))
        metadata.append(sample["metadata"])
        step_metadata.append(sample.get("step_metadata", []))
        xy_normalization.append(sample.get("xy_normalization", {}))
        block_xy_norm.append(sample.get("block_xy_norm", {}))

    collated = {
        "frames": frames_pad,
        "targets": targets_pad,
        "frame_lengths": frame_lengths,
        "original_frame_lengths": original_frame_lengths,
        "target_lengths": target_lengths,
        "target_heatmaps": target_heatmaps_pad,
        "mask": mask,
        "trial_ids": trial_ids,
        "camera_names": camera_names,
        "frame_paths": frame_paths,
        "reset_paths": reset_paths,
        "metadata": metadata,
        "step_metadata": step_metadata,
        "xy_normalization": xy_normalization,
        "block_xy_norm": block_xy_norm,
        # Backward-compatible aliases for older callers.
        "frames_pad": frames_pad,
        "targets_pad": targets_pad,
    }
    if target_xy_pad is not None:
        collated["target_xy"] = target_xy_pad
    if target_block_xy_pad is not None:
        collated["target_block_xy_norm"] = target_block_xy_pad
    if ee_xy_pad is not None:
        collated["ee_xy_norm"] = ee_xy_pad
    if ee_xyz_pad is not None:
        collated["ee_xyz_world"] = ee_xyz_pad
    if target_block_indices_pad is not None:
        collated["target_block_indices"] = target_block_indices_pad
    return collated
