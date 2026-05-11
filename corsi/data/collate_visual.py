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

    lengths = torch.tensor([int(sample["length"]) for sample in batch], dtype=torch.long)
    max_length = int(lengths.max().item())

    sample_frames = batch[0]["frames"]
    if len(sample_frames.shape) != 4:
        raise ValueError("Expected frames with shape [T, H, W, C]")
    _, height, width, channels = sample_frames.shape

    frames_pad = torch.full(
        (len(batch), max_length, channels, height, width),
        float(pad_value),
        dtype=torch.float32,
    )
    targets_pad = torch.full((len(batch), max_length), target_pad_value, dtype=torch.long)
    sample_heatmaps = batch[0].get("target_heatmaps")
    target_heatmaps_pad = None
    if sample_heatmaps is not None:
        heatmap_height, heatmap_width = sample_heatmaps.shape[-2:]
        target_heatmaps_pad = torch.zeros(
            (len(batch), max_length, heatmap_height, heatmap_width),
            dtype=torch.float32,
        )
    mask = torch.zeros((len(batch), max_length), dtype=torch.bool)

    trial_ids: List[str] = []
    camera_names: List[str] = []
    frame_paths: List[object] = []
    reset_paths: List[str] = []
    metadata: List[object] = []

    for batch_index, sample in enumerate(batch):
        sample_length = int(sample["length"])
        frames = torch.tensor(sample["frames"], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        targets = torch.tensor(sample["targets"], dtype=torch.long)

        frames_pad[batch_index, :sample_length] = frames[:sample_length]
        targets_pad[batch_index, :sample_length] = targets
        if target_heatmaps_pad is not None:
            heatmaps = torch.tensor(sample["target_heatmaps"], dtype=torch.float32)
            target_heatmaps_pad[batch_index, :sample_length] = heatmaps[:sample_length]
        mask[batch_index, :sample_length] = True

        trial_ids.append(str(sample["trial_id"]))
        camera_names.append(str(sample["camera_name"]))
        frame_paths.append(sample["frame_paths"])
        reset_paths.append(str(sample["reset_path"]))
        metadata.append(sample["metadata"])

    return {
        "frames_pad": frames_pad,
        "targets_pad": targets_pad,
        "target_heatmaps": target_heatmaps_pad,
        "lengths": lengths,
        "mask": mask,
        # Current aliases used by the newer training path.
        "frames": frames_pad,
        "targets": targets_pad,
        "target_lengths": lengths,
        "frame_lengths": lengths,
        "trial_ids": trial_ids,
        "camera_names": camera_names,
        "frame_paths": frame_paths,
        "reset_paths": reset_paths,
        "metadata": metadata,
    }
