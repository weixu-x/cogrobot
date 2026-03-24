"""Batch collation for variable-length coordinate Corsi data."""

from __future__ import annotations

from typing import Dict, List, Sequence


def collate_coordinate_batch(
    batch: Sequence[Dict[str, object]],
    *,
    pad_value: float = 0.0,
    target_pad_value: int = -100,
) -> Dict[str, object]:
    """Pads a coordinate batch and returns tensors when torch is available."""

    if not batch:
        raise ValueError("batch must not be empty")

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "collate_coordinate_batch requires PyTorch. Install torch before training."
        ) from exc

    lengths = torch.tensor([int(sample["length"]) for sample in batch], dtype=torch.long)
    max_length = int(lengths.max().item())
    feature_dim = len(batch[0]["coords"][0]) if batch[0]["coords"] else 0

    coords_pad = torch.full((len(batch), max_length, feature_dim), pad_value, dtype=torch.float32)
    targets_pad = torch.full((len(batch), max_length), target_pad_value, dtype=torch.long)
    mask = torch.zeros((len(batch), max_length), dtype=torch.bool)

    trial_ids: List[str] = []
    modes: List[str] = []
    layouts: List[object] = []

    for batch_index, sample in enumerate(batch):
        sample_length = int(sample["length"])
        coords = torch.tensor(sample["coords"], dtype=torch.float32)
        targets = torch.tensor(sample["targets"], dtype=torch.long)

        coords_pad[batch_index, :sample_length] = coords
        targets_pad[batch_index, :sample_length] = targets
        mask[batch_index, :sample_length] = True

        trial_ids.append(str(sample["trial_id"]))
        modes.append(str(sample["mode"]))
        layouts.append(sample["layout"])

    return {
        "coords_pad": coords_pad,
        "targets_pad": targets_pad,
        "lengths": lengths,
        "mask": mask,
        "trial_ids": trial_ids,
        "modes": modes,
        "layouts": layouts,
    }
