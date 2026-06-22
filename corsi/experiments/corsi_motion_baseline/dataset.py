"""Dataset and collation for canonical Corsi motion prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class CorsiMotionCanonicalDataset:
    """Loads immutable canonical 7-joint motion episodes."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str | None = None,
        normalize: bool = True,
        seq_ids: Sequence[str] | None = None,
        load_images: bool = True,
        cache: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.root = Path(str(self.manifest["canonical_root"]))
        self.normalize = bool(normalize)
        self.load_images = bool(load_images)
        self.cache = bool(cache)
        requested_ids = set(seq_ids) if seq_ids is not None else None
        if split is not None:
            split_ids = set(self.manifest["split"][split])
            requested_ids = split_ids if requested_ids is None else requested_ids & split_ids
        samples = list(self.manifest["samples"])
        if requested_ids is not None:
            samples = [sample for sample in samples if str(sample["seq_id"]) in requested_ids]
        self.samples = sorted(samples, key=lambda item: str(item["seq_id"]))
        stats = self.manifest["normalization"]
        self.image_mean = np.asarray(stats["image_mean"], dtype=np.float32).reshape(3, 1, 1)
        self.image_std = np.asarray(stats["image_std"], dtype=np.float32).reshape(3, 1, 1)
        self.joint_mean = np.asarray(stats["joint_mean"], dtype=np.float32)
        self.joint_std = np.asarray(stats["joint_std"], dtype=np.float32)
        self._cache: list[dict[str, Any] | None] = [None] * len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_arrays(self, index: int) -> dict[str, Any]:
        if self.cache and self._cache[index] is not None:
            return dict(self._cache[index])
        sample = self.samples[index]
        path = Path(str(sample["canonical_path"]))
        if not path.is_absolute() and not path.exists():
            path = self.root / "episodes" / path.name
        with np.load(path) as arrays:
            images = arrays["images"].copy() if self.load_images else None
            joints = arrays["joints"].astype(np.float32)
            transition_mask = arrays["transition_mask"].astype(bool)
            within_segment_transition = arrays["within_segment_transition"].astype(bool)
            segment_boundary_transition = arrays["segment_boundary_transition"].astype(bool)
            rank = arrays["rank"].astype(np.int64)
            block_id = arrays["block_id"].astype(np.int64)
            block_xy = arrays["block_xy"].astype(np.float32)
            segment_progress = arrays["segment_progress"].astype(np.float32)
            source_frame_index = arrays["source_frame_index"].astype(np.int64)
            source_timestamp = arrays["source_timestamp"].astype(np.float64)
            boundary = arrays["boundary"].astype(bool)
        loaded = {
            "images": images,
            "joints": joints,
            "transition_mask": transition_mask,
            "within_segment_transition": within_segment_transition,
            "segment_boundary_transition": segment_boundary_transition,
            "rank": rank,
            "block_id": block_id,
            "block_xy": block_xy,
            "segment_progress": segment_progress,
            "source_frame_index": source_frame_index,
            "source_timestamp": source_timestamp,
            "boundary": boundary,
        }
        if self.cache:
            self._cache[index] = dict(loaded)
        return loaded

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        loaded = self._load_arrays(index)
        images = loaded["images"]
        if images is not None:
            images = images.astype(np.float32) / 255.0
        joints = loaded["joints"].astype(np.float32)
        physical_joints = joints.copy()
        if self.normalize:
            if images is not None:
                images = (images - self.image_mean) / self.image_std
            joints = (joints - self.joint_mean) / self.joint_std
        return {
            "seq_id": str(sample["seq_id"]),
            "split": "",
            "length": int(sample["length"]),
            "block_order": list(sample["block_order"]),
            "images": None if images is None else images.astype(np.float32),
            "joints": joints.astype(np.float32),
            "physical_joints": physical_joints.astype(np.float32),
            "transition_mask": loaded["transition_mask"],
            "within_segment_transition": loaded["within_segment_transition"],
            "segment_boundary_transition": loaded["segment_boundary_transition"],
            "rank": loaded["rank"],
            "block_id": loaded["block_id"],
            "block_xy": loaded["block_xy"],
            "segment_progress": loaded["segment_progress"],
            "source_frame_index": loaded["source_frame_index"],
            "source_timestamp": loaded["source_timestamp"],
            "boundary": loaded["boundary"],
        }


def collate_motion_prediction_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch must not be empty")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("collate_motion_prediction_batch requires PyTorch") from exc

    batch_size = len(batch)
    lengths = torch.tensor([int(item["joints"].shape[0]) for item in batch], dtype=torch.long)
    max_steps = int(lengths.max().item())
    joint_dim = int(batch[0]["joints"].shape[1])
    has_images = batch[0]["images"] is not None
    images = torch.zeros((batch_size, max_steps, 3, 128, 128), dtype=torch.float32) if has_images else None
    joints = torch.zeros((batch_size, max_steps, joint_dim), dtype=torch.float32)
    physical_joints = torch.zeros((batch_size, max_steps, joint_dim), dtype=torch.float32)
    targets_next = torch.zeros((batch_size, max_steps, joint_dim), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, max_steps), dtype=torch.bool)
    loss_mask = torch.zeros((batch_size, max_steps), dtype=torch.bool)
    within_segment_transition = torch.zeros((batch_size, max_steps), dtype=torch.bool)
    segment_boundary_transition = torch.zeros((batch_size, max_steps), dtype=torch.bool)
    rank = torch.full((batch_size, max_steps), -1, dtype=torch.long)
    block_id = torch.full((batch_size, max_steps), -1, dtype=torch.long)
    block_xy = torch.zeros((batch_size, max_steps, 2), dtype=torch.float32)
    segment_progress = torch.zeros((batch_size, max_steps), dtype=torch.float32)
    source_frame_index = torch.full((batch_size, max_steps), -1, dtype=torch.long)
    source_timestamp = torch.zeros((batch_size, max_steps), dtype=torch.float64)
    boundary = torch.zeros((batch_size, max_steps), dtype=torch.bool)

    for batch_index, item in enumerate(batch):
        steps = int(item["joints"].shape[0])
        if has_images:
            images[batch_index, :steps] = torch.as_tensor(item["images"], dtype=torch.float32)
        joints[batch_index, :steps] = torch.as_tensor(item["joints"], dtype=torch.float32)
        physical_joints[batch_index, :steps] = torch.as_tensor(item["physical_joints"], dtype=torch.float32)
        if steps > 1:
            targets_next[batch_index, : steps - 1] = joints[batch_index, 1:steps]
        valid_mask[batch_index, :steps] = True
        loss_mask[batch_index, :steps] = torch.as_tensor(item["transition_mask"], dtype=torch.bool)
        within_segment_transition[batch_index, :steps] = torch.as_tensor(
            item["within_segment_transition"], dtype=torch.bool
        )
        segment_boundary_transition[batch_index, :steps] = torch.as_tensor(
            item["segment_boundary_transition"], dtype=torch.bool
        )
        rank[batch_index, :steps] = torch.as_tensor(item["rank"], dtype=torch.long)
        block_id[batch_index, :steps] = torch.as_tensor(item["block_id"], dtype=torch.long)
        block_xy[batch_index, :steps] = torch.as_tensor(item["block_xy"], dtype=torch.float32)
        segment_progress[batch_index, :steps] = torch.as_tensor(item["segment_progress"], dtype=torch.float32)
        source_frame_index[batch_index, :steps] = torch.as_tensor(item["source_frame_index"], dtype=torch.long)
        source_timestamp[batch_index, :steps] = torch.as_tensor(item["source_timestamp"], dtype=torch.float64)
        boundary[batch_index, :steps] = torch.as_tensor(item["boundary"], dtype=torch.bool)

    return {
        "seq_id": [item["seq_id"] for item in batch],
        "length": torch.tensor([int(item["length"]) for item in batch], dtype=torch.long),
        "block_order": [item["block_order"] for item in batch],
        "images": images,
        "joints": joints,
        "physical_joints": physical_joints,
        "targets_next": targets_next,
        "valid_mask": valid_mask,
        "loss_mask": loss_mask,
        "within_segment_transition": within_segment_transition,
        "segment_boundary_transition": segment_boundary_transition,
        "rank": rank,
        "block_id": block_id,
        "block_xy": block_xy,
        "segment_progress": segment_progress,
        "source_frame_index": source_frame_index,
        "source_timestamp": source_timestamp,
        "boundary": boundary,
        "lengths": lengths,
    }
