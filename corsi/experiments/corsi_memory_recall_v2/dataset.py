"""Dataset and collation for Corsi memory-recall V2 canonical data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from corsi.experiments.corsi_memory_recall_v2.canonicalize import (
    DEFAULT_IGNORE_INDEX,
    MODEL_INPUT_FIELDS,
    SCHEMA_VERSION,
)


class CorsiMemoryRecallV2Dataset:
    """Loads V2 canonical episodes as segmented RGB presentation samples."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str | None = None,
        seq_ids: Sequence[str] | None = None,
        normalize_images: bool = True,
        normalize_targets: bool = True,
        load_images: bool = True,
        cache: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"expected {SCHEMA_VERSION}, got {self.manifest.get('schema_version')!r}"
            )
        self.root = Path(str(self.manifest["canonical_root"]))
        self.normalize_images = bool(normalize_images)
        self.normalize_targets = bool(normalize_targets)
        self.load_images = bool(load_images)
        self.cache = bool(cache)
        requested_ids = set(str(seq_id) for seq_id in seq_ids) if seq_ids is not None else None
        if split is not None:
            split_ids = set(str(seq_id) for seq_id in self.manifest["split"][split])
            requested_ids = split_ids if requested_ids is None else requested_ids & split_ids
        samples = list(self.manifest["samples"])
        if requested_ids is not None:
            samples = [sample for sample in samples if str(sample["seq_id"]) in requested_ids]
        self.samples = sorted(samples, key=lambda item: str(item["seq_id"]))
        self.stats = self.manifest["normalization"]
        self.image_mean = np.asarray(self.stats["image_mean"], dtype=np.float32).reshape(1, 1, 3, 1, 1)
        self.image_std = np.asarray(self.stats["image_std"], dtype=np.float32).reshape(1, 1, 3, 1, 1)
        self._cache: list[dict[str, Any] | None] = [None] * len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def _episode_path(self, sample: dict[str, Any]) -> Path:
        path = Path(str(sample["canonical_path"]))
        if path.is_absolute() or path.exists():
            return path
        return self.root / "episodes" / path.name

    def _load_arrays(self, index: int) -> dict[str, Any]:
        if self.cache and self._cache[index] is not None:
            return dict(self._cache[index])
        sample = self.samples[index]
        with np.load(self._episode_path(sample)) as arrays:
            loaded = {
                "images": arrays["images"].copy() if self.load_images else None,
                "joint_targets": arrays["joint_targets"].astype(np.float32),
                "ee_pose_targets": arrays["ee_pose_targets"].astype(np.float32),
                "ee_xy_targets": arrays["ee_xy_targets"].astype(np.float32),
                "ee_xy_norm_targets": arrays["ee_xy_norm_targets"].astype(np.float32),
                "block_xy_targets": arrays["block_xy_targets"].astype(np.float32),
                "block_xyz_world_targets": arrays["block_xyz_world_targets"].astype(np.float32),
                "target_tokens": arrays["target_tokens"].astype(np.int64),
                "target_token_mask": arrays["target_token_mask"].astype(bool),
                "segment_mask": arrays["segment_mask"].astype(bool),
                "frame_mask": arrays["frame_mask"].astype(bool),
                "rank": arrays["rank"].astype(np.int64),
                "block_id": arrays["block_id"].astype(np.int64),
                "segment_progress": arrays["segment_progress"].astype(np.float32),
                "source_frame_index": arrays["source_frame_index"].astype(np.int64),
                "source_timestamp": arrays["source_timestamp"].astype(np.float64),
                "segment_start_frame": arrays["segment_start_frame"].astype(np.int64),
                "segment_end_frame": arrays["segment_end_frame"].astype(np.int64),
            }
        if self.cache:
            self._cache[index] = dict(loaded)
        return loaded

    def _standardize(self, field: str, values: np.ndarray) -> np.ndarray:
        if not self.normalize_targets:
            return values.astype(np.float32)
        mean = np.asarray(self.stats[f"{field}_mean"], dtype=np.float32)
        std = np.asarray(self.stats[f"{field}_std"], dtype=np.float32)
        shape = (1,) * (values.ndim - 1) + (values.shape[-1],)
        return ((values - mean.reshape(shape)) / std.reshape(shape)).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        loaded = self._load_arrays(index)
        images = loaded["images"]
        if images is not None:
            images = images.astype(np.float32) / 255.0
            if self.normalize_images:
                images = (images - self.image_mean) / self.image_std
            images = images.astype(np.float32)

        physical_joint = loaded["joint_targets"].astype(np.float32)
        physical_ee_pose = loaded["ee_pose_targets"].astype(np.float32)
        physical_ee_xy = loaded["ee_xy_targets"].astype(np.float32)
        ee_xy_norm = loaded["ee_xy_norm_targets"].astype(np.float32)
        block_xy = loaded["block_xy_targets"].astype(np.float32)

        return {
            "model_inputs": {
                "images": images,
                "segment_mask": loaded["segment_mask"],
                "frame_mask": loaded["frame_mask"],
            },
            "targets": {
                "tokens": loaded["target_tokens"],
                "token_mask": loaded["target_token_mask"],
                "target_xy": block_xy,
                "block_xy": block_xy,
                "joint": self._standardize("joint", physical_joint),
                "ee_pose": self._standardize("ee_pose", physical_ee_pose),
                "ee_xy": self._standardize("ee_xy", physical_ee_xy),
                "ee_xy_norm": ee_xy_norm,
                "physical_joint": physical_joint,
                "physical_ee_pose": physical_ee_pose,
                "physical_ee_xy": physical_ee_xy,
            },
            "metadata": {
                "seq_id": str(sample["seq_id"]),
                "length": int(sample["length"]),
                "block_order": [int(value) for value in sample["block_order"]],
                "raw_arrays_path": str(sample["raw_arrays_path"]),
                "raw_metadata_path": str(sample["raw_metadata_path"]),
                "raw_segments_path": str(sample["raw_segments_path"]),
                "rank": loaded["rank"],
                "block_id": loaded["block_id"],
                "block_xyz_world": loaded["block_xyz_world_targets"],
                "source_frame_index": loaded["source_frame_index"],
                "source_timestamp": loaded["source_timestamp"],
                "segment_progress": loaded["segment_progress"],
                "segment_start_frame": loaded["segment_start_frame"],
                "segment_end_frame": loaded["segment_end_frame"],
                "ignore_index": int(self.manifest.get("ignore_index", DEFAULT_IGNORE_INDEX)),
                "eos_token_id": int(self.manifest["eos_token_id"]),
            },
        }


def collate_memory_recall_v2_batch(
    batch: Sequence[dict[str, Any]],
    *,
    ignore_index: int | None = None,
) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch must not be empty")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("collate_memory_recall_v2_batch requires PyTorch") from exc

    batch_size = len(batch)
    if ignore_index is None:
        ignore_index = int(batch[0]["metadata"].get("ignore_index", DEFAULT_IGNORE_INDEX))
    lengths = torch.tensor([int(item["metadata"]["length"]) for item in batch], dtype=torch.long)
    max_segments = int(lengths.max().item())
    k = int(batch[0]["model_inputs"]["frame_mask"].shape[1])
    max_token_steps = max(int(item["targets"]["tokens"].shape[0]) for item in batch)
    has_images = batch[0]["model_inputs"]["images"] is not None
    images = None
    if has_images:
        _, _, channels, height, width = batch[0]["model_inputs"]["images"].shape
        images = torch.zeros(
            (batch_size, max_segments, k, channels, height, width),
            dtype=torch.float32,
        )

    segment_mask = torch.zeros((batch_size, max_segments), dtype=torch.bool)
    frame_mask = torch.zeros((batch_size, max_segments, k), dtype=torch.bool)
    tokens = torch.full((batch_size, max_token_steps), int(ignore_index), dtype=torch.long)
    token_mask = torch.zeros((batch_size, max_token_steps), dtype=torch.bool)

    block_xy = torch.zeros((batch_size, max_segments, 2), dtype=torch.float32)
    joint = torch.zeros((batch_size, max_segments, k, 7), dtype=torch.float32)
    ee_pose = torch.zeros((batch_size, max_segments, k, 7), dtype=torch.float32)
    ee_xy = torch.zeros((batch_size, max_segments, k, 2), dtype=torch.float32)
    ee_xy_norm = torch.zeros((batch_size, max_segments, k, 2), dtype=torch.float32)
    physical_joint = torch.zeros((batch_size, max_segments, k, 7), dtype=torch.float32)
    physical_ee_pose = torch.zeros((batch_size, max_segments, k, 7), dtype=torch.float32)
    physical_ee_xy = torch.zeros((batch_size, max_segments, k, 2), dtype=torch.float32)

    rank = torch.full((batch_size, max_segments), -1, dtype=torch.long)
    block_id = torch.full((batch_size, max_segments), -1, dtype=torch.long)
    block_xyz_world = torch.zeros((batch_size, max_segments, 3), dtype=torch.float32)
    source_frame_index = torch.full((batch_size, max_segments, k), -1, dtype=torch.long)
    source_timestamp = torch.zeros((batch_size, max_segments, k), dtype=torch.float64)
    segment_progress = torch.zeros((batch_size, max_segments, k), dtype=torch.float32)
    segment_start_frame = torch.full((batch_size, max_segments), -1, dtype=torch.long)
    segment_end_frame = torch.full((batch_size, max_segments), -1, dtype=torch.long)

    for batch_index, item in enumerate(batch):
        item_inputs = item["model_inputs"]
        item_targets = item["targets"]
        item_meta = item["metadata"]
        segment_count = int(item_meta["length"])
        token_count = int(item_targets["tokens"].shape[0])
        if has_images:
            images[batch_index, :segment_count] = torch.as_tensor(
                item_inputs["images"], dtype=torch.float32
            )
        segment_mask[batch_index, :segment_count] = torch.as_tensor(
            item_inputs["segment_mask"], dtype=torch.bool
        )
        frame_mask[batch_index, :segment_count] = torch.as_tensor(
            item_inputs["frame_mask"], dtype=torch.bool
        )
        tokens[batch_index, :token_count] = torch.as_tensor(item_targets["tokens"], dtype=torch.long)
        token_mask[batch_index, :token_count] = torch.as_tensor(
            item_targets["token_mask"], dtype=torch.bool
        )
        block_xy[batch_index, :segment_count] = torch.as_tensor(
            item_targets["block_xy"], dtype=torch.float32
        )
        joint[batch_index, :segment_count] = torch.as_tensor(item_targets["joint"], dtype=torch.float32)
        ee_pose[batch_index, :segment_count] = torch.as_tensor(
            item_targets["ee_pose"], dtype=torch.float32
        )
        ee_xy[batch_index, :segment_count] = torch.as_tensor(item_targets["ee_xy"], dtype=torch.float32)
        ee_xy_norm[batch_index, :segment_count] = torch.as_tensor(
            item_targets["ee_xy_norm"], dtype=torch.float32
        )
        physical_joint[batch_index, :segment_count] = torch.as_tensor(
            item_targets["physical_joint"], dtype=torch.float32
        )
        physical_ee_pose[batch_index, :segment_count] = torch.as_tensor(
            item_targets["physical_ee_pose"], dtype=torch.float32
        )
        physical_ee_xy[batch_index, :segment_count] = torch.as_tensor(
            item_targets["physical_ee_xy"], dtype=torch.float32
        )
        rank[batch_index, :segment_count] = torch.as_tensor(item_meta["rank"], dtype=torch.long)
        block_id[batch_index, :segment_count] = torch.as_tensor(item_meta["block_id"], dtype=torch.long)
        block_xyz_world[batch_index, :segment_count] = torch.as_tensor(
            item_meta["block_xyz_world"], dtype=torch.float32
        )
        source_frame_index[batch_index, :segment_count] = torch.as_tensor(
            item_meta["source_frame_index"], dtype=torch.long
        )
        source_timestamp[batch_index, :segment_count] = torch.as_tensor(
            item_meta["source_timestamp"], dtype=torch.float64
        )
        segment_progress[batch_index, :segment_count] = torch.as_tensor(
            item_meta["segment_progress"], dtype=torch.float32
        )
        segment_start_frame[batch_index, :segment_count] = torch.as_tensor(
            item_meta["segment_start_frame"], dtype=torch.long
        )
        segment_end_frame[batch_index, :segment_count] = torch.as_tensor(
            item_meta["segment_end_frame"], dtype=torch.long
        )

    return {
        "model_inputs": {
            "images": images,
            "segment_mask": segment_mask,
            "frame_mask": frame_mask,
        },
        "targets": {
            "tokens": tokens,
            "token_mask": token_mask,
            "target_xy": block_xy,
            "block_xy": block_xy,
            "joint": joint,
            "ee_pose": ee_pose,
            "ee_xy": ee_xy,
            "ee_xy_norm": ee_xy_norm,
            "physical_joint": physical_joint,
            "physical_ee_pose": physical_ee_pose,
            "physical_ee_xy": physical_ee_xy,
        },
        "metadata": {
            "seq_id": [str(item["metadata"]["seq_id"]) for item in batch],
            "length": lengths,
            "block_order": [list(item["metadata"]["block_order"]) for item in batch],
            "rank": rank,
            "block_id": block_id,
            "block_xyz_world": block_xyz_world,
            "source_frame_index": source_frame_index,
            "source_timestamp": source_timestamp,
            "segment_progress": segment_progress,
            "segment_start_frame": segment_start_frame,
            "segment_end_frame": segment_end_frame,
            "ignore_index": torch.tensor(
                [int(item["metadata"].get("ignore_index", ignore_index)) for item in batch],
                dtype=torch.long,
            ),
            "eos_token_id": torch.tensor(
                [int(item["metadata"]["eos_token_id"]) for item in batch],
                dtype=torch.long,
            ),
        },
        "ignore_index": int(ignore_index),
        "model_input_fields": list(MODEL_INPUT_FIELDS),
    }
