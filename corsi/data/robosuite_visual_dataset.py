"""Dataset helpers for offscreen robosuite Corsi frame sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import imageio.v2 as imageio
import numpy as np


class RobosuiteVisualCorsiDataset:
    """Loads pre-exported robosuite Corsi samples from a dataset root."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        camera_name: Optional[str] = None,
        include_reset_frame: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.camera_name = camera_name
        self.include_reset_frame = bool(include_reset_frame)

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        root_manifest_path = self.dataset_root / "dataset_manifest.json"
        if not root_manifest_path.exists():
            raise FileNotFoundError(
                f"Missing dataset manifest at {root_manifest_path}. "
                "Run export_robosuite_visual_dataset.py first."
            )

        self.root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        self.samples = list(self.root_manifest.get("samples", []))
        if not self.samples:
            raise ValueError(f"No samples found in {root_manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_camera_name(self, sample: Dict[str, object]) -> str:
        available_cameras = list(sample.get("camera_names", []))
        if not available_cameras:
            raise ValueError(f"Sample {sample.get('trial_id')} has no camera_names")

        if self.camera_name is None:
            return str(available_cameras[0])
        if self.camera_name not in available_cameras:
            raise ValueError(
                f"Requested camera '{self.camera_name}' is not available for sample "
                f"{sample.get('trial_id')}. Available: {available_cameras}"
            )
        return self.camera_name

    def _load_frame_stack(self, paths: Sequence[str]) -> np.ndarray:
        frames = [imageio.imread(Path(path)) for path in paths]
        if not frames:
            raise ValueError("Expected at least one frame path")
        return np.stack(frames, axis=0)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        camera_name = self._resolve_camera_name(sample)

        keyframe_paths_by_camera = sample.get("keyframe_paths", {})
        if camera_name not in keyframe_paths_by_camera:
            raise KeyError(
                f"Camera '{camera_name}' missing from keyframe_paths for sample {sample.get('trial_id')}"
            )

        frame_paths = list(keyframe_paths_by_camera[camera_name])
        reset_paths = sample.get("reset_paths", {})
        reset_path = str(reset_paths[camera_name])

        frames = self._load_frame_stack(frame_paths)
        if self.include_reset_frame:
            reset_frame = self._load_frame_stack([reset_path])
            frames = np.concatenate([reset_frame, frames], axis=0)

        return {
            "trial_id": str(sample["trial_id"]),
            "frames": frames,
            "targets": list(sample["sequence"]),
            "length": int(sample["length"]),
            "camera_name": camera_name,
            "frame_paths": frame_paths,
            "reset_path": reset_path,
            "manifest_path": str(sample["manifest_path"]),
            "metadata": dict(sample.get("metadata", {})),
            "include_reset_frame": self.include_reset_frame,
        }
