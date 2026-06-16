"""Dataset helpers for offscreen robosuite Corsi frame sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import imageio.v2 as imageio
import numpy as np

from corsi.heatmaps import sequence_to_target_heatmaps

REPO_ROOT = Path(__file__).resolve().parents[2]


class RobosuiteVisualCorsiDataset:
    """Loads pre-exported robosuite Corsi samples from a dataset root."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        camera_name: Optional[str] = None,
        include_reset_frame: bool = False,
        heatmap_size: int = 32,
        heatmap_sigma: float = 2.0,
        heatmap_normalize: bool = True,
        target_type: str = "block_index",
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.camera_name = camera_name
        self.include_reset_frame = bool(include_reset_frame)
        self.heatmap_size = int(heatmap_size)
        self.heatmap_sigma = float(heatmap_sigma)
        self.heatmap_normalize = bool(heatmap_normalize)
        self.target_type = str(target_type)
        if self.target_type not in {"block_index", "block_center_xy", "end_effector_xy"}:
            raise ValueError(
                "target_type must be one of: block_index, block_center_xy, end_effector_xy"
            )

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
        self.dataset_name = str(self.root_manifest.get("dataset_name", self.dataset_root.name))
        self.split_name = str(self.root_manifest.get("split_name", self.dataset_root.name))
        self.camera_names = list(self.root_manifest.get("camera_names", []))
        self.dataset_metadata = dict(self.root_manifest.get("export_params", {}))
        self.xy_normalization = dict(self.root_manifest.get("xy_normalization", {}))
        self.block_xy_norm = self._load_block_xy_norm()

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
        frames = [imageio.imread(self._resolve_frame_path(path)) for path in paths]
        if not frames:
            raise ValueError("Expected at least one frame path")
        return np.stack(frames, axis=0)

    def _resolve_frame_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.exists():
            return candidate

        if not candidate.is_absolute():
            repo_relative = REPO_ROOT / candidate
            if repo_relative.exists():
                return repo_relative
            root_relative = self.dataset_root / candidate
            if root_relative.exists():
                return root_relative

        parts = candidate.parts
        if self.dataset_root.name in parts:
            root_index = parts.index(self.dataset_root.name)
            root_relative = self.dataset_root.joinpath(*parts[root_index + 1 :])
            if root_relative.exists():
                return root_relative

        if "corsi_artifacts" in parts:
            artifact_index = parts.index("corsi_artifacts")
            artifact_relative = REPO_ROOT.joinpath(*parts[artifact_index:])
            if artifact_relative.exists():
                return artifact_relative

        return candidate

    def _load_block_xy_norm(self) -> Dict[int, list[float]]:
        raw_positions = self.root_manifest.get("debug_block_xy_norm", {})
        positions: Dict[int, list[float]] = {}
        if isinstance(raw_positions, dict):
            for key, value in raw_positions.items():
                positions[int(key)] = [float(value[0]), float(value[1])]
        return positions

    def _load_step_metadata(self, sample: Dict[str, object]) -> list[dict]:
        raw_metadata = sample.get("step_metadata")
        if isinstance(raw_metadata, list):
            return [dict(item) for item in raw_metadata]

        manifest_path_text = str(sample.get("manifest_path", ""))
        if not manifest_path_text:
            return []
        manifest_path = self._resolve_frame_path(manifest_path_text)
        if not manifest_path.exists():
            return []
        trial_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_metadata = trial_manifest.get("step_metadata", [])
        if not isinstance(raw_metadata, list):
            return []
        return [dict(item) for item in raw_metadata]

    def _step_metadata_arrays(
        self,
        *,
        sample: Dict[str, object],
        target_length: int,
    ) -> Dict[str, object]:
        step_metadata = self._load_step_metadata(sample)
        if not step_metadata:
            if self.target_type in {"block_center_xy", "end_effector_xy"}:
                raise ValueError(
                    f"Sample {sample.get('trial_id')} has no step_metadata, "
                    f"required for target_type={self.target_type!r}"
                )
            return {"step_metadata": []}
        if len(step_metadata) != target_length:
            raise ValueError(
                f"Sample {sample.get('trial_id')} step_metadata length {len(step_metadata)} "
                f"does not match sequence length {target_length}"
            )

        def require_xy(field: str) -> np.ndarray:
            values = []
            for step_index, metadata in enumerate(step_metadata):
                if field not in metadata:
                    raise ValueError(
                        f"Sample {sample.get('trial_id')} step {step_index} is missing {field}"
                    )
                values.append(metadata[field])
            array = np.asarray(values, dtype=np.float32)
            if array.shape != (target_length, 2):
                raise ValueError(
                    f"Sample {sample.get('trial_id')} field {field} has shape {array.shape}, "
                    f"expected ({target_length}, 2)"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"Sample {sample.get('trial_id')} field {field} contains non-finite values")
            return array

        target_block_xy_norm = require_xy("target_block_xy_norm")
        ee_xy_norm = require_xy("ee_xy_norm")
        ee_xyz_world = np.asarray(
            [metadata.get("ee_xyz_world", [np.nan, np.nan, np.nan]) for metadata in step_metadata],
            dtype=np.float32,
        )
        target_block_indices = np.asarray(
            [int(metadata.get("block_index", sample["sequence"][index])) for index, metadata in enumerate(step_metadata)],
            dtype=np.int64,
        )
        if self.target_type == "block_center_xy":
            target_xy = target_block_xy_norm
        elif self.target_type == "end_effector_xy":
            target_xy = ee_xy_norm
        else:
            target_xy = None

        return {
            "step_metadata": step_metadata,
            "target_block_xy_norm": target_block_xy_norm,
            "ee_xy_norm": ee_xy_norm,
            "ee_xyz_world": ee_xyz_world,
            "target_block_indices": target_block_indices,
            "target_xy": target_xy,
        }

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

        target_length = int(sample["length"])
        frame_length = int(frames.shape[0])
        targets = list(sample["sequence"])
        target_heatmaps = sequence_to_target_heatmaps(
            targets,
            size=self.heatmap_size,
            sigma=self.heatmap_sigma,
            normalize=self.heatmap_normalize,
        )
        metadata_payload = self._step_metadata_arrays(
            sample=sample,
            target_length=target_length,
        )

        item = {
            "dataset_name": self.dataset_name,
            "split_name": self.split_name,
            "trial_id": str(sample["trial_id"]),
            "frames": frames,
            "targets": targets,
            "target_heatmaps": target_heatmaps,
            "length": target_length,
            "target_length": target_length,
            "frame_length": frame_length,
            "original_frame_length": frame_length,
            "camera_name": camera_name,
            "frame_paths": frame_paths,
            "reset_path": reset_path,
            "manifest_path": str(sample["manifest_path"]),
            "root_manifest_path": str(self.dataset_root / "dataset_manifest.json"),
            "metadata": dict(sample.get("metadata", {})),
            "xy_normalization": self.xy_normalization,
            "block_xy_norm": self.block_xy_norm,
            "target_type": self.target_type,
            "include_reset_frame": self.include_reset_frame,
        }
        item.update(metadata_payload)
        return item
