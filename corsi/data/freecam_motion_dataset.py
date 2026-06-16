"""freecam_motion_v1 dataset view over Corsi ee_xy manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from corsi.data.robosuite_visual_dataset import RobosuiteVisualCorsiDataset


class FreecamMotionDataset:
    """Loads keyframe images plus motion metadata from freecam ee_xy exports.

    Stage 0 keeps this as a lightweight view over the existing ee_xy manifest
    format. New exports include ``motion_state`` fields; older manifests still
    load and expose empty joint arrays so visual/xy workflows stay compatible.
    """

    motion_schema_version = "freecam_motion_v1"

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        camera_name: Optional[str] = "freecam",
        include_reset_frame: bool = False,
        require_joint_state: bool = False,
    ) -> None:
        self.visual_dataset = RobosuiteVisualCorsiDataset(
            dataset_root,
            camera_name=camera_name,
            include_reset_frame=include_reset_frame,
            target_type="end_effector_xy",
        )
        self.dataset_root = self.visual_dataset.dataset_root
        self.require_joint_state = bool(require_joint_state)

    def __len__(self) -> int:
        return len(self.visual_dataset)

    def _load_trial_manifest(self, sample: Dict[str, object]) -> Dict[str, object]:
        manifest_path_text = str(sample.get("manifest_path", ""))
        if not manifest_path_text:
            return {}
        manifest_path = self.visual_dataset._resolve_frame_path(manifest_path_text)
        if not manifest_path.exists():
            return {}
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def _load_trajectory_metadata(self, sample: Dict[str, object]) -> list[dict]:
        inline = sample.get("trajectory_metadata")
        if isinstance(inline, list):
            return [dict(item) for item in inline]
        trial_manifest = self._load_trial_manifest(sample)
        raw = trial_manifest.get("trajectory_metadata", [])
        if isinstance(raw, list):
            return [dict(item) for item in raw]
        return []

    @staticmethod
    def _motion_state(metadata: dict) -> dict:
        state = metadata.get("motion_state", {})
        return dict(state) if isinstance(state, dict) else {}

    def _trajectory_arrays(self, trajectory_metadata: list[dict]) -> dict[str, np.ndarray | str]:
        control_steps = np.asarray(
            [int(item.get("control_step", -1)) for item in trajectory_metadata],
            dtype=np.int64,
        )
        ee_xy_norm = np.asarray(
            [item.get("ee_xy_norm", [np.nan, np.nan]) for item in trajectory_metadata],
            dtype=np.float32,
        )
        target_block_indices = np.asarray(
            [
                -1 if item.get("target_block_index") is None else int(item.get("target_block_index", -1))
                for item in trajectory_metadata
            ],
            dtype=np.int64,
        )
        sequence_positions = np.asarray(
            [int(item.get("sequence_position", -1)) for item in trajectory_metadata],
            dtype=np.int64,
        )

        joint_rows = []
        action_rows = []
        source = "missing"
        for item in trajectory_metadata:
            motion_state = self._motion_state(item)
            if "arm_joint_qpos" in motion_state:
                joint_rows.append(motion_state["arm_joint_qpos"])
                source = str(motion_state.get("joint_position_source", "motion_state.arm_joint_qpos"))
            if "arm_action" in motion_state:
                action_rows.append(motion_state["arm_action"])

        joint_qpos = np.asarray(joint_rows, dtype=np.float32) if joint_rows else np.zeros((len(trajectory_metadata), 0), dtype=np.float32)
        arm_action = np.asarray(action_rows, dtype=np.float32) if action_rows else np.zeros((len(trajectory_metadata), 0), dtype=np.float32)
        if self.require_joint_state and joint_qpos.shape[1] == 0:
            raise ValueError("freecam_motion_v1 sample has no arm_joint_qpos motion_state fields")

        return {
            "trajectory_control_steps": control_steps,
            "trajectory_ee_xy_norm": ee_xy_norm,
            "trajectory_target_block_indices": target_block_indices,
            "trajectory_sequence_positions": sequence_positions,
            "trajectory_arm_joint_qpos": joint_qpos,
            "trajectory_arm_action": arm_action,
            "joint_position_source": source,
        }

    @staticmethod
    def _keyframe_rows(values: np.ndarray, trajectory_steps: np.ndarray, keyframe_steps: np.ndarray) -> np.ndarray:
        if values.ndim != 2 or values.shape[1] == 0:
            return np.zeros((len(keyframe_steps), 0), dtype=np.float32)
        rows = []
        index_by_step = {int(step): i for i, step in enumerate(trajectory_steps.tolist())}
        for step in keyframe_steps.tolist():
            row_index = index_by_step.get(int(step))
            if row_index is None:
                rows.append(np.full((values.shape[1],), np.nan, dtype=np.float32))
            else:
                rows.append(values[row_index])
        return np.asarray(rows, dtype=np.float32)

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = dict(self.visual_dataset[index])
        sample = self.visual_dataset.samples[index]
        trajectory_metadata = self._load_trajectory_metadata(sample)
        arrays = self._trajectory_arrays(trajectory_metadata)
        keyframe_steps = np.asarray(
            [int(metadata.get("control_step", -1)) for metadata in item.get("step_metadata", [])],
            dtype=np.int64,
        )
        target_xy = np.asarray(item["target_xy"], dtype=np.float32)
        previous_xy = np.concatenate([target_xy[:1], target_xy[:-1]], axis=0)

        item.update(arrays)
        item.update(
            {
                "motion_schema_version": self.motion_schema_version,
                "trajectory_metadata": trajectory_metadata,
                "keyframe_control_steps": keyframe_steps,
                "motion_target_xy": target_xy,
                "motion_delta_xy": target_xy - previous_xy,
                "keyframe_arm_joint_qpos": self._keyframe_rows(
                    item["trajectory_arm_joint_qpos"],
                    item["trajectory_control_steps"],
                    keyframe_steps,
                ),
                "keyframe_arm_action": self._keyframe_rows(
                    item["trajectory_arm_action"],
                    item["trajectory_control_steps"],
                    keyframe_steps,
                ),
            }
        )
        return item
