"""Reusable robosuite helpers for Corsi pointing demos and smoke tests."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import imageio
import numpy as np

import robosuite as suite
import robosuite.macros as macros
import robosuite.environments.manipulation.lift_corsi_min
import robosuite.environments.manipulation.lift_with_corsi_arena
from robosuite.controllers import load_composite_controller_config
from robosuite.models.grippers import InspireRightHand, register_gripper
from robosuite.models.robots import Panda
from robosuite.robots import register_robot_class

DEFAULT_ONLINE_RENDER_CAMERA = "frontview"
DEFAULT_OFFLINE_CAMERAS = ["frontview", "birdview"]
DEFAULT_GRIPPER_ACTION = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
DEFAULT_GRIPPER_ACTION = np.clip(DEFAULT_GRIPPER_ACTION, 0.0, 1.0)

macros.IMAGE_CONVENTION = "opencv"


@register_gripper
class InspireRightHandWithInit(InspireRightHand):
    """Inspire hand variant with a fixed pointing-friendly initialization."""

    CTRL_MAX = np.array(
        [
            1.87,
            1.62,
            1.75,
            1.62,
            1.82,
            1.62,
            1.82,
            1.62,
            0.77,
            0.68,
            0.68,
            1.30,
        ],
        dtype=np.float32,
    )
    INDEX_OPEN_GAIN = 1.5
    INDEX_OPEN_BIAS = -0.5
    INDICES = np.array([5, 5, 3, 3, 2, 2, 1, 1, 4, 4, 4, 0], dtype=np.int32)

    def format_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (6,)
        action = np.clip(action, 0.0, 1.0)

        control = action[self.INDICES].copy()
        index_distal = 6
        index_prox = 7
        control[index_distal] = np.clip(
            control[index_distal] * self.INDEX_OPEN_GAIN + self.INDEX_OPEN_BIAS,
            0.0,
            1.0,
        )
        control[index_prox] = np.clip(
            control[index_prox] * self.INDEX_OPEN_GAIN + self.INDEX_OPEN_BIAS,
            0.0,
            1.0,
        )
        return control * 2.0 - 1.0

    @property
    def init_qpos(self):
        base = np.array([0.1, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        return base[self.INDICES] * self.CTRL_MAX


@register_robot_class("FixedBaseRobot")
class PandaDexRH(Panda):
    """Panda robot with the fixed pointing hand attached on the right arm."""

    @property
    def default_gripper(self):
        return {"right": "InspireRightHandWithInit"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [-0.5, 0.5, 0.5, -0.5]}


def default_save_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("ROBOSUITE_SAVE_DIR", repo_root / "robosuite" / "savevideo"))


def create_env(
    *,
    render_mode: str = "online",
    env_name: str = "CorsiSceneDemo",
    online_render_camera: str = DEFAULT_ONLINE_RENDER_CAMERA,
    offline_cameras: Optional[Sequence[str]] = None,
    control_freq: int = 20,
):
    online = render_mode == "online"
    camera_names = online_render_camera if online else list(offline_cameras or DEFAULT_OFFLINE_CAMERAS)
    camera_heights = 512 if online else [512] * len(camera_names)
    camera_widths = 512 if online else [512] * len(camera_names)
    return suite.make(
        env_name=env_name,
        robots="PandaDexRH",
        controller_configs=load_composite_controller_config(controller="BASIC"),
        has_renderer=online,
        has_offscreen_renderer=not online,
        render_camera=online_render_camera,
        use_camera_obs=not online,
        camera_names=camera_names,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        control_freq=control_freq,
        ignore_done=True,
    )


def resolve_block_names(env) -> List[str]:
    if hasattr(env.model, "mujoco_arena") and hasattr(env.model.mujoco_arena, "block_names"):
        return list(env.model.mujoco_arena.block_names)
    return [name for name in env.sim.model.body_names if name.startswith("corsi_block_")]


def normalize_block_sequence(block_sequence: Sequence[int], num_blocks: int) -> List[int]:
    if not block_sequence:
        raise ValueError("block_sequence must not be empty")
    sequence = [int(block_id) for block_id in block_sequence]
    for block_id in sequence:
        if block_id < 0 or block_id >= num_blocks:
            raise ValueError(f"Block id {block_id} is outside the valid range [0, {num_blocks - 1}]")
    return sequence


def init_sequence_state(
    env,
    block_sequence: Sequence[int],
    *,
    target_height: float = 0.2,
    speed_gain: float = 0.1,
    arrival_threshold: float = 0.015,
    dwell_steps: int = 8,
    gripper_action: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    obs = env.reset()
    robot = env.robots[0]
    arm = robot.arms[0]
    gripper_name = robot.get_gripper_name(arm)
    arm_dim = robot.part_controllers[arm].control_dim
    block_names = resolve_block_names(env)
    sequence = normalize_block_sequence(block_sequence, len(block_names))

    return {
        "obs": obs,
        "robot": robot,
        "arm": arm,
        "gripper_name": gripper_name,
        "arm_dim": arm_dim,
        "block_names": block_names,
        "block_sequence": sequence,
        "sequence_position": 0,
        "dwell_counter": 0,
        "step": 0,
        "completed": False,
        "target_height": float(target_height),
        "speed_gain": float(speed_gain),
        "arrival_threshold": float(arrival_threshold),
        "dwell_steps": int(dwell_steps),
        "gripper_action": np.array(
            DEFAULT_GRIPPER_ACTION if gripper_action is None else gripper_action,
            dtype=np.float32,
        ),
    }


def current_target_name(state: Dict[str, object]) -> str:
    block_id = state["block_sequence"][state["sequence_position"]]
    return state["block_names"][block_id]


def step_pointing_policy(state: Dict[str, object]):
    robot = state["robot"]
    arm = state["arm"]
    arm_action = np.zeros(state["arm_dim"], dtype=np.float32)

    if state["completed"]:
        return robot.create_action_vector({arm: arm_action, state["gripper_name"]: state["gripper_action"]})

    target_name = current_target_name(state)
    target_pose_in_base = robot.pose_in_base_from_name(target_name)
    target_pos = target_pose_in_base[:3, 3] + np.array([0.0, 0.0, state["target_height"]], dtype=np.float32)
    current_pos = robot._hand_pos[arm]

    delta = target_pos - current_pos
    out_max = np.array(robot.part_controllers[arm].output_max[:3], dtype=np.float32)
    scaled = np.divide(delta, out_max, out=np.zeros_like(delta), where=out_max != 0)
    arm_action[:3] = np.clip(scaled * state["speed_gain"], -1.0, 1.0)

    if np.linalg.norm(delta) < state["arrival_threshold"]:
        if state["dwell_counter"] < state["dwell_steps"]:
            state["dwell_counter"] += 1
            arm_action[:3] = 0.0
        else:
            state["sequence_position"] += 1
            state["dwell_counter"] = 0
            if state["sequence_position"] >= len(state["block_sequence"]):
                state["completed"] = True
                arm_action[:3] = 0.0
    else:
        state["dwell_counter"] = 0

    return robot.create_action_vector({arm: arm_action, state["gripper_name"]: state["gripper_action"]})


def rollout_sequence_online(env, block_sequence: Sequence[int], **state_kwargs) -> Dict[str, object]:
    state = init_sequence_state(env, block_sequence, **state_kwargs)
    while not state["completed"]:
        action = step_pointing_policy(state)
        state["obs"], _, _, _ = env.step(action)
        env.render()
        state["step"] += 1
    return state


def rollout_sequence_offline(
    env,
    block_sequence: Sequence[int],
    *,
    video_path: Path,
    frames_dir: Path,
    fps: int = 20,
    keep_frames: bool = False,
    offline_cameras: Optional[Sequence[str]] = None,
    **state_kwargs,
) -> Dict[str, object]:
    cameras = list(offline_cameras or DEFAULT_OFFLINE_CAMERAS)
    frames_dir = Path(frames_dir)
    video_path = Path(video_path)
    shutil.rmtree(frames_dir, ignore_errors=True)
    if keep_frames:
        for camera_name in cameras:
            (frames_dir / camera_name).mkdir(parents=True, exist_ok=True)

    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_root = video_path.stem
    video_ext = video_path.suffix
    writers = {
        camera_name: imageio.get_writer(video_path.parent / f"{video_root}_{camera_name}{video_ext}", fps=fps)
        for camera_name in cameras
    }

    try:
        state = init_sequence_state(env, block_sequence, **state_kwargs)
        while not state["completed"]:
            action = step_pointing_policy(state)
            state["obs"], _, _, _ = env.step(action)
            for camera_name in cameras:
                obs_key = f"{camera_name}_image"
                if obs_key not in state["obs"]:
                    raise KeyError(f"Missing camera obs '{obs_key}'. Available keys: {list(state['obs'].keys())}")
                frame = state["obs"][obs_key]
                writers[camera_name].append_data(frame)
                if keep_frames:
                    imageio.imwrite(frames_dir / camera_name / f"frame_{state['step']:05d}.png", frame)
            state["step"] += 1
    finally:
        for camera_name, writer in writers.items():
            writer.close()
            print(f"Video saved to {video_path.parent / f'{video_root}_{camera_name}{video_ext}'}")

    if keep_frames:
        print(f"Frames kept at {frames_dir}")
    return state
