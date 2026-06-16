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
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.robots import Panda
from robosuite.robots import register_robot_class
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
from robosuite.utils.mjcf_utils import xml_path_completion
import mujoco

from corsi.envs.sequence_generator import (
    standard_corsi_robosuite_board_size,
    standard_corsi_robosuite_layout,
)

DEFAULT_ONLINE_RENDER_CAMERA = "frontview"
DEFAULT_OFFLINE_CAMERAS = ["frontview", "birdview"]
FREE_CAMERA_NAME = "freecam"
DEFAULT_CORSI_FREE_CAM_CONFIG = {
    "cam_config": {
        "lookat": [0.0, 0.0, 0.9],
        "distance": 0.489545,
        "azimuth": -179.858241,
        "elevation": -63.442729,
    }
}
DEFAULT_GRIPPER_ACTION = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
DEFAULT_GRIPPER_ACTION = np.clip(DEFAULT_GRIPPER_ACTION, 0.0, 1.0)
DEFAULT_GRIPPER_SETTLE_STEPS = 50

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


@register_gripper
class InspireRightHandLightWithInit(InspireRightHandWithInit):
    """Light-colored Inspire hand variant for contrast against black Corsi blocks."""

    def __init__(self, idn=0):
        GripperModel.__init__(self, xml_path_completion("grippers/inspire_right_hand_light.xml"), idn=idn)


@register_robot_class("FixedBaseRobot")
class PandaDexRH(Panda):
    """Panda robot with the fixed pointing hand attached on the right arm."""

    @property
    def default_gripper(self):
        return {"right": "InspireRightHandLightWithInit"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [-0.5, 0.5, 0.5, -0.5]}


def default_save_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("ROBOSUITE_SAVE_DIR", repo_root / "robosuite" / "savevideo"))


def _resolved_free_cam_config(cam_config: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    if cam_config is None:
        return dict(DEFAULT_CORSI_FREE_CAM_CONFIG["cam_config"])
    if "cam_config" in cam_config:
        return dict(cam_config["cam_config"])
    return dict(cam_config)


def render_tuned_free_camera_frame(
    env,
    *,
    width: int = 512,
    height: int = 512,
    cam_config: Optional[Dict[str, object]] = None,
):
    """Renders an offscreen frame using the tuned free-camera parameters."""

    if env.sim._render_context_offscreen is None:
        raise RuntimeError("Offscreen render context is not initialized")

    config = _resolved_free_cam_config(cam_config)
    render_context = env.sim._render_context_offscreen
    render_context.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    render_context.cam.lookat[:] = np.array(config["lookat"], dtype=np.float32)
    render_context.cam.distance = float(config["distance"])
    render_context.cam.azimuth = float(config["azimuth"])
    render_context.cam.elevation = float(config["elevation"])

    frame = env.sim.render(width=width, height=height, camera_name=None)
    convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    return frame[::convention]


def create_env(
    *,
    render_mode: str = "online",
    env_name: str = "CorsiSceneDemo",
    online_render_camera: Optional[str] = DEFAULT_ONLINE_RENDER_CAMERA,
    offline_cameras: Optional[Sequence[str]] = None,
    control_freq: int = 20,
    corsi_layout: Optional[Dict[int, tuple[float, float]]] = None,
    block_rgba_list: Optional[Sequence[Sequence[float]]] = None,
    renderer_config: Optional[Dict[str, object]] = None,
):
    online = render_mode == "online"
    camera_names = (
        (online_render_camera if online_render_camera is not None else DEFAULT_ONLINE_RENDER_CAMERA)
        if online
        else list(offline_cameras or DEFAULT_OFFLINE_CAMERAS)
    )
    camera_heights = 512 if online else [512] * len(camera_names)
    camera_widths = 512 if online else [512] * len(camera_names)
    renderer = "mjviewer" if online else "mujoco"
    resolved_renderer_config = dict(renderer_config) if renderer_config is not None else None
    if online and online_render_camera is None and resolved_renderer_config is None:
        resolved_renderer_config = DEFAULT_CORSI_FREE_CAM_CONFIG
    layout = corsi_layout or standard_corsi_robosuite_layout()
    board_size_xy = standard_corsi_robosuite_board_size()
    ordered_block_xy_positions = [layout[index] for index in sorted(layout)]
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
        block_xy_positions=ordered_block_xy_positions,
        corsi_board_size_xy=board_size_xy,
        block_rgba_list=block_rgba_list,
        renderer=renderer,
        renderer_config=resolved_renderer_config,
    )


def resolve_block_names(env) -> List[str]:
    if hasattr(env.model, "mujoco_arena") and hasattr(env.model.mujoco_arena, "block_names"):
        return list(env.model.mujoco_arena.block_names)
    return [name for name in env.sim.model.body_names if name.startswith("corsi_block_")]


def settled_gripper_action(
    env,
    *,
    steps: int = DEFAULT_GRIPPER_SETTLE_STEPS,
    gripper_action: Optional[np.ndarray] = None,
):
    """Steps the sim with a fixed gripper command so the hand reaches its pointing pose."""

    obs = None
    if steps <= 0:
        return obs

    robot = env.robots[0]
    arm = robot.arms[0]
    gripper_name = robot.get_gripper_name(arm)
    arm_dim = robot.part_controllers[arm].control_dim
    arm_action = np.zeros(arm_dim, dtype=np.float32)
    gripper_action = np.array(
        DEFAULT_GRIPPER_ACTION if gripper_action is None else gripper_action,
        dtype=np.float32,
    )
    action = robot.create_action_vector({arm: arm_action, gripper_name: gripper_action})
    for _ in range(int(steps)):
        obs, _, _, _ = env.step(action)
    return obs


def site_pose_in_base(robot, site_name: str) -> np.ndarray:
    """Returns a site pose expressed in the robot base frame."""

    site_id = robot.sim.model.site_name2id(site_name)
    pos_in_world = np.array(robot.sim.data.site_xpos[site_id], dtype=np.float32)
    rot_in_world = np.array(robot.sim.data.site_xmat[site_id], dtype=np.float32).reshape((3, 3))
    pose_in_world = T.make_pose(pos_in_world, rot_in_world)

    base_pos_in_world = robot.sim.data.get_body_xpos(robot.robot_model.root_body)
    base_rot_in_world = robot.sim.data.get_body_xmat(robot.robot_model.root_body).reshape((3, 3))
    base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
    world_pose_in_base = T.pose_inv(base_pose_in_world)
    return T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)


def site_pos_in_base(robot, site_name: str) -> np.ndarray:
    """Returns a site position expressed in the robot base frame."""

    return site_pose_in_base(robot, site_name)[:3, 3]


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
    target_height: float = 0.03,
    speed_gain: float = 0.1,
    arrival_threshold: float = 0.015,
    dwell_steps: int = 8,
    gripper_settle_steps: int = DEFAULT_GRIPPER_SETTLE_STEPS,
    gripper_action: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    obs = env.reset()
    resolved_gripper_action = np.array(
        DEFAULT_GRIPPER_ACTION if gripper_action is None else gripper_action,
        dtype=np.float32,
    )
    settled_obs = settled_gripper_action(
        env,
        steps=gripper_settle_steps,
        gripper_action=resolved_gripper_action,
    )
    if settled_obs is not None:
        obs = settled_obs
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
        "control_site_name": robot.gripper[arm].important_sites["grip_site"],
        "target_site_name": "gripper0_right_index_tip_site",
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
        "gripper_action": resolved_gripper_action,
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
    block_center_pos = robot.pose_in_base_from_name(target_name)[:3, 3]
    desired_tip_pos = block_center_pos + np.array([0.0, 0.0, state["target_height"]], dtype=np.float32)

    control_pos = site_pos_in_base(robot, state["control_site_name"])
    current_tip_pos = site_pos_in_base(robot, state["target_site_name"])
    control_to_tip_offset = current_tip_pos - control_pos

    target_pos = desired_tip_pos - control_to_tip_offset
    current_pos = control_pos

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
    write_videos: bool = True,
    keep_frames: bool = False,
    save_keyframes: bool = False,
    keyframes_dir: Optional[Path] = None,
    offline_cameras: Optional[Sequence[str]] = None,
    **state_kwargs,
) -> Dict[str, object]:
    cameras = list(offline_cameras or DEFAULT_OFFLINE_CAMERAS)
    frames_dir = Path(frames_dir)
    video_path = Path(video_path)
    keyframes_dir = Path(keyframes_dir) if keyframes_dir is not None else frames_dir.parent / f"{video_path.stem}_keyframes"
    shutil.rmtree(frames_dir, ignore_errors=True)
    if keep_frames:
        for camera_name in cameras:
            (frames_dir / camera_name).mkdir(parents=True, exist_ok=True)
    if save_keyframes:
        shutil.rmtree(keyframes_dir, ignore_errors=True)
        for camera_name in cameras:
            (keyframes_dir / camera_name).mkdir(parents=True, exist_ok=True)

    writers = {}
    video_root = video_path.stem
    video_ext = video_path.suffix
    if write_videos:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writers = {
            camera_name: imageio.get_writer(video_path.parent / f"{video_root}_{camera_name}{video_ext}", fps=fps)
            for camera_name in cameras
        }

    try:
        state = init_sequence_state(env, block_sequence, **state_kwargs)
        saved_keyframe_steps: set[int] = set()
        while not state["completed"]:
            action = step_pointing_policy(state)
            state["obs"], _, _, _ = env.step(action)
            should_save_keyframe = (
                save_keyframes
                and state["dwell_counter"] == 1
                and not state["completed"]
                and int(state["sequence_position"]) not in saved_keyframe_steps
            )
            seq_index = int(state["sequence_position"]) if should_save_keyframe else None
            block_id = int(state["block_sequence"][seq_index]) if should_save_keyframe else None
            for camera_name in cameras:
                if not (write_videos or keep_frames or should_save_keyframe):
                    continue
                if camera_name == FREE_CAMERA_NAME:
                    frame = render_tuned_free_camera_frame(env)
                else:
                    obs_key = f"{camera_name}_image"
                    if obs_key not in state["obs"]:
                        raise KeyError(f"Missing camera obs '{obs_key}'. Available keys: {list(state['obs'].keys())}")
                    frame = state["obs"][obs_key]
                if write_videos:
                    writers[camera_name].append_data(frame)
                if keep_frames:
                    imageio.imwrite(frames_dir / camera_name / f"frame_{state['step']:05d}.png", frame)
                if should_save_keyframe:
                    imageio.imwrite(
                        keyframes_dir / camera_name / f"keyframe_step_{seq_index:02d}_block_{block_id}_{camera_name}.png",
                        frame,
                    )
            if should_save_keyframe and seq_index is not None:
                saved_keyframe_steps.add(seq_index)
            state["step"] += 1
    finally:
        for camera_name, writer in writers.items():
            writer.close()
            print(f"Video saved to {video_path.parent / f'{video_root}_{camera_name}{video_ext}'}")

    if keep_frames:
        print(f"Frames kept at {frames_dir}")
    if save_keyframes:
        print(f"Keyframes saved at {keyframes_dir}")
    return state


def render_camera_views(env, camera_names: Sequence[str], output_paths: Sequence[Path]) -> None:
    """Renders one RGB frame per requested camera and saves PNGs."""

    if len(camera_names) != len(output_paths):
        raise ValueError("camera_names and output_paths must have the same length")

    for camera_name, output_path in zip(camera_names, output_paths):
        frame = env.sim.render(camera_name=camera_name, width=512, height=512)[..., ::-1]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, frame)
