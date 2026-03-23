import os
import shutil
from pathlib import Path

import imageio
import numpy as np

from robosuite.robots import register_robot_class
from robosuite.models.robots import Panda
from robosuite.models.grippers import InspireRightHand, register_gripper
import robosuite as suite
import robosuite.macros as macros
from robosuite.controllers import load_composite_controller_config
import robosuite.environments.manipulation.lift_with_corsi_arena
import robosuite.environments.manipulation.lift_corsi_min

RENDER_MODE = "online"  # "online" or "offline"
ONLINE_RENDER_CAMERA = "frontview"
OFFLINE_CAMERAS = ["frontview", "birdview"]
SAVE_PNG_FRAMES = False

@register_gripper
class InspireRightHandWithInit(InspireRightHand):
    # [thumb_rot, index, middle, ring, thumb_bend, pinky]
    CTRL_MAX = np.array([
        1.87, 1.62,  # pinky distal, pinky proximal
        1.75, 1.62,  # ring distal, ring proximal
        1.82, 1.62,  # middle distal, middle proximal
        1.82, 1.62,  # index distal, index proximal 
        0.77, 0.68, 0.68,  # thumb distal, middle, proximal_2
        1.30,         # thumb proximal_1 (rot)
    ], dtype=np.float32)

    INDEX_OPEN_GAIN = 1.5   # >1 会更“直”      
    INDEX_OPEN_BIAS = -0.5 # 负数会更“直”（更往0方向推）
    INDICES = np.array([5, 5,  3, 3,  2, 2,  1, 1,  4, 4, 4,  0], dtype=np.int32)

    # def format_action(self, action):
    #     action = np.asarray(action, dtype=np.float32)
    #     assert action.shape == (6,)

    #     # 让你的 action 语义固定为 0~1：0=伸直/张开，1=弯曲/闭合
    #     action = np.clip(action, 0.0, 1.0)

    #     u = action[self.INDICES]     # 12维语义
    #     ctrl = u * self.CTRL_MAX     # 映射到 xml 的 ctrlrange
    #     return ctrl
    def format_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (6,)
        action = np.clip(action, 0.0, 1.0)

        u = action[self.INDICES].copy()   # 12维语义 (0=伸直/张开, 1=弯曲/闭合)

        # 这两个位置对应 index distal / index proximal（看你 CTRL_MAX 注释顺序）
        # 你的 CTRL_MAX 顺序是：pinky(0,1) ring(2,3) middle(4,5) index(6,7) thumb(8,9,10) thumb_rot(11)
        idx_distal = 6
        idx_prox   = 7

        # 让 index 在“张开方向”更用力：u 趋近于 0
        u[idx_distal] = np.clip(u[idx_distal] * self.INDEX_OPEN_GAIN + self.INDEX_OPEN_BIAS, 0.0, 1.0)
        u[idx_prox]   = np.clip(u[idx_prox]   * self.INDEX_OPEN_GAIN + self.INDEX_OPEN_BIAS, 0.0, 1.0)

        # Map [0, 1] -> [-1, 1] for SimpleGripController scaling
        return u * 2.0 - 1.0


    @property
    def init_qpos(self):
        # 建议 init 就是“完全闭合/握拳”= 1.0（如果你想初始就握拳）
        base = np.array([0.1, 0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # thumb_rot稍微转一点
        return (base[self.INDICES] * self.CTRL_MAX)


# @register_robot_class("FixedBaseRobot")
class PandaDexRH(Panda):
    @property
    def default_gripper(self):
        return {"right": "InspireRightHandWithInit"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [-0.5, 0.5, 0.5, -0.5]}
    
macros.IMAGE_CONVENTION = "opencv" 

REPO_ROOT = Path(__file__).resolve().parents[2]
SAVE_ROOT = Path(os.environ.get("ROBOSUITE_SAVE_DIR", REPO_ROOT / "robosuite" / "savevideo"))
VIDEO_PATH = SAVE_ROOT / "corsi_pointing.mp4"
FRAMES_DIR = SAVE_ROOT / "corsi_pointing_frames"
FPS = 20
MAX_ROUNDS = 2

# Gripper control follows InspireRightHand / inspire_right_hand.xml:
# action order: [thumb_rot, index, middle, ring, thumb_bend, pinky]
# 0.0 = straight / open, 1.0 = fully bent / closed (scaled to each joint's ctrlrange)
GRIPPER_ACTION = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
GRIPPER_ACTION = np.clip(GRIPPER_ACTION, 0.0, 1.0)


def create_env(render_mode="offline"):
    online = render_mode == "online"
    camera_names = ONLINE_RENDER_CAMERA if online else OFFLINE_CAMERAS
    camera_heights = 512 if online else [512] * len(OFFLINE_CAMERAS)
    camera_widths = 512 if online else [512] * len(OFFLINE_CAMERAS)
    return suite.make(
        env_name="CorsiSceneDemo",
        robots="PandaDexRH",
        controller_configs=load_composite_controller_config(controller="BASIC"),
        has_renderer=online,
        has_offscreen_renderer=not online,
        render_camera=ONLINE_RENDER_CAMERA,
        use_camera_obs=not online,
        camera_names=camera_names,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        control_freq=20,
        ignore_done=True,
    )


def init_episode_state(env):
    obs = env.reset()
    robot = env.robots[0]
    arm = robot.arms[0]
    gripper_name = robot.get_gripper_name(arm)
    arm_dim = robot.part_controllers[arm].control_dim

    if hasattr(env.model, "mujoco_arena") and hasattr(env.model.mujoco_arena, "block_names"):
        block_names = list(env.model.mujoco_arena.block_names)
    else:
        block_names = [name for name in env.sim.model.body_names if name.startswith("corsi_block_")]

    return {
        "obs": obs,
        "robot": robot,
        "arm": arm,
        "gripper_name": gripper_name,
        "arm_dim": arm_dim,
        "target_blocks": block_names[:3],
        "target_idx": 0,
        "rounds_done": 0,
        "step": 0,
    }


def step_policy(state):
    robot = state["robot"]
    arm = state["arm"]
    arm_dim = state["arm_dim"]
    target_blocks = state["target_blocks"]

    arm_action = np.zeros(arm_dim, dtype=np.float32)
    if arm_dim >= 3 and target_blocks:
        target_name = target_blocks[state["target_idx"]]
        target_pose_in_base = robot.pose_in_base_from_name(target_name)
        target_pos = target_pose_in_base[:3, 3] + np.array([0.0, 0.0, 0.2], dtype=np.float32)
        current_pos = robot._hand_pos[arm]

        delta = target_pos - current_pos
        out_max = np.array(robot.part_controllers[arm].output_max[:3], dtype=np.float32)
        scaled = np.divide(delta, out_max, out=np.zeros_like(delta), where=out_max != 0)
        speed_gain = 0.1
        arm_action[:3] = np.clip(scaled * speed_gain, -1.0, 1.0)

        if np.linalg.norm(target_pos - current_pos) < 0.015:
            state["target_idx"] = (state["target_idx"] + 1) % len(target_blocks)
            if state["target_idx"] == 0:
                state["rounds_done"] += 1

    return robot.create_action_vector({arm: arm_action, state["gripper_name"]: GRIPPER_ACTION})


def run_online_render(env):
    state = init_episode_state(env)
    while state["rounds_done"] < MAX_ROUNDS:
        action = step_policy(state)
        state["obs"], _, _, _ = env.step(action)
        env.render()
        state["step"] += 1


def build_video_from_frames(frames_dir, video_path, fps=20):
    frames_dir = Path(frames_dir)
    video_path = Path(video_path)
    frame_files = sorted(path for path in frames_dir.iterdir() if path.suffix == ".png")
    if not frame_files:
        raise RuntimeError(f"No frames found under {frames_dir}")

    video_path.parent.mkdir(parents=True, exist_ok=True)
    valid_count = 0
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame_path in frame_files:
            if frame_path.stat().st_size == 0:
                print(f"Skip empty frame: {frame_path}")
                continue
            try:
                frame = imageio.imread(frame_path)
                writer.append_data(frame)
                valid_count += 1
            except OSError as exc:
                print(f"Skip corrupted frame: {frame_path} ({exc})")
                continue

    if valid_count == 0:
        raise RuntimeError(f"All frames are invalid under {frames_dir}")


def run_offline_render(env, video_path, frames_dir, fps=20, keep_frames=False):
    frames_dir = Path(frames_dir)
    video_path = Path(video_path)
    shutil.rmtree(frames_dir, ignore_errors=True)
    if keep_frames:
        for cam_name in OFFLINE_CAMERAS:
            (frames_dir / cam_name).mkdir(parents=True, exist_ok=True)

    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_root = video_path.stem
    video_ext = video_path.suffix
    writers = {
        cam_name: imageio.get_writer(video_path.parent / f"{video_root}_{cam_name}{video_ext}", fps=fps)
        for cam_name in OFFLINE_CAMERAS
    }
    try:
        state = init_episode_state(env)
        while state["rounds_done"] < MAX_ROUNDS:
            action = step_policy(state)
            state["obs"], _, _, _ = env.step(action)
            for cam_name in OFFLINE_CAMERAS:
                obs_key = f"{cam_name}_image"
                if obs_key not in state["obs"]:
                    raise KeyError(f"Missing camera obs '{obs_key}'. Available keys: {list(state['obs'].keys())}")
                frame = state["obs"][obs_key]
                writers[cam_name].append_data(frame)
                if keep_frames:
                    imageio.imwrite(
                        frames_dir / cam_name / f"frame_{state['step']:05d}.png",
                        frame,
                    )
            state["step"] += 1
    finally:
        for cam_name, writer in writers.items():
            writer.close()
            print(f"Video saved to {video_path.parent / f'{video_root}_{cam_name}{video_ext}'}")

    if keep_frames:
        print(f"Frames kept at {frames_dir}")


def main(render_mode="offline"):
    env = create_env(render_mode=render_mode)
    try:
        if render_mode == "online":
            run_online_render(env)
        else:
            run_offline_render(
                env,
                VIDEO_PATH,
                FRAMES_DIR,
                fps=FPS,
                keep_frames=SAVE_PNG_FRAMES,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main(RENDER_MODE)
