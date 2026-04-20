"""Interactive fixed-camera tuning tool for robosuite Corsi scenes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from pynput import keyboard

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.envs.robosuite_corsi import create_env
from robosuite.utils.mjmod import CameraModder
import robosuite.utils.transform_utils as T


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--position-step", type=float, default=0.01)
    parser.add_argument("--angle-step-deg", type=float, default=3.0)
    parser.add_argument("--fovy-step", type=float, default=2.0)
    parser.add_argument(
        "--output-path",
        type=str,
        default="corsi_artifacts/tuned_camera.json",
    )
    return parser.parse_args()


def format_camera_state(modder: CameraModder, camera_name: str) -> dict[str, object]:
    pos = [round(float(value), 6) for value in modder.get_pos(camera_name)]
    quat = [round(float(value), 6) for value in modder.get_quat(camera_name)]
    fovy = round(float(modder.get_fovy(camera_name)), 6)
    return {"camera": camera_name, "pos": pos, "quat_wxyz": quat, "fovy": fovy}


def rotate_camera_quat(quat_wxyz: np.ndarray, axis_xyz: np.ndarray, angle_deg: float) -> np.ndarray:
    delta = T.axisangle2quat(np.asarray(axis_xyz, dtype=np.float32) * np.deg2rad(angle_deg))
    current_xyzw = T.convert_quat(np.asarray(quat_wxyz, dtype=np.float32), to="xyzw")
    updated_xyzw = T.quat_multiply(delta, current_xyzw)
    updated_xyzw = updated_xyzw / np.linalg.norm(updated_xyzw)
    return T.convert_quat(updated_xyzw, to="wxyz")


def print_help():
    print("Camera tuning controls")
    print("  w / s : move +x / -x")
    print("  a / d : move +y / -y")
    print("  r / f : move +z / -z")
    print("  j / l : yaw + / -")
    print("  i / k : pitch + / -")
    print("  u / o : roll + / -")
    print("  - / = : increase / decrease fovy")
    print("  p     : print current camera parameters")
    print("  0     : reset to launch defaults")
    print("  esc   : save and quit")


def main() -> None:
    args = parse_args()
    env = create_env(
        render_mode="online",
        online_render_camera=args.camera,
        control_freq=20,
    )
    env.reset()
    env.render()

    modder = CameraModder(env.sim, camera_names=[args.camera], randomize_position=True, randomize_rotation=True, randomize_fovy=True)
    initial_pos = np.array(modder.get_pos(args.camera), dtype=np.float32)
    initial_quat = np.array(modder.get_quat(args.camera), dtype=np.float32)
    initial_fovy = float(modder.get_fovy(args.camera))

    running = True

    def on_press(key):
        nonlocal running

        try:
            char = key.char.lower() if key.char is not None else None
        except AttributeError:
            char = None

        if key == keyboard.Key.esc:
            running = False
            return False

        pos = np.array(modder.get_pos(args.camera), dtype=np.float32)
        quat = np.array(modder.get_quat(args.camera), dtype=np.float32)
        fovy = float(modder.get_fovy(args.camera))
        changed = False

        if char == "w":
            pos[0] += args.position_step
            changed = True
        elif char == "s":
            pos[0] -= args.position_step
            changed = True
        elif char == "a":
            pos[1] += args.position_step
            changed = True
        elif char == "d":
            pos[1] -= args.position_step
            changed = True
        elif char == "r":
            pos[2] += args.position_step
            changed = True
        elif char == "f":
            pos[2] -= args.position_step
            changed = True
        elif char == "j":
            quat = rotate_camera_quat(quat, np.array([0.0, 0.0, 1.0], dtype=np.float32), args.angle_step_deg)
            changed = True
        elif char == "l":
            quat = rotate_camera_quat(quat, np.array([0.0, 0.0, 1.0], dtype=np.float32), -args.angle_step_deg)
            changed = True
        elif char == "i":
            quat = rotate_camera_quat(quat, np.array([0.0, 1.0, 0.0], dtype=np.float32), args.angle_step_deg)
            changed = True
        elif char == "k":
            quat = rotate_camera_quat(quat, np.array([0.0, 1.0, 0.0], dtype=np.float32), -args.angle_step_deg)
            changed = True
        elif char == "u":
            quat = rotate_camera_quat(quat, np.array([1.0, 0.0, 0.0], dtype=np.float32), args.angle_step_deg)
            changed = True
        elif char == "o":
            quat = rotate_camera_quat(quat, np.array([1.0, 0.0, 0.0], dtype=np.float32), -args.angle_step_deg)
            changed = True
        elif char == "-":
            fovy = min(179.0, fovy + args.fovy_step)
            changed = True
        elif char == "=":
            fovy = max(1.0, fovy - args.fovy_step)
            changed = True
        elif char == "p":
            print(json.dumps(format_camera_state(modder, args.camera), indent=2))
        elif char == "0":
            modder.set_pos(args.camera, initial_pos)
            modder.set_quat(args.camera, initial_quat)
            modder.set_fovy(args.camera, initial_fovy)
            print("Camera reset to defaults")
            print(json.dumps(format_camera_state(modder, args.camera), indent=2))
            return

        if changed:
            modder.set_pos(args.camera, pos)
            modder.set_quat(args.camera, quat)
            modder.set_fovy(args.camera, fovy)
            print(json.dumps(format_camera_state(modder, args.camera), indent=2))

    print_help()
    print("Initial camera parameters")
    print(json.dumps(format_camera_state(modder, args.camera), indent=2))

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while running:
            env.render()
            viewer_handle = getattr(env.viewer, "viewer", None)
            if viewer_handle is not None and hasattr(viewer_handle, "is_running"):
                if not viewer_handle.is_running():
                    running = False
                    break
            time.sleep(0.02)
    finally:
        listener.stop()
        final_state = format_camera_state(modder, args.camera)
        output_path = REPO_ROOT / args.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(final_state, indent=2), encoding="utf-8")
        print("Final camera parameters")
        print(json.dumps(final_state, indent=2))
        print(f"Saved to {output_path}")
        env.close()


if __name__ == "__main__":
    main()
