"""Direct entry script for running the robosuite Corsi pointing demo.

Edit the config block below, then run:

    .venv/bin/mjpython corsi/scripts/run_robosuite_corsi.py
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.envs.robosuite_corsi import (  # noqa: E402
    create_env,
    rollout_sequence_offline,
    rollout_sequence_online,
)


# ---------------------------
# Edit Parameters Here
# ---------------------------
RENDER_MODE = "offline"  # "online" or "offline"
ONLINE_RENDER_CAMERA = "frontview"  # e.g. "frontview"
# OFFLINE_CAMERAS = ["frontview", "birdview"]
OFFLINE_CAMERAS = ["agentview"]  # e.g. ["frontview", "birdview"]

SEQUENCE = [3, 8, 2, 0, 5, 1]
SPEED_GAIN = 0.03
DWELL_STEPS = 24
ARRIVAL_THRESHOLD = 0.01
TARGET_HEIGHT = 0.04
CONTROL_FREQ = 20

FPS = 20
KEEP_FRAMES = False
SAVE_KEYFRAMES = True
VIDEO_NAME = "corsi_manual_run.mp4"


def main() -> None:
    print(
        {
            "render_mode": RENDER_MODE,
            "online_render_camera": ONLINE_RENDER_CAMERA,
            "offline_cameras": OFFLINE_CAMERAS,
            "sequence": SEQUENCE,
            "speed_gain": SPEED_GAIN,
            "dwell_steps": DWELL_STEPS,
            "arrival_threshold": ARRIVAL_THRESHOLD,
            "target_height": TARGET_HEIGHT,
            "control_freq": CONTROL_FREQ,
            "save_keyframes": SAVE_KEYFRAMES,
        }
    )

    env = create_env(
        render_mode=RENDER_MODE,
        online_render_camera=ONLINE_RENDER_CAMERA,
        offline_cameras=OFFLINE_CAMERAS,
        control_freq=CONTROL_FREQ,
    )

    try:
        if RENDER_MODE == "online":
            rollout_sequence_online(
                env,
                SEQUENCE,
                speed_gain=SPEED_GAIN,
                dwell_steps=DWELL_STEPS,
                arrival_threshold=ARRIVAL_THRESHOLD,
                target_height=TARGET_HEIGHT,
            )
            return

        save_root = REPO_ROOT / "robosuite" / "savevideo"
        video_path = save_root / VIDEO_NAME
        frames_dir = save_root / video_path.stem
        keyframes_dir = save_root / f"{video_path.stem}_keyframes"
        rollout_sequence_offline(
            env,
            SEQUENCE,
            video_path=video_path,
            frames_dir=frames_dir,
            fps=FPS,
            keep_frames=KEEP_FRAMES,
            save_keyframes=SAVE_KEYFRAMES,
            keyframes_dir=keyframes_dir,
            offline_cameras=OFFLINE_CAMERAS,
            speed_gain=SPEED_GAIN,
            dwell_steps=DWELL_STEPS,
            arrival_threshold=ARRIVAL_THRESHOLD,
            target_height=TARGET_HEIGHT,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
