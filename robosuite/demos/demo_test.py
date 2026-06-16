import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.envs.robosuite_corsi import (
    create_env,
    default_save_root,
    rollout_sequence_offline,
    rollout_sequence_online,
)

RENDER_MODE = "online"  # "online" or "offline"
SAVE_PNG_FRAMES = False
FPS = 20
SEQUENCE_REPEATS = 2
DEMO_SEQUENCE = [0, 1, 2] * SEQUENCE_REPEATS

SAVE_ROOT = default_save_root()
VIDEO_PATH = SAVE_ROOT / "corsi_pointing.mp4"
FRAMES_DIR = SAVE_ROOT / "corsi_pointing_frames"


def main(render_mode="offline"):
    print(f"Running demo sequence: {DEMO_SEQUENCE}")
    env = create_env(render_mode=render_mode)
    try:
        if render_mode == "online":
            rollout_sequence_online(env, DEMO_SEQUENCE)
        else:
            rollout_sequence_offline(
                env,
                DEMO_SEQUENCE,
                video_path=VIDEO_PATH,
                frames_dir=FRAMES_DIR,
                fps=FPS,
                keep_frames=SAVE_PNG_FRAMES,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main(RENDER_MODE)
