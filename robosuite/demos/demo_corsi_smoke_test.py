"""Robosuite smoke test for a generated Corsi pointing sequence.

Examples:
    mjpython robosuite/demos/demo_corsi_smoke_test.py --seq-len 4 --render-mode online
    python robosuite/demos/demo_corsi_smoke_test.py --sequence 0,4,8,3 --render-mode offline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.envs import generate_coordinate_trial
from corsi.envs.robosuite_corsi import (
    create_env,
    default_save_root,
    rollout_sequence_offline,
    rollout_sequence_online,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--mode", choices=["forward", "backward"], default="forward")
    parser.add_argument("--sequence", type=str, default="")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--video-name", type=str, default="corsi_smoke_test.mp4")
    return parser.parse_args()


def parse_sequence(text: str):
    if not text:
        return None
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def build_smoke_trial(args):
    explicit_sequence = parse_sequence(args.sequence)
    if explicit_sequence is not None:
        return {
            "trial_id": "manual_sequence",
            "seq_len": len(explicit_sequence),
            "sequence": explicit_sequence,
            "coords": None,
            "delta_coords": None,
            "mode": "manual",
        }

    trial = generate_coordinate_trial(seq_len=args.seq_len, mode=args.mode, trial_id="smoke_test")
    return {
        "trial_id": trial.trial_id,
        "seq_len": trial.seq_len,
        "sequence": trial.sequence,
        "coords": trial.coords,
        "delta_coords": trial.delta_coords,
        "mode": trial.mode,
    }


def main():
    args = parse_args()
    trial = build_smoke_trial(args)
    print(json.dumps(trial, indent=2))

    save_root = default_save_root()
    video_path = save_root / args.video_name
    frames_dir = save_root / video_path.stem

    env = create_env(render_mode=args.render_mode)
    try:
        if args.render_mode == "online":
            rollout_sequence_online(env, trial["sequence"])
        else:
            rollout_sequence_offline(
                env,
                trial["sequence"],
                video_path=video_path,
                frames_dir=frames_dir,
                fps=args.fps,
                keep_frames=args.keep_frames,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
