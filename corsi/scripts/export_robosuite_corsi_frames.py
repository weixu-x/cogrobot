"""Exports fixed-layout Corsi camera frames directly from robosuite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.envs.robosuite_corsi import FREE_CAMERA_NAME, create_env, render_tuned_free_camera_frame, rollout_sequence_offline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", type=str, default="frontview,birdview")
    parser.add_argument("--sequence", type=str, default="0,4,8,3")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--speed-gain", type=float, default=0.03)
    parser.add_argument("--dwell-steps", type=int, default=24)
    parser.add_argument("--arrival-threshold", type=float, default=0.01)
    parser.add_argument("--target-height", type=float, default=0.04)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="corsi_artifacts/visual_base/previews/robosuite_camera_preview",
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [token.strip() for token in text.split(",") if token.strip()]


def parse_sequence(text: str) -> list[int]:
    return [int(token) for token in parse_csv_list(text)]


def save_reset_frames(obs: dict, camera_names: list[str], output_dir: Path) -> dict[str, str]:
    raise NotImplementedError("Use save_reset_frames_with_env so free camera rendering is supported")


def save_reset_frames_with_env(env, obs: dict, camera_names: list[str], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for camera_name in camera_names:
        output_path = output_dir / f"{camera_name}.png"
        if camera_name == FREE_CAMERA_NAME:
            frame = render_tuned_free_camera_frame(env)
        else:
            obs_key = f"{camera_name}_image"
            if obs_key not in obs:
                raise KeyError(f"Missing camera obs '{obs_key}'. Available keys: {sorted(obs.keys())}")
            frame = obs[obs_key]
        imageio.imwrite(output_path, frame)
        paths[camera_name] = str(output_path)
    return paths


def main() -> None:
    args = parse_args()
    camera_names = parse_csv_list(args.cameras)
    env_camera_names = [camera_name for camera_name in camera_names if camera_name != FREE_CAMERA_NAME]
    if not env_camera_names:
        env_camera_names = ["agentview"]
    sequence = parse_sequence(args.sequence)
    output_dir = Path(args.output_dir)
    reset_dir = output_dir / "reset"
    rollout_frames_dir = output_dir / "rollout_frames"
    keyframes_dir = output_dir / "rollout_keyframes"
    video_path = output_dir / "rollout.mp4"

    env = create_env(
        render_mode="offline",
        offline_cameras=env_camera_names,
        control_freq=args.control_freq,
    )
    try:
        obs = env.reset()
        reset_paths = save_reset_frames_with_env(env, obs, camera_names, reset_dir)

        rollout_sequence_offline(
            env,
            sequence,
            video_path=video_path,
            frames_dir=rollout_frames_dir,
            fps=args.fps,
            keep_frames=args.keep_frames,
            save_keyframes=True,
            keyframes_dir=keyframes_dir,
            offline_cameras=camera_names,
            speed_gain=args.speed_gain,
            dwell_steps=args.dwell_steps,
            arrival_threshold=args.arrival_threshold,
            target_height=args.target_height,
        )
    finally:
        env.close()

    if not args.save_video:
        if video_path.exists():
            video_path.unlink()
        for camera_name in camera_names:
            camera_video = video_path.parent / f"{video_path.stem}_{camera_name}{video_path.suffix}"
            if camera_video.exists():
                camera_video.unlink()

    manifest = {
        "camera_names": camera_names,
        "sequence": sequence,
        "reset_paths": reset_paths,
        "keyframes_dir": str(keyframes_dir),
        "keep_frames": bool(args.keep_frames),
        "rollout_frames_dir": str(rollout_frames_dir),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
