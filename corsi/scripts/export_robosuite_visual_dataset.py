"""Exports a small robosuite visual Corsi dataset with per-trial manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.envs.robosuite_corsi import FREE_CAMERA_NAME, create_env, rollout_sequence_offline
from corsi.envs.sequence_generator import generate_trial_collection
from corsi.scripts.export_robosuite_corsi_frames import parse_csv_list, save_reset_frames_with_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="corsi_artifacts/visual_base/datasets/robosuite_visual_dataset",
    )
    parser.add_argument("--dataset-name", type=str, default="")
    parser.add_argument("--split-name", type=str, default="")
    parser.add_argument("--cameras", type=str, default=FREE_CAMERA_NAME)
    parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument("--seq-len-range", type=str, default="2,4")
    parser.add_argument("--sequences-json", type=str, default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--speed-gain", type=float, default=0.03)
    parser.add_argument("--dwell-steps", type=int, default=24)
    parser.add_argument("--arrival-threshold", type=float, default=0.01)
    parser.add_argument("--target-height", type=float, default=0.04)
    parser.add_argument("--keep-rollout-frames", action="store_true")
    parser.add_argument("--keep-rollout-videos", action="store_true")
    return parser.parse_args()


def parse_seq_len_range(text: str) -> tuple[int, int]:
    values = [int(token) for token in parse_csv_list(text)]
    if len(values) != 2:
        raise ValueError("--seq-len-range must contain exactly two integers, e.g. 2,4")
    return values[0], values[1]


def load_sequences(sequences_json: str, *, num_trials: int, seq_len_range: tuple[int, int], seed: int) -> list[list[int]]:
    if sequences_json:
        payload = json.loads(Path(sequences_json).read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("sequences_json must contain a non-empty list of sequences")
        return [[int(block_id) for block_id in sequence] for sequence in payload]

    trials = generate_trial_collection(
        num_trials=num_trials,
        seq_len_range=seq_len_range,
        seed=seed,
    )
    return [list(trial.sequence) for trial in trials]


def build_keyframe_paths(sample_dir: Path, camera_names: list[str], sequence: list[int]) -> dict[str, list[str]]:
    keyframe_paths: dict[str, list[str]] = {}
    for camera_name in camera_names:
        keyframe_paths[camera_name] = [
            str(
                sample_dir
                / "rollout_keyframes"
                / camera_name
                / f"keyframe_step_{step_index:02d}_block_{block_id}_{camera_name}.png"
            )
            for step_index, block_id in enumerate(sequence)
        ]
    return keyframe_paths


def main() -> None:
    args = parse_args()
    camera_names = parse_csv_list(args.cameras)
    env_camera_names = [camera_name for camera_name in camera_names if camera_name != FREE_CAMERA_NAME]
    if not env_camera_names:
        env_camera_names = ["agentview"]

    output_dir = Path(args.output_dir)
    dataset_name = args.dataset_name or output_dir.name
    split_name = args.split_name or output_dir.name
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    seq_len_range = parse_seq_len_range(args.seq_len_range)

    sequences = load_sequences(
        args.sequences_json,
        num_trials=args.num_trials,
        seq_len_range=seq_len_range,
        seed=args.seed,
    )

    env = create_env(
        render_mode="offline",
        offline_cameras=env_camera_names,
        control_freq=args.control_freq,
    )

    dataset_samples: list[dict[str, object]] = []
    try:
        for sample_index, sequence in enumerate(sequences):
            trial_id = f"trial_{sample_index:06d}"
            sample_dir = samples_dir / trial_id
            reset_dir = sample_dir / "reset"
            rollout_frames_dir = sample_dir / "rollout_frames"
            keyframes_dir = sample_dir / "rollout_keyframes"
            manifest_path = sample_dir / "manifest.json"

            obs = env.reset()
            reset_paths = save_reset_frames_with_env(env, obs, camera_names, reset_dir)
            rollout_sequence_offline(
                env,
                sequence,
                video_path=sample_dir / "rollout.mp4",
                frames_dir=rollout_frames_dir,
                fps=args.fps,
                keep_frames=args.keep_rollout_frames,
                save_keyframes=True,
                keyframes_dir=keyframes_dir,
                offline_cameras=camera_names,
                speed_gain=args.speed_gain,
                dwell_steps=args.dwell_steps,
                arrival_threshold=args.arrival_threshold,
                target_height=args.target_height,
            )

            if not args.keep_rollout_videos:
                for camera_name in camera_names:
                    camera_video_path = sample_dir / f"rollout_{camera_name}.mp4"
                    if camera_video_path.exists():
                        camera_video_path.unlink()

            sample_manifest = {
                "dataset_name": dataset_name,
                "split_name": split_name,
                "trial_id": trial_id,
                "camera_names": camera_names,
                "sequence": sequence,
                "length": len(sequence),
                "reset_paths": reset_paths,
                "keyframe_paths": build_keyframe_paths(sample_dir, camera_names, sequence),
                "sample_structure": [
                    "reset",
                    "rollout_keyframes",
                    "manifest.json",
                ],
                "metadata": {
                    "fps": args.fps,
                    "control_freq": args.control_freq,
                    "speed_gain": args.speed_gain,
                    "dwell_steps": args.dwell_steps,
                    "arrival_threshold": args.arrival_threshold,
                    "target_height": args.target_height,
                    "seed": args.seed,
                },
            }
            manifest_path.write_text(json.dumps(sample_manifest, indent=2), encoding="utf-8")

            dataset_samples.append(
                {
                    **sample_manifest,
                    "sample_dir": str(sample_dir),
                    "manifest_path": str(manifest_path),
                }
            )
    finally:
        env.close()

    dataset_manifest = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "camera_names": camera_names,
        "num_samples": len(dataset_samples),
        "samples_dir": str(samples_dir),
        "sequence_length_range": {
            "min": seq_len_range[0],
            "max": seq_len_range[1],
        },
        "sample_structure": [
            "reset",
            "rollout_keyframes",
            "manifest.json",
        ],
        "export_params": {
            "control_freq": args.control_freq,
            "fps": args.fps,
            "speed_gain": args.speed_gain,
            "dwell_steps": args.dwell_steps,
            "arrival_threshold": args.arrival_threshold,
            "target_height": args.target_height,
            "keep_rollout_frames": bool(args.keep_rollout_frames),
            "keep_rollout_videos": bool(args.keep_rollout_videos),
            "seed": args.seed,
        },
        "samples": dataset_samples,
    }
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")
    print(json.dumps(dataset_manifest, indent=2))
    print(f"Dataset manifest saved to {dataset_manifest_path}")


if __name__ == "__main__":
    main()
