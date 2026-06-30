"""Raw multimodal Corsi motion dataset generation."""

from __future__ import annotations

import ast
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from corsi.envs.robosuite_corsi import (
    DEFAULT_GRIPPER_SETTLE_STEPS,
    FREE_CAMERA_NAME,
    collect_motion_state,
    create_env,
    init_sequence_state,
    render_tuned_free_camera_frame,
    step_pointing_policy,
)
from corsi.envs.sequence_generator import standard_corsi_robosuite_board_size
from corsi.experiments.visual_base.scripts.export_robosuite_ee_xy_dataset import (
    collect_block_positions,
    get_site_xyz_world,
    get_table_center_xy_world,
    world_xy_to_corsi_xy,
)
from corsi.experiments.visual_base.scripts.ee_xy_dataset_utils import corsi_lower_left_xy_bounds
from robosuite.utils import transform_utils as T


RAW_SCHEMA_VERSION = "scala_corsi_motion_raw_v1"
REQUIRED_ARRAYS = [
    "rgb",
    "joint",
    "joint_velocity",
    "ee_pose",
    "ee_xy",
    "action",
    "qpos",
    "qvel",
    "timestamp",
    "rank",
    "block_id",
]
SPLIT_NAMES = ("train", "val", "test")


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Loads the small YAML subset used by Corsi configs without requiring PyYAML."""

    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        config = json.loads(text)
    else:
        config: dict[str, Any] = {}
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if line.startswith((" ", "\t")):
                raise ValueError(
                    f"{path}:{line_number}: nested YAML is not supported by this lightweight parser"
                )
            if ":" not in line:
                raise ValueError(f"{path}:{line_number}: expected key: value")
            key, value = line.split(":", 1)
            config[key.strip()] = _parse_scalar(value)
    config["_config_path"] = str(path)
    return normalize_config(config)


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    normalized.setdefault("dataset_name", "corsi_motion_raw_len2_9_n50")
    normalized.setdefault(
        "dataset_root",
        f"corsi_artifacts/motion_base/datasets/{normalized['dataset_name']}",
    )
    normalized.setdefault("layout_id", "fixed_9block_v1")
    normalized.setdefault("camera_name", FREE_CAMERA_NAME)
    normalized.setdefault("image_width", 128)
    normalized.setdefault("image_height", 128)
    normalized.setdefault("length_min", 2)
    normalized.setdefault("length_max", 9)
    normalized.setdefault("num_trials_per_length", 50)
    normalized.setdefault("num_blocks", 9)
    normalized.setdefault("seed", 20260621)
    normalized.setdefault("episode_seed_stride", 1000)
    normalized.setdefault("control_freq", 20)
    normalized.setdefault("speed_gain", 0.12)
    normalized.setdefault("dwell_steps", 4)
    normalized.setdefault("arrival_threshold", 0.01)
    normalized.setdefault("target_height", 0.04)
    normalized.setdefault("gripper_settle_steps", DEFAULT_GRIPPER_SETTLE_STEPS)
    normalized.setdefault("max_control_steps", 2400)
    normalized["dataset_root"] = str(normalized["dataset_root"])
    normalized["manifest_path"] = str(Path(normalized["dataset_root"]) / "manifest.json")
    return normalized


def dataset_root(config: dict[str, Any]) -> Path:
    return Path(str(config["dataset_root"]))


def sequence_lengths(config: dict[str, Any]) -> list[int]:
    return list(range(int(config["length_min"]), int(config["length_max"]) + 1))


def _episode_seed(config: dict[str, Any], *, length: int, index: int) -> int:
    return int(config["seed"]) + int(length) * int(config["episode_seed_stride"]) + int(index)


def _split_length_counts(config: dict[str, Any]) -> dict[str, dict[int, int]] | None:
    raw_counts = config.get("split_length_counts") or config.get("expected_split_length_counts")
    if raw_counts is None:
        return None
    result: dict[str, dict[int, int]] = {}
    for split in SPLIT_NAMES:
        split_counts = raw_counts.get(split, {}) if isinstance(raw_counts, dict) else {}
        result[split] = {int(length): int(count) for length, count in dict(split_counts).items()}
    if not any(result[split] for split in SPLIT_NAMES):
        raise ValueError("split_length_counts must contain at least one split count")
    return result


def _expected_length_counts(config: dict[str, Any]) -> dict[int, int]:
    split_counts = _split_length_counts(config)
    if split_counts is not None:
        totals: dict[int, int] = defaultdict(int)
        for counts in split_counts.values():
            for length, count in counts.items():
                totals[int(length)] += int(count)
        return dict(sorted(totals.items()))
    per_length = int(config["num_trials_per_length"])
    return {int(length): per_length for length in sequence_lengths(config)}


def generate_unique_block_orders(
    *,
    length: int,
    count: int,
    num_blocks: int,
    rng: random.Random,
) -> list[list[int]]:
    if length > num_blocks:
        raise ValueError("Cannot generate no-repeat sequences longer than num_blocks")
    all_count = 1
    for value in range(num_blocks, num_blocks - length, -1):
        all_count *= value
    if count > all_count:
        raise ValueError(f"Requested {count} length-{length} sequences, but only {all_count} exist")

    selected: set[tuple[int, ...]] = set()
    while len(selected) < count:
        selected.add(tuple(rng.sample(range(num_blocks), length)))
    return [list(order) for order in sorted(selected)]


def _all_block_orders(*, length: int, num_blocks: int) -> list[tuple[int, ...]]:
    if length > num_blocks:
        raise ValueError("Cannot generate no-repeat sequences longer than num_blocks")
    return list(permutations(range(int(num_blocks)), int(length)))


def _sample_orders(
    *,
    candidates: Sequence[tuple[int, ...]],
    count: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    if int(count) > len(candidates):
        raise ValueError(f"requested {count} sequences but only {len(candidates)} candidates are available")
    return sorted(rng.sample(list(candidates), int(count)))


def build_split_episode_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    split_counts = _split_length_counts(config)
    if split_counts is None:
        raise ValueError("split_length_counts are required for split-aware episode specs")
    num_blocks = int(config["num_blocks"])
    base_seed = int(config["seed"])
    episodes: list[dict[str, Any]] = []
    global_index = 0
    lengths = sorted({length for counts in split_counts.values() for length in counts})
    for length in lengths:
        all_orders = _all_block_orders(length=length, num_blocks=num_blocks)
        used: set[tuple[int, ...]] = set()
        for split_index, split in enumerate(SPLIT_NAMES):
            count = int(split_counts.get(split, {}).get(length, 0))
            if count <= 0:
                continue
            remaining = [order for order in all_orders if order not in used]
            pool = remaining if count <= len(remaining) else all_orders
            if split == "train" and count == len(all_orders):
                orders = sorted(all_orders)
            else:
                rng = random.Random(base_seed + length * 100 + split_index)
                orders = _sample_orders(candidates=pool, count=count, rng=rng)
            used.update(orders)
            for index, order in enumerate(orders):
                seq_id = f"{split}_len{length:02d}_trial{index:04d}"
                episodes.append(
                    {
                        "seq_id": seq_id,
                        "split": split,
                        "length": int(length),
                        "block_order": [int(block_id) for block_id in order],
                        "layout_id": str(config["layout_id"]),
                        "seed": _episode_seed(config, length=length, index=global_index),
                    }
                )
                global_index += 1
    return episodes


def build_episode_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    if _split_length_counts(config) is not None:
        return build_split_episode_specs(config)
    episodes: list[dict[str, Any]] = []
    per_length = int(config["num_trials_per_length"])
    num_blocks = int(config["num_blocks"])
    base_seed = int(config["seed"])
    for length in sequence_lengths(config):
        rng = random.Random(base_seed + int(length))
        orders = generate_unique_block_orders(
            length=length,
            count=per_length,
            num_blocks=num_blocks,
            rng=rng,
        )
        for index, block_order in enumerate(orders):
            seq_id = f"len{length:02d}_trial{index:03d}"
            episodes.append(
                {
                    "seq_id": seq_id,
                    "length": int(length),
                    "block_order": [int(block_id) for block_id in block_order],
                    "layout_id": str(config["layout_id"]),
                    "seed": _episode_seed(config, length=length, index=index),
                }
            )
    return episodes


def counts_per_length(episodes: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(item["length"]) for item in episodes)
    return {str(length): int(counts.get(length, 0)) for length in sorted(counts)}


def counts_per_split_length(episodes: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[int]] = {split: Counter() for split in SPLIT_NAMES}
    for item in episodes:
        split = str(item.get("split", ""))
        if split in counts:
            counts[split][int(item["length"])] += 1
    return {
        split: {str(length): int(count) for length, count in sorted(counter.items())}
        for split, counter in counts.items()
    }


def _render_frame(env, obs: dict[str, Any], *, camera_name: str, width: int, height: int) -> np.ndarray:
    if camera_name == FREE_CAMERA_NAME:
        return render_tuned_free_camera_frame(env, width=width, height=height)
    obs_key = f"{camera_name}_image"
    if obs_key not in obs:
        raise KeyError(f"Missing camera obs {obs_key!r}. Available keys: {list(obs)}")
    return np.asarray(obs[obs_key], dtype=np.uint8)


def create_raw_env(config: dict[str, Any], *, seed: int | None = None):
    camera_name = str(config["camera_name"])
    env_camera_names = ["agentview"] if camera_name == FREE_CAMERA_NAME else [camera_name]
    env = create_env(
        render_mode="offline",
        offline_cameras=env_camera_names,
        control_freq=int(config["control_freq"]),
        seed=seed,
        use_camera_obs=False,
    )
    # The raw dataset uses a fixed layout; keeping the MuJoCo model loaded makes
    # per-episode deterministic sim resets practical for the full 400 episodes.
    env.hard_reset = False
    return env


def reset_env_rng(env, seed: int) -> None:
    env.seed = int(seed)
    env.rng = np.random.default_rng(int(seed))


def _ee_pose(env, site_name: str) -> np.ndarray:
    site_id = env.sim.model.site_name2id(site_name)
    pos = np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float32)
    rot = np.asarray(env.sim.data.site_xmat[site_id], dtype=np.float32).reshape((3, 3))
    quat_xyzw = np.asarray(T.mat2quat(rot), dtype=np.float32)
    return np.concatenate([pos, quat_xyzw], axis=0).astype(np.float32)


def _record_frame(
    env,
    state: dict[str, Any],
    *,
    obs: dict[str, Any],
    action: np.ndarray,
    rank: int,
    block_id: int,
    camera_name: str,
    image_width: int,
    image_height: int,
    table_center_xy_world: Sequence[float],
) -> dict[str, np.ndarray | float | int]:
    motion_state = collect_motion_state(env, state, action)
    ee_xyz_world = get_site_xyz_world(env, str(state["target_site_name"]))
    ee_xy_table = world_xy_to_corsi_xy(ee_xyz_world[:2], table_center_xy_world)
    return {
        "rgb": _render_frame(env, obs, camera_name=camera_name, width=image_width, height=image_height),
        "joint": np.asarray(motion_state.get("arm_joint_qpos", []), dtype=np.float32),
        "joint_velocity": np.asarray(motion_state.get("arm_joint_qvel", []), dtype=np.float32),
        "ee_pose": _ee_pose(env, str(state["target_site_name"])),
        "ee_xy": np.asarray(ee_xy_table, dtype=np.float32),
        "action": np.asarray(motion_state.get("full_action", []), dtype=np.float32),
        "qpos": np.asarray(env.sim.data.qpos, dtype=np.float32).copy(),
        "qvel": np.asarray(env.sim.data.qvel, dtype=np.float32).copy(),
        "timestamp": float(env.sim.data.time),
        "rank": int(rank),
        "block_id": int(block_id),
    }


def _stack_rows(rows: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Cannot save an episode with no rows")
    arrays: dict[str, np.ndarray] = {}
    for key in REQUIRED_ARRAYS:
        values = [row[key] for row in rows]
        if key in {"timestamp"}:
            arrays[key] = np.asarray(values, dtype=np.float64)
        elif key in {"rank", "block_id"}:
            arrays[key] = np.asarray(values, dtype=np.int64)
        else:
            arrays[key] = np.stack(values, axis=0)
    return arrays


def _empty_segments(block_order: Sequence[int]) -> list[dict[str, int | None]]:
    return [
        {
            "segment_id": int(rank),
            "rank": int(rank),
            "block_id": int(block_id),
            "start_frame": None,
            "end_frame": None,
        }
        for rank, block_id in enumerate(block_order)
    ]


def generate_episode(
    config: dict[str, Any],
    episode: dict[str, Any],
    *,
    env=None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = dataset_root(config)
    seq_id = str(episode["seq_id"])
    episode_dir = root / "episodes" / seq_id
    arrays_path = episode_dir / "arrays.npz"
    metadata_path = episode_dir / "metadata.json"
    segments_path = episode_dir / "segments.json"
    if arrays_path.exists() and metadata_path.exists() and segments_path.exists() and not overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "split" in episode and "split" not in metadata:
            metadata["split"] = str(episode["split"])
        return {
            **metadata,
            "arrays_path": str(arrays_path),
            "metadata_path": str(metadata_path),
            "segments_path": str(segments_path),
            "status": "existing",
        }

    if overwrite:
        shutil.rmtree(episode_dir, ignore_errors=True)
    episode_dir.mkdir(parents=True, exist_ok=True)

    block_order = [int(block_id) for block_id in episode["block_order"]]
    camera_name = str(config["camera_name"])
    own_env = env is None
    if own_env:
        env = create_raw_env(config, seed=int(episode["seed"]))
    else:
        reset_env_rng(env, int(episode["seed"]))
    rows: list[dict[str, Any]] = []
    segments = _empty_segments(block_order)
    xy_norm_bounds = corsi_lower_left_xy_bounds((255.0, 205.0))
    block_positions: dict[str, Any] = {}
    try:
        table_center_xy_world = get_table_center_xy_world(env)
        state = init_sequence_state(
            env,
            block_order,
            speed_gain=float(config["speed_gain"]),
            dwell_steps=int(config["dwell_steps"]),
            arrival_threshold=float(config["arrival_threshold"]),
            target_height=float(config["target_height"]),
            gripper_settle_steps=int(config["gripper_settle_steps"]),
        )
        block_positions = collect_block_positions(
            env,
            state,
            xy_norm_bounds=xy_norm_bounds,
            table_center_xy_world=table_center_xy_world,
        )
        while not state["completed"]:
            if int(state["step"]) >= int(config["max_control_steps"]):
                raise RuntimeError(
                    f"{seq_id}: exceeded max_control_steps={config['max_control_steps']} "
                    f"at rank={state['sequence_position']}"
                )
            rank = int(state["sequence_position"])
            if rank < 0 or rank >= len(block_order):
                raise RuntimeError(f"{seq_id}: invalid active rank {rank}")
            block_id = int(block_order[rank])
            action = step_pointing_policy(state)
            state["obs"], _, _, _ = env.step(action)
            frame_index = len(rows)
            if segments[rank]["start_frame"] is None:
                segments[rank]["start_frame"] = frame_index
            segments[rank]["end_frame"] = frame_index
            rows.append(
                _record_frame(
                    env,
                    state,
                    obs=state["obs"],
                    action=np.asarray(action, dtype=np.float32),
                    rank=rank,
                    block_id=block_id,
                    camera_name=camera_name,
                    image_width=int(config["image_width"]),
                    image_height=int(config["image_height"]),
                    table_center_xy_world=table_center_xy_world,
                )
            )
            state["step"] += 1
    finally:
        if own_env:
            env.close()

    for segment in segments:
        if segment["start_frame"] is None or segment["end_frame"] is None:
            raise RuntimeError(f"{seq_id}: missing frames for segment {segment['segment_id']}")
        segment["start_frame"] = int(segment["start_frame"])
        segment["end_frame"] = int(segment["end_frame"])

    arrays = _stack_rows(rows)
    np.savez_compressed(arrays_path, **arrays)
    episode_metadata = {
        "schema_version": RAW_SCHEMA_VERSION,
        "seq_id": seq_id,
        **({"split": str(episode["split"])} if "split" in episode else {}),
        "length": int(episode["length"]),
        "block_order": block_order,
        "layout_id": str(episode["layout_id"]),
        "seed": int(episode["seed"]),
        "frame_count": int(arrays["rank"].shape[0]),
        "camera_name": camera_name,
        "image_shape": list(arrays["rgb"].shape[1:]),
        "array_keys": REQUIRED_ARRAYS,
        "block_positions": block_positions,
        "xy_normalization": {
            "frame": "corsi_lower_left",
            "bounds": xy_norm_bounds,
            "table_center_xy_world": [float(v) for v in table_center_xy_world],
            "robosuite_board_size_xy": [float(v) for v in standard_corsi_robosuite_board_size()],
            "range": [-1.0, 1.0],
        },
        "control": {
            "control_freq": int(config["control_freq"]),
            "speed_gain": float(config["speed_gain"]),
            "dwell_steps": int(config["dwell_steps"]),
            "arrival_threshold": float(config["arrival_threshold"]),
            "target_height": float(config["target_height"]),
            "gripper_settle_steps": int(config["gripper_settle_steps"]),
        },
        "segments": segments,
    }
    metadata_path.write_text(json.dumps(episode_metadata, indent=2), encoding="utf-8")
    segments_path.write_text(json.dumps(segments, indent=2), encoding="utf-8")
    return {
        **episode_metadata,
        "arrays_path": str(arrays_path),
        "metadata_path": str(metadata_path),
        "segments_path": str(segments_path),
        "status": "generated",
    }


def write_manifest(
    config: dict[str, Any],
    *,
    episodes: Sequence[dict[str, Any]],
    failed: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    root = dataset_root(config)
    manifest_path = Path(str(config["manifest_path"]))
    samples = []
    for episode in episodes:
        sample = {
            "seq_id": episode["seq_id"],
            "length": int(episode["length"]),
            "block_order": [int(v) for v in episode["block_order"]],
            "layout_id": str(episode["layout_id"]),
            "seed": int(episode["seed"]),
            "frame_count": int(episode["frame_count"]),
            "arrays_path": str(episode["arrays_path"]),
            "metadata_path": str(episode["metadata_path"]),
            "segments_path": str(episode["segments_path"]),
            "status": str(episode.get("status", "generated")),
        }
        if "split" in episode:
            sample["split"] = str(episode["split"])
        samples.append(sample)
    expected_length_counts = _expected_length_counts(config)
    split_length_counts = counts_per_split_length(samples)
    has_splits = any(split_length_counts[split] for split in SPLIT_NAMES)
    manifest = {
        "schema_version": RAW_SCHEMA_VERSION,
        "dataset_name": str(config["dataset_name"]),
        "dataset_root": str(root),
        "plan_path": str(root / "sequence_plan.json"),
        "num_episodes": len(samples),
        "expected_episodes": int(sum(expected_length_counts.values())),
        "length_counts": counts_per_length(samples),
        "camera_name": str(config["camera_name"]),
        "array_keys": REQUIRED_ARRAYS,
        "failed_episodes": list(failed),
        "skipped_episodes": list(failed),
        "samples": samples,
    }
    if has_splits:
        manifest["split_length_counts"] = split_length_counts
        manifest["split"] = {
            split: sorted(str(sample["seq_id"]) for sample in samples if sample.get("split") == split)
            for split in SPLIT_NAMES
        }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_sequence_plan(config: dict[str, Any], episodes: Sequence[dict[str, Any]]) -> Path:
    root = dataset_root(config)
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": RAW_SCHEMA_VERSION,
        "dataset_name": str(config["dataset_name"]),
        "layout_id": str(config["layout_id"]),
        "num_blocks": int(config["num_blocks"]),
        "length_counts": counts_per_length(episodes),
        "split_length_counts": counts_per_split_length(episodes),
        "total_episodes": int(len(episodes)),
        "seed": int(config["seed"]),
        "episodes": [
            {
                key: episode[key]
                for key in ("seq_id", "split", "length", "block_order", "layout_id", "seed")
                if key in episode
            }
            for episode in episodes
        ],
    }
    plan_path = root / "sequence_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan_path


def generate_raw_dataset(
    config: dict[str, Any],
    *,
    max_episodes: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    episodes_to_generate = build_episode_specs(config)
    write_sequence_plan(config, episodes_to_generate)
    if max_episodes is not None:
        episodes_to_generate = episodes_to_generate[: int(max_episodes)]
    generated: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    env = create_raw_env(config, seed=int(config["seed"]))
    try:
        for index, episode in enumerate(episodes_to_generate):
            try:
                generated.append(generate_episode(config, episode, env=env, overwrite=overwrite))
            except Exception as exc:  # pragma: no cover - exercised by CLI behavior.
                failed.append(
                    {
                        "seq_id": str(episode.get("seq_id", index)),
                        "length": int(episode.get("length", -1)),
                        "block_order": list(episode.get("block_order", [])),
                        "error": str(exc),
                    }
                )
            if (index + 1) % 10 == 0 or index + 1 == len(episodes_to_generate):
                print(
                    f"[raw] processed {index + 1}/{len(episodes_to_generate)} episodes "
                    f"(ok={len(generated)}, failed={len(failed)})",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        env.close()
    manifest = write_manifest(config, episodes=generated, failed=failed)
    return {
        "dataset_root": str(dataset_root(config)),
        "manifest_path": str(config["manifest_path"]),
        "episode_count": len(generated),
        "failed_count": len(failed),
        "failed_episodes": failed,
        "counts_per_length": manifest["length_counts"],
    }


def raw_main(argv: Iterable[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate SCALA Corsi raw motion episodes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = generate_raw_dataset(
        load_config(args.config),
        max_episodes=args.max_episodes,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failed_count"] == 0 else 1
