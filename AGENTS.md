# AGENTS.md

Handoff for the Corsi / robosuite work in this repository.

## Operating Rules

- Use the conda environment `robosuite`.
- Preserve existing `freecam_index`, heatmap, and `freecam_ee_xy` workflows.
- Keep changes incremental and separate raw data generation from later training / dataset-loader work.
- Do not blindly overwrite or remove `AUTHORS`, `requirements.txt`, `.vscode/settings.json`, or local `corsi_artifacts/`.
- Treat `corsi_artifacts/` as local generated data; it is ignored by Git.

## Current Branch

- Branch: `codex/corsi-motion`
- Upstream: `origin/codex/corsi-motion`
- Latest pushed commit: `5c545043 Add SCALA Corsi raw motion generator`
- Relevant prior commits:
  - `97aae878 Integrate Corsi visual baseline`
  - `83eb5882 Add freecam motion baseline smoke`

## Raw Motion Dataset Handoff

The first SCALA Corsi Motion Raw Dataset has been generated locally.

- Dataset root: `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50`
- Manifest: `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50/manifest.json`
- Size: about `1.1G`
- Episodes: `400`
- Lengths: `2` through `9`
- Count per length: `50`
- Sequence rule: ordered `block_order`, no repeated block within one episode.
- Failed/skipped episodes at validation time: none.

Each episode directory contains:

- `arrays.npz`: synchronized per-frame arrays: `rgb`, `joint`, `joint_velocity`, `ee_pose`, `ee_xy`, `action`, `qpos`, `qvel`, `timestamp`, `rank`, `block_id`
- `metadata.json`: episode metadata, block positions, normalization, control settings, and segment copy
- `segments.json`: one segment per visited block with `segment_id`, `rank`, `block_id`, `start_frame`, `end_frame`

Committed raw generation code:

- `configs/corsi_motion_raw_len2_9_n50.yaml`
- `corsi/data/motion_raw.py`
- `corsi/data/generate_raw.py`
- `corsi/envs/robosuite_corsi.py` seed / `use_camera_obs` support
- `collect_block_positions` helper in `corsi/experiments/visual_base/scripts/export_robosuite_ee_xy_dataset.py`

Generate / resume command:

```bash
conda run -n robosuite python -m corsi.data.generate_raw \
  --config configs/corsi_motion_raw_len2_9_n50.yaml
```

The raw-only commit intentionally does not keep separate `generate_plan`, `validate_raw`, or raw test files. They were useful during generation, but were removed to keep the submitted code focused on generation.

## Dense Freecam Motion Support

There is still uncommitted dense `freecam_motion_v1` support in the worktree. This is separate from the raw `.npz` dataset.

Purpose:

- Extend the older manifest-based `freecam_ee_xy` exporter with a `dense_motion` payload.
- Save synchronized fields such as `image_t`, `qpos_t`, `qvel_t`, `arm_action_t`, `ee_pos_t`, `ee_xy_t`, `ee_xy_norm_t`, `block_sequence`, `block_positions`, `tap_timestamps`, and `segment_ids`.
- Let `FreecamMotionDataset` and `collate_motion_batch` read/pad this dense manifest data.

Current uncommitted files for this line:

- `corsi/data/collate_motion.py`
- `corsi/data/freecam_motion_dataset.py`
- `corsi/experiments/visual_base/scripts/export_robosuite_ee_xy_dataset.py`
- `tests/test_corsi_motion.py`
- `corsi/experiments/motion_base/`
- `corsi/experiments/README.md`

Recommendation:

- Keep this work only if the old manifest-based `freecam_motion_v1` path is still needed.
- If kept, commit it separately from raw generation.
- Segment-uniform experiment code was deliberately deleted and should not be reintroduced unless explicitly requested.

## Motion Base Scratch Files

Current untracked `motion_base` files are only for dense freecam smoke work:

- `corsi/experiments/motion_base/README.md`
- `corsi/experiments/motion_base/__init__.py`
- `corsi/experiments/motion_base/configs/freecam_motion_v1_len2_smoke.yaml`

Keep them only together with dense freecam motion support. Delete them if that support is dropped.

## Validation Commands

Small checks used after cleanup:

```bash
conda run -n robosuite python -m pytest tests/test_corsi_motion.py -q
conda run -n robosuite python -m corsi.data.generate_raw --help
```

Broader Corsi smoke check:

```bash
conda run -n robosuite python -m pytest \
  tests/test_corsi_heatmaps.py \
  tests/test_corsi_attention.py \
  tests/test_corsi_motion.py
```

