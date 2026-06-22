# SCALA Corsi Motion Baseline Audit

Date: 2026-06-22

This is the Phase 0 audit for the SCALA Corsi visual-proprioceptive motion
baseline. The original 6-joint target failed the hard data gate because the raw
dataset stores seven Panda arm joints. On 2026-06-22 the user revised the
baseline scope to 7-joint Panda arm prediction:

`q_t = [robot0_joint1, ..., robot0_joint7]`

Under that revised scope, the raw data gate passes.

## Hard Gate Result

Status: **PASSED for revised 7-joint baseline**

The raw dataset provides synchronized images, segments, timestamps, and seven
Panda arm joint arrays. The seven-joint name/order is recoverable from the
current project environment and matches the stored `joint` array width.

Observed facts:

- Raw episode arrays contain `joint` and `joint_velocity` with shape `[T, 7]`.
- The robosuite environment reports seven arm joints in order:
  `robot0_joint1`, `robot0_joint2`, `robot0_joint3`, `robot0_joint4`,
  `robot0_joint5`, `robot0_joint6`, `robot0_joint7`.
- Raw metadata contains no embedded `joint_names`, `joint_order`,
  `joint_indices`, or equivalent field, so the canonical derived dataset must
  persist the seven-joint order explicitly.
- `action` has shape `[T, 12]`, and the controller `arm_action` available from
  `collect_motion_state` has six control dimensions, but those are controller
  action dimensions, not six named joint coordinates.

The revised target requires:

- input `q_t` and target `q_{t+1}` to be a 7-joint Panda arm vector;
- exactly seven joint names and their order to be explicitly persisted in the
  derived canonical dataset/config;
- no implicit column dropping.

The current raw dataset can support this revised target.

## 1. Raw Dataset Path And Storage Format

Raw dataset path:

`corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50`

Top-level files:

- `manifest.json`
- `sequence_plan.json`
- `rgb_preview_len02_trial000.png`
- `len04_trial037_rgb_preview.png`

Per-episode files:

- `episodes/<seq_id>/arrays.npz`
- `episodes/<seq_id>/metadata.json`
- `episodes/<seq_id>/segments.json`

The multimodal arrays are stored in compressed NumPy `.npz` files. Metadata,
manifest, sequence plan, and segment tables are JSON.

## 2. Exact Episode And Array Schema

Manifest schema version:

`scala_corsi_motion_raw_v1`

Episode metadata fields observed in
`episodes/len02_trial000/metadata.json`:

- `schema_version`
- `seq_id`
- `length`
- `block_order`
- `layout_id`
- `seed`
- `frame_count`
- `camera_name`
- `image_shape`
- `array_keys`
- `block_positions`
- `xy_normalization`
- `control`
- `segments`

Array keys in each `arrays.npz`:

- `rgb`
- `joint`
- `joint_velocity`
- `ee_pose`
- `ee_xy`
- `action`
- `qpos`
- `qvel`
- `timestamp`
- `rank`
- `block_id`

Observed per-frame shapes:

- `rgb`: `[T, 128, 128, 3]`, `uint8`
- `joint`: `[T, 7]`, `float32`
- `joint_velocity`: `[T, 7]`, `float32`
- `ee_pose`: `[T, 7]`, `float32`
- `ee_xy`: `[T, 2]`, `float32`
- `action`: `[T, 12]`, `float32`
- `qpos`: `[T, 19]`, `float32`
- `qvel`: `[T, 19]`, `float32`
- `timestamp`: `[T]`, `float64`
- `rank`: `[T]`, `int64`
- `block_id`: `[T]`, `int64`

All required arrays share the same timestep dimension within each episode.

## 3. Episode Counts By Length

Manifest counts:

| Length | Episodes |
| --- | ---: |
| 2 | 50 |
| 3 | 50 |
| 4 | 50 |
| 5 | 50 |
| 6 | 50 |
| 7 | 50 |
| 8 | 50 |
| 9 | 50 |

Total episodes: 400.

## 4. Image Shape, Dtype, Channel Order, Timestamps

Image shape and dtype:

- `rgb`: `[T, 128, 128, 3]`
- dtype: `uint8`
- layout: channel-last HWC
- channel order: RGB, as indicated by the `rgb` key and freecam renderer path

Timestamp facts:

- `timestamp`: `[T]`, `float64`
- Control frequency in metadata: 20 Hz
- Observed timestep delta: min `0.04999999999997229`, max
  `0.0500000000000167`, mean `0.05`

Episode frame-count range:

- min: 48
- max: 290
- mean: 162.982

## 5. Raw Joint Shape And Explicit Joint Names

Raw joint arrays:

- `joint`: `[T, 7]`, `float32`
- `joint_velocity`: `[T, 7]`, `float32`

Environment introspection reports:

- robot: `PandaDexRH` / `FixedBaseRobot`
- controller container: `CompositeController`
- env action dimension: 12
- arm joint source:
  `robot.sim.data.qpos[robot._ref_arm_joint_pos_indexes]`
- arm joint qpos indexes: `[0, 1, 2, 3, 4, 5, 6]`
- arm joint qvel indexes: `[0, 1, 2, 3, 4, 5, 6]`
- arm joint names:
  `robot0_joint1`, `robot0_joint2`, `robot0_joint3`, `robot0_joint4`,
  `robot0_joint5`, `robot0_joint6`, `robot0_joint7`

The exact seven-joint order is recoverable from the current environment code
and is the approved prediction target for the revised baseline. Because raw
metadata does not embed the names, the canonical dataset must store this order
explicitly and fingerprint it.

## 6. Segment Schema And Segment-Length Histogram

Segment schema:

- `segment_id`
- `rank`
- `block_id`
- `start_frame`
- `end_frame`

Segment rules validated by the raw validator:

- one segment per block in `block_order`
- `segment_id` increases from 0
- `rank` increases from 0
- `block_id == block_order[rank]`
- frame bounds are valid
- `end_frame >= start_frame`
- per-frame `rank` and `block_id` match the segment span

Observed segment lengths:

- min: 19 frames
- max: 38 frames
- mean: 29.633 frames

Histogram:

| Frames | Count |
| ---: | ---: |
| 19 | 6 |
| 20 | 7 |
| 21 | 59 |
| 22 | 102 |
| 23 | 95 |
| 24 | 162 |
| 25 | 127 |
| 26 | 185 |
| 27 | 41 |
| 28 | 93 |
| 29 | 85 |
| 30 | 200 |
| 31 | 107 |
| 32 | 150 |
| 33 | 220 |
| 34 | 188 |
| 35 | 155 |
| 36 | 68 |
| 37 | 120 |
| 38 | 30 |

## 7. Active Target Block Visual Identifiability

The active target block is not explicitly visually cued in the raw RGB frames.
Preview images show fixed black blocks and the robot end-effector moving toward
or near a block, but no target highlighting, label, flash, or symbolic cue is
present.

If training continues after fixing the joint gate, this task should be reported
as local sensorimotor prediction from current image and proprioception. The
visual frame may reveal current hand/block geometry, but it does not identify
the upcoming target block independently of the already ongoing trajectory.

No synthetic target cue should be added.

## 8. Angle Wrapping Or Discontinuity

A coarse scan of adjacent raw `joint` differences found:

- maximum absolute adjacent joint step: `0.03947591781616211`
- episode: `len09_trial033`

This scan did not reveal large wrap-like discontinuities in the stored arm
joint positions. A later canonicalization step should still preserve raw values
and document any selected angle representation.

## 9. Reusable Cogrobot Code

Reusable dataset and raw generation code:

- `corsi/data/motion_raw.py`
- `corsi/data/generate_raw.py`
- `corsi/experiments/visual_base/scripts/export_robosuite_ee_xy_dataset.py`
- `corsi/experiments/visual_base/scripts/ee_xy_dataset_utils.py`

Reusable robosuite/Corsi environment helpers:

- `corsi/envs/robosuite_corsi.py`
  - `create_env`
  - `init_sequence_state`
  - `step_pointing_policy`
  - `collect_motion_state`
  - `render_tuned_free_camera_frame`
- `corsi/envs/sequence_generator.py`

Reusable visual/model/training patterns:

- `corsi/models/lstm_visual.py`
  - `FrameCNNEncoder`
  - existing visual LSTM conventions
- `corsi/training/train_visual.py`
  - config parsing, checkpointing, deterministic splits, metric logging patterns
- `corsi/evaluate_visual.py`
  - checkpoint loading and evaluation-output patterns
- `corsi/analysis/metrics.py`
  - existing metric aggregation style

These should be reused after the data gate is corrected. The exact baseline
requested in the goal file still needs a new experiment package because it
requires a custom instrumented LSTM cell, causal masking semantics, canonical
timestamp sampling, split fingerprints, shuffled/zero-vision diagnostics, and
state export.

## 10. corsiFEP Concepts Borrowed Or Excluded

No local `corsiFEP` file or directory was found in the repository or Codex
attachments during this audit. `Idea.md`, paper, and supplement files were also
not found by filename search.

Borrowed concepts for this task:

- none from a local `corsiFEP` source could be verified;
- the current requested baseline is limited to causal sensorimotor sequence
  prediction.

Intentionally excluded by the requested scientific scope:

- active inference
- PV-RNN
- planning or closed-loop robot execution
- encode/recall/delay phases
- visual reconstruction
- working-memory slot interpretation
- attention/VWM modules
- PCA/RSA/TDA/probing/perturbation analyses

## Revised Baseline Decision

The baseline target is revised to seven-joint Panda arm prediction:

`q_hat_{t+1} = f(I_0...I_t, q_0...q_t)`, where
`q_t = [robot0_joint1, robot0_joint2, robot0_joint3, robot0_joint4,
robot0_joint5, robot0_joint6, robot0_joint7]`.

The canonical dataset and all model heads must therefore use `joint_dim = 7`.
