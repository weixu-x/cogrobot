# Model V1 Snapshot

Date: 2026-06-24

Branch: `codex/corsi-motion`

Current commit: `6f7419c848c278c462ffca6768d105c303f417bb`

Artifact training commit recorded in run summaries: `780d101ec28284478261cfe6c8064e83076badf2`

Status: frozen for comparison before designing Model V2. No training or model-code changes were made for this snapshot.

## Purpose

Model V1 is the current Corsi 7-joint Panda arm motion-prediction baseline. It predicts the next normalized 7-joint arm state from the current and previous teacher-forced trajectory observations:

```text
q_hat_{t+1} = f(I_0..I_t, q_0..q_t)
```

This snapshot records the model, data, commands, metrics, and local artifact paths needed to reproduce or compare against V1. It is a motion-prediction baseline, not a closed-loop controller and not evidence of Corsi working-memory behavior.

## What Was Implemented

- Raw Corsi robot motion generation for lengths 2 through 9, 50 episodes per length.
- Canonical `K=12` per-segment sampling into `corsi_motion_7joint_k12`.
- Two learned baselines:
  - `visual_joint`: RGB image plus normalized 7-joint vector.
  - `joint_only`: normalized 7-joint vector only.
- One non-learned persistence baseline: `q_hat_{t+1} = q_t`.
- Train, evaluate, posthoc convergence/accuracy, state export, and visualization tooling for the 7-joint baseline.
- Posthoc validation-selected checkpoint choice, including warm-start continuation runs where the original run was still improving.

## Model Structure

Model name/version: `Model V1`, `corsi_motion_7joint_k12`, 7-joint causal motion-prediction baseline.

Main implementation files:

- `corsi/experiments/corsi_motion_baseline/model.py`
- `corsi/experiments/corsi_motion_baseline/dataset.py`
- `corsi/experiments/corsi_motion_baseline/train.py`
- `corsi/experiments/corsi_motion_baseline/evaluate.py`
- `corsi/experiments/corsi_motion_baseline/run_suite.py`
- `corsi/experiments/corsi_motion_baseline/canonicalize.py`
- `corsi/experiments/corsi_motion_baseline/posthoc_suite.py`
- `corsi/experiments/corsi_motion_baseline/extract_states.py`
- `corsi/experiments/corsi_motion_baseline/visualize_predictions.py`
- `corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json`

Architecture summary:

| Model | Inputs | Structure | Parameters |
|---|---|---|---:|
| `visual_joint` | RGB `[3,128,128]` plus normalized 7-joint vector | CNN visual encoder `3x128x128 -> 128`; joint MLP `7 -> 32`; fusion MLP `160 -> 128`; instrumented LSTM hidden 128; linear head `128 -> 7` | 342,919 |
| `joint_only` | normalized 7-joint vector | joint MLP `7 -> 32`; fusion MLP `32 -> 128`; same instrumented LSTM and linear head | 138,855 |
| `persistence` | normalized 7-joint vector | non-learned copy baseline | 0 |

The LSTM exposes `h_t`, `c_t`, `input_gate`, `forget_gate`, `candidate`, and `output_gate` for state analysis. These traces should be treated as dynamics-analysis features, not as working-memory slots.

## Data Used

Raw source dataset:

- Root: `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50`
- Manifest: `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50/manifest.json`
- Schema: `scala_corsi_motion_raw_v1`
- Size: about `1.1G`
- Episodes: `400`
- Lengths: `2..9`, `50` per length
- Failed/skipped episodes: `0/0`
- Raw frames: `65,193`
- Raw manifest SHA256: `31be68b089f4e131a6a29ebd239c4a5bb62ebd4def5a2c06c08ff4db4b8673cd`
- `sequence_plan.json` SHA256 from audit: `741c706b0df961b8c9e569c7d057aaa9d5440bfb8ce8449217e935305e0d6ac2`

Raw arrays per episode:

```text
rgb [T,128,128,3] uint8
joint [T,7] float32
joint_velocity [T,7] float32
ee_pose [T,7] float32
ee_xy [T,2] float32
action [T,12] float32
qpos [T,19] float32
qvel [T,19] float32
timestamp [T] float64
rank [T] int64
block_id [T] int64
```

Canonical dataset used by V1:

- Root: `corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12`
- Manifest: `corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12/manifest.json`
- Schema: `scala_corsi_motion_canonical_7joint_v1`
- Size: about `451M`
- Samples: `400`
- Split: train `320`, val `40`, test `40`
- Per-length split: train `40`, val `5`, test `5` for each length `2..9`
- `K=12` canonical samples per block-motion segment
- Total canonical rows: `26,400`
- Sequence length: `T = Corsi length * 12`
- Canonical fingerprint: `116565bf565b42de0ebeabc97b484d153c08d2c4d17fd0560144887220a5dbf8`

Canonical arrays:

```text
images [T,3,128,128] uint8
joints [T,7] float32
transition_mask [T] bool
within_segment_transition [T] bool
segment_boundary_transition [T] bool
rank [T] int64
block_id [T] int64
block_xy [T,2] float32
segment_progress [T] float32
source_frame_index [T] int64
source_timestamp [T] float64
boundary [T] bool
```

## Input / Target Definition

Input at canonical timestep `t`:

- `visual_joint`: normalized RGB `image_t`, normalized 7-joint arm vector `q_t`, and `valid_mask_t`.
- `joint_only`: normalized 7-joint arm vector `q_t` and `valid_mask_t`.

Metadata such as `rank`, `block_id`, `block_xy`, `segment_progress`, and boundary masks is loaded and used for analysis, but it is not an input to the V1 model forward pass.

Prediction target:

```text
target_t = normalized q_{t+1}
```

The training loss is masked MSE between `pred_joints_next` and `targets_next`, with `loss_mask == transition_mask`. This includes both within-segment and segment-boundary transitions. The model does not predict OSC actions, full `qpos`, end-effector pose, block identity, task success, or closed-loop control commands.

Joint order:

```text
robot0_joint1
robot0_joint2
robot0_joint3
robot0_joint4
robot0_joint5
robot0_joint6
robot0_joint7
```

## Training Setup

Primary config:

```text
corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json
```

Training settings:

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | `0.001` |
| Weight decay | `0.00001` |
| Batch size | `16` |
| Max epochs | `300` |
| Min epochs | `100` |
| Early stopping patience | `50` |
| Gradient clip norm | `1.0` |
| AMP | enabled on CUDA |
| Seeds | `0, 5, 10, 15, 20` |

Canonicalization command:

```bash
conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.canonicalize \
  --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json \
  --overwrite
```

Training commands:

```bash
conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.run_suite \
  --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json \
  --model-type joint_only \
  --device cuda:1 \
  --seeds 0 5 10 15 20 \
  --run-suffix _full

conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.train \
  --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json \
  --model-type visual_joint \
  --seed 20 \
  --run-name visual_joint_seed20_full \
  --device cuda:0
```

Posthoc continuation note: original checkpoints did not include scheduler, AMP scaler, or RNG state, so extension runs are warm-start continuations from `best.pt`, not exact latest-state resumes.

## Evaluation Setup

Single-checkpoint evaluation command:

```bash
conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.evaluate \
  --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json \
  --model-type visual_joint \
  --checkpoint corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed20_full/best.pt \
  --split test \
  --mode normal \
  --batch-size 16 \
  --device cuda:0 \
  --output corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed20_full/test_normal.json
```

State export example:

```bash
conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.extract_states \
  --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json \
  --checkpoint corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed0_full/best.pt \
  --split test \
  --output corsi_artifacts/motion_baseline/states/visual_joint_seed0_full_test.h5 \
  --batch-size 16 \
  --device cpu
```

Visualization reproduction command:

```bash
conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.visualize_predictions \
  --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json \
  --models visual_joint,joint_only,persistence \
  --modes teacher_forced,open_loop_joint_feedback_with_exogenous_images \
  --export-video \
  --export-comparison-video \
  --export-montage \
  --resume
```

Validation commands used in prior reports:

```bash
conda run -n robosuite python -m pytest \
  tests/test_corsi_motion_baseline.py \
  tests/test_corsi_motion_posthoc.py \
  tests/test_corsi_prediction_visualization.py -q

conda run -n robosuite python -m pytest \
  tests/test_corsi_heatmaps.py \
  tests/test_corsi_attention.py \
  tests/test_corsi_motion_baseline.py -q
```

Prior reported results:

- `tests/test_corsi_motion_baseline.py tests/test_corsi_motion_posthoc.py tests/test_corsi_prediction_visualization.py`: `32 passed`
- `tests/test_corsi_heatmaps.py tests/test_corsi_attention.py tests/test_corsi_motion_baseline.py`: `20 passed`

## Results

Current posthoc final validation-selected metrics:

Source: `corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/accuracy/step_accuracy_aggregate.json`

| Metric | Value | Source |
|---|---:|---|
| `joint_only_normal` normalized RMSE mean | 0.027432 | posthoc step accuracy aggregate |
| `joint_only_normal` joint MAE deg mean | 0.095649 | posthoc step accuracy aggregate |
| `joint_only_normal` step acc @1deg | 0.987385 | posthoc step accuracy aggregate |
| `visual_joint_normal` normalized RMSE mean | 0.040667 | posthoc step accuracy aggregate |
| `visual_joint_normal` joint MAE deg mean | 0.151154 | posthoc step accuracy aggregate |
| `visual_joint_normal` step acc @1deg | 0.946846 | posthoc step accuracy aggregate |
| `visual_joint_shuffled_vision` normalized RMSE mean | 0.070763 | posthoc step accuracy aggregate |
| `visual_joint_zero_vision` normalized RMSE mean | 0.305192 | posthoc step accuracy aggregate |
| `persistence_normal` normalized RMSE | 0.173469 | posthoc step accuracy aggregate |
| FK validation median error | 0.042 mm | `posthoc_convergence_accuracy_v1/fk/fk_validation.json` |
| FK validation max error | 0.131 mm | `posthoc_convergence_accuracy_v1/fk/fk_validation.json` |

Original pre-posthoc aggregate metrics:

Source: `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/aggregate_metrics.json`

| Model / mode | Test RMSE mean | SD | Joint MAE deg mean | SD |
|---|---:|---:|---:|---:|
| `visual_joint normal` | 0.041966 | 0.002249 | 0.160088 | 0.006927 |
| `visual_joint shuffled_vision` | 0.072485 | 0.005647 | 0.283274 | 0.026038 |
| `visual_joint zero_vision` | 0.292132 | 0.024258 | 1.467600 | 0.238032 |
| `joint_only normal` | 0.032574 | 0.000747 | 0.123552 | 0.004729 |
| `persistence normal` | 0.173469 | 0.000000 | 0.755587 | 0.000000 |

Boundary diagnostics from original aggregate metrics:

| Model | Within-segment RMSE | Boundary RMSE |
|---|---:|---:|
| `visual_joint` | 0.039938 | 0.063136 |
| `joint_only` | 0.029714 | 0.058771 |

Posthoc token and autoregressive diagnostics:

- Teacher-forced token nearest-block and physical-hit accuracy are `1.0` for normal `visual_joint`, `joint_only`, and `persistence`.
- Zero-vision teacher-forced token accuracy drops to nearest-block `0.758182` and physical-hit `0.338182`.
- Autoregressive mode is open-loop joint feedback with recorded exogenous images. It is not closed-loop control.
- Autoregressive token nearest/physical-hit accuracy:
  - `visual_joint_normal`: `0.300909 / 0.173636`
  - `joint_only_normal`: `0.222727 / 0.140909`
  - `persistence`: `0.104545 / 0.000000`

Visualization status:

- Output root: `corsi_artifacts/motion_baseline/visualizations_v1`
- Generated manifest rows: `220`
- Prediction exports: `66`
- Per-model videos: `66`
- Comparison videos: `22`
- Montage groups: `66`
- Failures recorded in posthoc summary: `[]`
- Renderer Tier B unavailable because FK renderer validation exceeded the max threshold: median `0.000774 m`, max `0.005265 m`, threshold max `0.005 m`.

## Artifact Locations

| Artifact | Path | Notes |
|---|---|---|
| raw dataset | `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50` | local generated data, about `1.1G`, ignored by Git |
| raw manifest | `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50/manifest.json` | schema `scala_corsi_motion_raw_v1` |
| canonical dataset | `corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12` | about `451M`, schema `scala_corsi_motion_canonical_7joint_v1` |
| canonical manifest | `corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12/manifest.json` | fingerprint `116565bf565b42de0ebeabc97b484d153c08d2c4d17fd0560144887220a5dbf8` |
| config | `corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json` | V1 canonical/training config |
| run root | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12` | about `120M`; contains checkpoints, curves, summaries, test JSON |
| original checkpoints | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/*_full/best.pt` | original full-run best checkpoints |
| final checkpoint selection | `corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/convergence/final_checkpoint_selection.json` | validation-RMSE-selected final checkpoint list |
| extension checkpoints | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/*_extended_continuation_from_best/best.pt` | warm-start extension best checkpoints |
| training curves | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/*/curves.json` | no `.log`/`.out` training logs found |
| run summaries | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/*/summary.json` | includes best val RMSE, epoch count, parameter count, device |
| original aggregate metrics | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/aggregate_metrics.json` | pre-posthoc test metrics |
| persistence metrics | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/persistence_test_metrics.json` | non-learned baseline |
| posthoc root | `corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1` | about `118M` |
| posthoc step metrics | `corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/accuracy/step_accuracy_aggregate.json` | current validation-selected step metrics |
| posthoc token metrics | `corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/accuracy/token_accuracy_per_seed.json` | teacher-forced endpoint token metrics |
| posthoc full-trial metrics | `corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/accuracy/full_accuracy_per_seed.json` | teacher-forced full-trial metrics |
| autoregressive metrics | `corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/accuracy/autoregressive_metrics.json` | open-loop joint feedback with recorded images |
| FK validation | `corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/fk/fk_validation.json` | local FK pass |
| original visual state traces | `corsi_artifacts/motion_baseline/states/visual_joint_seed*_full_*.h5` | about `565M`, 15 HDF5 files plus JSON sidecars |
| posthoc final state traces | `corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/states` | includes final seed-5 extension visual traces |
| visualizations | `corsi_artifacts/motion_baseline/visualizations_v1` | about `206M`, generated videos/montages/manifests |
| visualization audit | `corsi_artifacts/motion_baseline/visualizations_v1/audit/visualization_audit.json` | checkpoint hashes and selected episodes |
| renderer validation | `corsi_artifacts/motion_baseline/visualizations_v1/audit/renderer_validation.json` | Tier B unavailable; max FK error threshold exceeded |
| generated output manifest | `corsi_artifacts/motion_baseline/visualizations_v1/manifests/generated_outputs.jsonl` | 220 generated rows |
| baseline reports | `reports/corsi_motion_baseline_audit.md`; `reports/corsi_motion_baseline_final.md`; `reports/corsi_motion_convergence_accuracy.md`; `reports/corsi_prediction_visualization_report.md` | human-readable source reports |
| snapshot report | `reports/model_v1_snapshot.md` | this file |

Final selected checkpoint rule:

- Selection source is `final_checkpoint_selection.json`.
- `visual_joint` seed `5` uses `visual_joint_seed5_extended_continuation_from_best/best.pt`.
- `visual_joint` seeds `0`, `10`, `15`, and `20` use original `*_full/best.pt`.
- all `joint_only` seeds `0`, `5`, `10`, `15`, and `20` use `*_extended_continuation_from_best/best.pt`.

## Conclusions

- V1 is a valid 7-joint causal one-step motion-prediction baseline.
- Learned models outperform persistence on normalized next-joint prediction.
- `joint_only` is more accurate than `visual_joint` for this one-step target, which indicates the current task is dominated by smooth proprioceptive dynamics.
- Shuffled and zeroed vision strongly degrade `visual_joint`, so the visual branch is used, even though it does not improve normal one-step RMSE over joint-only.
- Segment-boundary transitions are harder than within-segment transitions.
- Teacher-forced endpoint/token metrics can be high even for persistence, so they should not be overinterpreted as task intelligence.
- Autoregressive diagnostics are much weaker than teacher-forced metrics and still use recorded images, so they do not prove closed-loop robot performance.

## Limitations

- V1 predicts only the next normalized 7-joint state.
- It does not model full robot `qpos`, OSC actions, end-effector pose targets, block identity targets, or task success.
- It has no separate presentation, retention, and recall phases, so hidden states should not be called Corsi memory slots.
- It is teacher-forced during normal evaluation.
- Autoregressive evaluation feeds predicted joints back but still uses prerecorded images.
- Renderer Tier B visualization was unavailable because renderer FK validation exceeded the max threshold by a small amount.
- Original checkpoints do not contain full exact-resume state: scheduler, AMP scaler, and RNG state are missing.
- `joint_only_seed0_extended_continuation_from_best` stopped after the selected best checkpoint because of a non-finite gradient at epoch 865.
- No committed conda environment lockfile was found.
- Raw/canonical manifests do not fully record Python/package versions, OS, rendering backend, GPU state, dirty git state, or numeric freecam settings.
- Training artifact summaries record commit `780d101ec28284478261cfe6c8064e83076badf2`, while the current branch is at `6f7419c848c278c462ffca6768d105c303f417bb`.

## What Not To Carry Forward

- Do not carry forward the assumption that one-step normalized joint prediction is the right V2 target.
- Do not conflate OSC action semantics with 7-joint `q_t` semantics.
- Do not treat V1 teacher-forced token accuracy as closed-loop robot success.
- Do not describe LSTM traces as Corsi working-memory slots without a task design that supports that claim.
- Do not train V2 on deprecated `freecam_motion_segment_uniform_*` datasets.
- Do not overwrite V1 checkpoints, state traces, metrics, visualizations, or raw data.
- Do not modify or break existing `freecam_index`, heatmap, or `freecam_ee_xy` workflows.
- Do not reuse the V1 canonical dataset blindly if V2 needs raw `action`, full `qpos`, `ee_pose`, `ee_xy`, `rank`, `block_id`, or segment-level supervision.

## Next Version Plan

Use the existing raw dataset as the immutable source for V2:

```text
corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50
```

Recommended next step:

1. Create `reports/model_v2_plan.md` before implementation.
2. Explicitly compare V2 against this V1 snapshot.
3. Define the V2 target before writing code.
4. Decide which raw fields V2 uses: `rgb`, `joint`, `joint_velocity`, `ee_pose`, `ee_xy`, `action`, `qpos`, `qvel`, `rank`, `block_id`, `segments`.
5. Keep any new V2 canonical dataset under a new path, not under `corsi_motion_7joint_k12`.
6. Keep V1 artifacts read-only for comparison.
7. Add evaluation criteria that separate teacher-forced prediction, open-loop autoregression, and any real closed-loop robot/replay claim.
