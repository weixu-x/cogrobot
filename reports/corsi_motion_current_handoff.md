# Corsi Motion Current Handoff

Date: 2026-06-22

This file summarizes the current Corsi 7-joint motion baseline state, local data,
experiment outputs, and recommended next steps. Use the conda environment
`robosuite` for all commands.

## Current Workspace

- Branch: `codex/corsi-motion`
- Current commit: `780d101e`
- Main environment: `conda run -n robosuite ...`
- CUDA is available when commands are run outside the sandbox.
- `corsi_artifacts/` is local generated data and is ignored by Git.

The obsolete scratch experiment line has been removed from the current
worktree. The remaining active work is the 7-joint baseline/posthoc/
visualization line plus this documentation cleanup.

## Whole-Repository Uncommitted Change Inventory

This inventory is based on the full current Git worktree, not just the latest
posthoc conversation.

### Tracked Cleanup Changes

These tracked changes remove the obsolete motion package and update package
exports / handoff instructions:

```text
M AGENTS.md
M corsi/data/__init__.py
M corsi/models/__init__.py
D obsolete motion package files and tests
```

Purpose:

- Removes obsolete motion dataset/collate/model/test code.
- Removes package exports that pointed at the deleted modules.
- Keeps the SCALA raw dataset handoff and current validation commands.

### Untracked New Files And Directories

These files are new relative to Git:

```text
corsi/experiments/corsi_motion_baseline/
reports/
tests/test_corsi_motion_baseline.py
tests/test_corsi_motion_posthoc.py
tests/test_corsi_prediction_visualization.py
```

Breakdown:

#### 7-Joint Baseline / Posthoc Package

```text
corsi/experiments/corsi_motion_baseline/__init__.py
corsi/experiments/corsi_motion_baseline/canonicalize.py
corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json
corsi/experiments/corsi_motion_baseline/dataset.py
corsi/experiments/corsi_motion_baseline/evaluate.py
corsi/experiments/corsi_motion_baseline/extract_states.py
corsi/experiments/corsi_motion_baseline/model.py
corsi/experiments/corsi_motion_baseline/posthoc_suite.py
corsi/experiments/corsi_motion_baseline/run_suite.py
corsi/experiments/corsi_motion_baseline/train.py
corsi/experiments/corsi_motion_baseline/visualize_predictions.py
```

Purpose:

- Builds and loads the canonical 7-joint Corsi motion dataset.
- Trains/evaluates `visual_joint`, `joint_only`, and `persistence` baselines.
- Implements convergence audit, conditional continuation, Scheme 1/2/3 motion
  accuracy, FK validation, final visual state HDF5 export, and posthoc report
  generation.
- Adds visualization export for teacher-forced and autoregressive prediction
  videos / montages.

Recommendation:

- Keep this package. It is the main 7-joint motion baseline and posthoc result
  line.
- Commit it separately from report-only documentation if you want a narrow
  code review.

#### New Tests

```text
tests/test_corsi_motion_baseline.py
tests/test_corsi_motion_posthoc.py
tests/test_corsi_prediction_visualization.py
```

Purpose:

- `test_corsi_motion_baseline.py`: canonical sampling, normalization,
  collation, recurrent masking, boundary behavior, persistence, CPU/GPU forward,
  and checkpoint resume restoration.
- `test_corsi_motion_posthoc.py`: convergence classification, exact resume
  requirements, warm-start metadata, step tolerance, endpoint indexing,
  nearest-block / physical-hit logic, joint-limit violation handling,
  deterministic evaluation, local FK gate, and autoregressive feedback.
- `test_corsi_prediction_visualization.py`: validation-selected checkpoint
  resolution, representative seed selection, teacher-forced / autoregressive
  prediction alignment, renderer qpos mapping, FK validation, prediction export,
  video writing, and episode selection.

Recommendation:

- Keep with the corresponding baseline/posthoc/visualization code.

#### Reports And Figures

```text
reports/corsi_motion_baseline_audit.md
reports/corsi_motion_baseline_final.md
reports/corsi_motion_convergence_accuracy.md
reports/corsi_motion_convergence_audit.md
reports/corsi_motion_current_handoff.md
reports/corsi_prediction_visualization_audit.md
reports/corsi_prediction_visualization_report.md
reports/figures/corsi_motion_accuracy_overview.png
reports/figures/corsi_validation_rmse_original_vs_final.png
reports/make_corsi_posthoc_figures.py
```

Purpose:

- Baseline audit/final reports from the original 7-joint baseline work.
- Convergence and motion-level accuracy reports from posthoc analysis.
- Prediction visualization audit/report.
- Two compact explanatory figures and their generation script.
- This handoff file.

Recommendation:

- Keep the markdown reports as experiment documentation.
- Keep figures and the figure script if you want a compact visual summary for
  presentations or review.

### Local Generated Artifacts Ignored By Git

These are not shown as untracked by Git because `corsi_artifacts/` is ignored,
but they are important current local outputs:

```text
corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50
corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12
corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1
corsi_artifacts/motion_baseline/visualizations_v1
```

Recommendation:

- Do not delete `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50`
  or the canonical/runs/posthoc outputs unless intentionally reclaiming local
  disk after external backup.
- Treat `visualizations_v1` as generated media output; it can be regenerated
  from `visualize_predictions.py` if checkpoints and canonical data remain.

### Suggested Commit Split

1. 7-joint motion baseline:
   - `corsi/experiments/corsi_motion_baseline/` except optionally
     `visualize_predictions.py`
   - `tests/test_corsi_motion_baseline.py`
   - baseline reports

2. Posthoc convergence/accuracy:
   - `posthoc_suite.py`, `extract_states.py`, related report outputs,
     `tests/test_corsi_motion_posthoc.py`
   - This can be combined with commit 1 if you want one baseline/posthoc PR.

3. Prediction visualization:
   - `visualize_predictions.py`
   - `tests/test_corsi_prediction_visualization.py`
   - visualization reports and generated local `visualizations_v1` artifacts

4. Handoff / figures:
   - `AGENTS.md`
   - `reports/corsi_motion_current_handoff.md`
   - `reports/make_corsi_posthoc_figures.py`
   - `reports/figures/*.png`

## Data

### Raw Motion Dataset

- Root: `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50`
- Manifest: `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50/manifest.json`
- Episodes: `400`
- Lengths: `2` through `9`
- Count per length: `50`
- Sequence rule: ordered `block_order`, no repeated block within an episode.

Each raw episode contains synchronized per-frame arrays:

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

### Canonical 7-Joint Dataset

- Root: `corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12`
- Manifest: `corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12/manifest.json`
- Schema: `scala_corsi_motion_canonical_7joint_v1`
- Fingerprint: `116565bf565b42de0ebeabc97b484d153c08d2c4d17fd0560144887220a5dbf8`
- Joint dimension: `7`
- Samples per segment: `K=12`
- Total episodes: `400`
- Split counts: train `320`, val `40`, test `40`
- Length counts: `50` episodes each for lengths `2..9`
- Split seed: `20260622`

Config:

```text
corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json
```

Important fixed config values:

- `joint_names`: `robot0_joint1` through `robot0_joint7`
- `image_size`: `128`
- `batch_size`: `16`
- `learning_rate`: `0.001`
- `weight_decay`: `0.00001`
- `max_epochs`: `300`
- `min_epochs`: `100`
- original early stopping patience: `50`
- continuation patience: `75` when exact early-stop state cannot be restored
- `amp`: `true`
- seeds: `0, 5, 10, 15, 20`

Do not regenerate or modify raw/canonical data unless intentionally starting a
new dataset version.

## Model Structure

Code root:

```text
corsi/experiments/corsi_motion_baseline/
```

Implemented model families:

- `visual_joint`
- `joint_only`
- `persistence`

Prediction target:

- Predict normalized `q_{t+1}` from causal history of normalized `q_t`.
- `visual_joint` additionally uses normalized `image_t`.
- Output is 7 Panda arm joint values.

Architecture:

- `VisualEncoder`: random-init CNN for RGB `3x128x128` frames, output dim `128`.
- `JointEncoder`: MLP `7 -> 32 -> 32`.
- Fusion:
  - visual model: `[visual_feature(128), joint_feature(32)] -> 128`
  - joint-only model: `joint_feature(32) -> 128`
- Recurrent core: custom `InstrumentedLSTMCell`, hidden dim `128`.
- Output head: linear `128 -> 7`.
- State export records `h_t`, `c_t`, LSTM gates, visual/joint/fused features,
  joint inputs, predictions, and targets.

Parameter counts from training summaries:

- `visual_joint`: `342,919`
- `joint_only`: `138,855`

Loss / mask audit:

- Training uses `loss_mask == transition_mask`.
- Canonical `transition_mask` equals
  `within_segment_transition | segment_boundary_transition`.
- Therefore the original training objective includes both within-segment and
  segment-boundary transitions, excluding only the final timestep of each
  episode.
- Continuation runs preserve this objective and do not silently change masks.

## Training And Convergence

Original checkpoints contain model state, optimizer state, epoch, best metric,
config, normalization, and joint names. Original checkpoints do not contain
scheduler, AMP scaler, or RNG states, so exact latest-state resume is not
available for original runs. Required extensions are explicitly labeled
warm-start from `best.pt`.

Future checkpoints now save:

- model state
- optimizer state
- scaler state
- scheduler state field
- torch RNG state
- CUDA RNG state list
- numpy RNG state
- Python RNG state

Posthoc root:

```text
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1
```

Main resumable command:

```bash
conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.posthoc_suite \
  --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json \
  --resume \
  --devices auto
```

This command audits histories, classifies convergence, skips completed
continuations, selects final checkpoints by validation RMSE only, runs
Scheme 1/2/3 evaluations, exports final visual states if needed, and rewrites
aggregate outputs.

## Final Checkpoints

Final selection is validation-only, then test metrics are computed afterward.

| Model | Seed | Final source | Final best val RMSE |
| --- | ---: | --- | ---: |
| visual_joint | 0 | original | 0.041656 |
| visual_joint | 5 | extension | 0.035686 |
| visual_joint | 10 | original | 0.044323 |
| visual_joint | 15 | original | 0.044294 |
| visual_joint | 20 | original | 0.046827 |
| joint_only | 0 | extension | 0.028369 |
| joint_only | 5 | extension | 0.031084 |
| joint_only | 10 | extension | 0.029424 |
| joint_only | 15 | extension | 0.027191 |
| joint_only | 20 | extension | 0.026252 |

Extended runs:

- `visual_joint_seed5_extended_continuation_from_best`: epoch `900`, best val `0.035686`
- `joint_only_seed0_extended_continuation_from_best`: epoch `864`, best val `0.028369`; stopped at epoch `865` because of `non_finite_gradient_joint_encoder.net.0.weight_epoch_865`
- `joint_only_seed5_extended_continuation_from_best`: epoch `587`, best val `0.031084`
- `joint_only_seed10_extended_continuation_from_best`: epoch `600`, best val `0.029424`
- `joint_only_seed15_extended_continuation_from_best`: epoch `869`, best val `0.027191`
- `joint_only_seed20_extended_continuation_from_best`: epoch `754`, best val `0.026252`

Lineage:

```text
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/convergence/continuation_lineage.json
```

Final checkpoint selection:

```text
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/convergence/final_checkpoint_selection.json
```

## Accuracy Results

Interpretation limits:

- Timestep tolerance accuracy is a thresholded continuous-regression metric,
  not classification accuracy.
- Token accuracy means segment-endpoint block identity or target hit.
- Teacher-forced full accuracy is not closed-loop robot success.
- Autoregressive evaluation uses exogenous recorded images, so it is still not
  closed-loop visual-motor execution.
- These metrics evaluate motion prediction, not Corsi working-memory recall.

### Step Accuracy Summary

| Group | Normalized RMSE mean | Step acc @1deg | Step acc @5deg |
| --- | ---: | ---: | ---: |
| joint_only_normal | 0.027432 | 0.987385 | 1.000000 |
| visual_joint_normal | 0.040667 | 0.946846 | 1.000000 |
| visual_joint_shuffled_vision | 0.070763 | 0.817385 | 0.999000 |
| visual_joint_zero_vision | 0.305192 | 0.044538 | 0.756308 |
| persistence_normal | 0.173469 | 0.323077 | 0.972308 |

Full threshold tables and tolerance curves:

```text
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/accuracy/step_accuracy_per_seed.json
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/accuracy/step_accuracy_aggregate.json
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/accuracy/tolerance_curves.csv
```

The tolerance curve covers `0.0` to `5.0` degrees in `0.05` degree increments.

### Token And Full-Trial Accuracy

Teacher-forced normal conditions are perfect at the token/full-trial level for
visual, joint-only, and persistence under nearest-block and 5 mm physical-hit
definitions. The zero-vision diagnostic degrades substantially, confirming the
visual branch affects predictions.

Important outputs:

```text
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/accuracy/segment_endpoint_predictions.parquet
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/accuracy/token_accuracy_per_seed.json
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/accuracy/full_trial_predictions.parquet
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/accuracy/full_accuracy_per_seed.json
```

Autoregressive diagnostic:

```text
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/accuracy/autoregressive_metrics.json
```

Mode name:

```text
open_loop_joint_feedback_with_exogenous_images
```

This mode feeds predicted joint feedback after `t=0`, keeps recurrent state
across segment boundaries, and continues to feed prerecorded image observations.

## FK Validation

FK evaluator:

- Environment: local `CorsiSceneDemo`
- Robot: `PandaDexRH`
- EE/contact site: `gripper0_right_index_tip_site`
- Arm qpos indexes: read from `robot._ref_arm_joint_pos_indexes`
- Fixed hand qpos: median hand qpos from endpoint samples
- Block geometry: read from actual MuJoCo geom/body transforms and sizes

Validation gate:

- Samples: `220`
- Median position error: `0.042 mm`
- Max position error: `0.131 mm`
- Passed: `true`

FK outputs:

```text
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/fk/fk_validation.json
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/fk/identified_joint_and_site_mapping.json
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/fk/block_geometry.json
```

## Final Visual State HDF5

Final visual states were exported only for the visual checkpoint that changed:
`visual_joint_seed5_extended_continuation_from_best`.

Files:

```text
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/states/visual_joint_seed5_extended_continuation_from_best_train.h5
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/states/visual_joint_seed5_extended_continuation_from_best_val.h5
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/states/visual_joint_seed5_extended_continuation_from_best_test.h5
```

Test HDF5 structure:

- rows: `2640`
- attributes include canonical fingerprint, checkpoint, seed, split
- key datasets include:
  - `joint_physical`, `prediction_physical`, `target_physical`
  - `joint_normalized`, `prediction_normalized`, `target_normalized`
  - `h_t`, `c_t`
  - `input_gate`, `forget_gate`, `candidate`, `output_gate`
  - `visual_feature`, `joint_feature`, `fused_feature`
  - `seq_id`, `length`, `rank`, `block_id`, `timestep`

View HDF5:

```bash
conda run -n robosuite python -c "import h5py; p='corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/states/visual_joint_seed5_extended_continuation_from_best_test.h5'; f=h5py.File(p,'r'); print(dict(f.attrs)); [print(k, f[k].shape, f[k].dtype) for k in sorted(f.keys())]; f.close()"
```

## Figures

Generated explanatory figures:

```text
reports/figures/corsi_validation_rmse_original_vs_final.png
reports/figures/corsi_motion_accuracy_overview.png
```

Figure script:

```text
reports/make_corsi_posthoc_figures.py
```

Regenerate:

```bash
conda run -n robosuite python reports/make_corsi_posthoc_figures.py
```

## Reports

Primary reports:

```text
reports/corsi_motion_convergence_audit.md
reports/corsi_motion_convergence_accuracy.md
reports/corsi_motion_current_handoff.md
```

The final report contains:

- original and final checkpoint per seed
- 30/50 epoch convergence statistics
- extension decisions and validation gains
- final test metrics after validation-only selection
- Scheme 1 step tolerance results
- Scheme 2 endpoint/token results
- Scheme 3 full-trial and autoregressive diagnostics
- FK validation
- failures/unavailable histories
- commands and runtime

## Tests And Validation Commands

Latest checks run successfully:

```bash
conda run -n robosuite python -m py_compile \
  corsi/experiments/corsi_motion_baseline/train.py \
  corsi/experiments/corsi_motion_baseline/posthoc_suite.py \
  tests/test_corsi_motion_baseline.py \
  tests/test_corsi_motion_posthoc.py
```

```bash
conda run -n robosuite python -m pytest \
  tests/test_corsi_motion_posthoc.py \
  tests/test_corsi_motion_baseline.py -q
```

Result: `32 passed`.

```bash
conda run -n robosuite python -m pytest \
  tests/test_corsi_heatmaps.py \
  tests/test_corsi_attention.py \
  tests/test_corsi_motion_baseline.py -q
```

Result: `20 passed`.

```bash
conda run -n robosuite python -m corsi.data.generate_raw --help
```

Result: help text printed successfully. Robosuite printed only macro/controller
warnings.

## Next TODO

1. Decide commit split.
   - Keep Corsi 7-joint baseline/posthoc work in one focused commit or PR.

2. Review final report results before making claims.
   - Teacher-forced token/full metrics are very high and should be described as
     endpoint prediction metrics, not task success.
   - Autoregressive metrics are much lower and are the better diagnostic for
     compounding error.

3. Investigate `joint_only_seed0` non-finite gradient during the 900 extension.
   - Best checkpoint remains valid and selected by validation RMSE.
   - The failure is recorded in the report and lineage.
   - If training stability matters, inspect LR/AMP/gradient clipping behavior
     for long continuations.

4. If a full-trajectory dataset is reintroduced later, design it as a new
   dataset version instead of modifying this canonical `K=12` baseline in place.
   - Preserve the existing raw/canonical 7-joint baseline.
   - Add a new dense canonical loader or a new raw-to-dense canonicalization step.
   - Decide between fixed inter-block frame count and fixed sampled sequence
     length before writing model-facing tensors.

5. Add a notebook or small CLI if deeper HDF5 inspection is needed.
   - Useful views: gate statistics by length/rank, hidden-state PCA/UMAP, and
     endpoint error by segment rank.

6. Optional: export a concise slide/figure package from:
   - `reports/figures/corsi_validation_rmse_original_vs_final.png`
   - `reports/figures/corsi_motion_accuracy_overview.png`
   - selected tables from `reports/corsi_motion_convergence_accuracy.md`

## Files Most Relevant To This Work

Code:

```text
corsi/experiments/corsi_motion_baseline/canonicalize.py
corsi/experiments/corsi_motion_baseline/dataset.py
corsi/experiments/corsi_motion_baseline/model.py
corsi/experiments/corsi_motion_baseline/train.py
corsi/experiments/corsi_motion_baseline/evaluate.py
corsi/experiments/corsi_motion_baseline/extract_states.py
corsi/experiments/corsi_motion_baseline/posthoc_suite.py
corsi/experiments/corsi_motion_baseline/run_suite.py
corsi/experiments/corsi_motion_baseline/visualize_predictions.py
```

Tests:

```text
tests/test_corsi_motion_baseline.py
tests/test_corsi_motion_posthoc.py
tests/test_corsi_prediction_visualization.py
```

Local outputs:

```text
corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12
corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12
corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1
reports/corsi_motion_convergence_audit.md
reports/corsi_motion_convergence_accuracy.md
reports/figures/
```
