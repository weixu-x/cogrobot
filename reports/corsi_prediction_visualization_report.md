# Corsi Prediction Visualization Report

Date: 2026-06-22

These visualizations show predictions from a causal sensorimotor motion model. Teacher-forced videos provide ground-truth current joint angles at each step. Autoregressive videos feed predicted joints back into the model but continue to use prerecorded exogenous images. Neither visualization is a closed-loop robot execution or evidence of Corsi working-memory recall.

## Checkpoints And Selection

Checkpoints are selected from validation RMSE only, using the post-hoc final selection file when present.
Representative visual seed: `15`.

Audit: `reports/corsi_prediction_visualization_audit.md`
Visualization audit JSON: `corsi_artifacts/motion_baseline/visualizations_v1/audit/visualization_audit.json`
Device audit JSON: `corsi_artifacts/motion_baseline/visualizations_v1/audit/device.json`
Requested device: `cuda`; resolved device: `cpu`.
Torch CUDA available in this process: `False`.
Device fallback reason: `requested cuda, but torch.cuda.is_available() is False`.

## Selected Episodes

| Category | Seq id | Length | Reason |
| --- | --- | ---: | --- |
| median_rmse_length_2 | `len02_trial019` | 2 | closest to per-length median visual_joint RMSE, then lexicographic seq_id |
| median_rmse_length_3 | `len03_trial043` | 3 | closest to per-length median visual_joint RMSE, then lexicographic seq_id |
| median_rmse_length_4 | `len04_trial001` | 4 | closest to per-length median visual_joint RMSE, then lexicographic seq_id |
| median_rmse_length_5 | `len05_trial045` | 5 | closest to per-length median visual_joint RMSE, then lexicographic seq_id |
| median_rmse_length_6 | `len06_trial004` | 6 | closest to per-length median visual_joint RMSE, then lexicographic seq_id |
| median_rmse_length_7 | `len07_trial014` | 7 | closest to per-length median visual_joint RMSE, then lexicographic seq_id |
| median_rmse_length_8 | `len08_trial024` | 8 | closest to per-length median visual_joint RMSE, then lexicographic seq_id |
| median_rmse_length_9 | `len09_trial049` | 9 | closest to per-length median visual_joint RMSE, then lexicographic seq_id |
| global_lowest_rmse | `len02_trial024` | 2 | lowest visual_joint RMSE, then lexicographic seq_id |
| global_highest_rmse | `len09_trial015` | 9 | highest visual_joint RMSE, then lexicographic seq_id |
| largest_segment_boundary_error | `len09_trial015` | 9 | highest visual_joint boundary mean MAE, then lexicographic seq_id |
| largest_endpoint_error | `len09_trial015` | 9 | highest endpoint mean joint MAE before FK endpoint EE is available, then lexicographic seq_id |
| visual_best_gain_over_joint_only | `len03_trial015` | 3 | largest joint_only MAE minus visual_joint MAE, then lexicographic seq_id |
| visual_worst_gap_vs_joint_only | `len09_trial049` | 9 | smallest joint_only MAE minus visual_joint MAE, then lexicographic seq_id |

## Prediction Modes

- `teacher_forced`: input at display index `t` is recorded `I_t, q_t`; target is `q_{t+1}`.
- `open_loop_joint_feedback_with_exogenous_images`: starts from recorded `q_0`; after that, predicted joints are fed back while recorded images `I_t` remain exogenous.

Neither mode is a closed-loop robot rollout.

## EE Plot Coordinates

`target_ee_pose` remains the raw world-frame EE pose used for FK validation. Video EE plots use `target_ee_xy_norm`, converted from raw table XY to the same normalized board coordinates as `block_xy`.

## Tier Availability

Tier A: available.
Tier B: unavailable.
Renderer validation: `corsi_artifacts/motion_baseline/visualizations_v1/audit/renderer_validation.json`
Median FK error: `0.0007737657870166004` m.
Max FK error: `0.0052648005075752735` m.
Tier-B reason: `FK error exceeded threshold`.
When Tier B is unavailable, videos retain Tier-A source RGB and numeric joint/EE plots only.

## Generated Outputs

Prediction exports: 66
Per-model videos generated/skipped: 66
Comparison videos generated/skipped: 22
Montage groups generated: 66
Failures: 0

Output root: `corsi_artifacts/motion_baseline/visualizations_v1`
Selected episodes manifest: `corsi_artifacts/motion_baseline/visualizations_v1/manifests/selected_episodes.json`
Generated outputs manifest: `corsi_artifacts/motion_baseline/visualizations_v1/manifests/generated_outputs.jsonl`
Failures manifest: `corsi_artifacts/motion_baseline/visualizations_v1/manifests/failures.jsonl`

## Reproduction

```bash
conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.visualize_predictions \
  --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json \
  --checkpoint-source final-validation-selected \
  --modes teacher_forced,open_loop_joint_feedback_with_exogenous_images \
  --models visual_joint,joint_only,persistence \
  --select median-per-length,best,worst,worst-boundary,worst-endpoint,visual-best-gain,visual-worst-gap \
  --export-video \
  --export-comparison-video \
  --export-montage \
  --resume
```

## Completion Summary

- files changed: `visualize_predictions.py`, visualization tests/report outputs
- tests passed: see validation command output in the assistant summary
- per-model videos generated/skipped: 66
- comparison videos generated/skipped: 22
- montages generated: 66 groups
- failed outputs: 0
- total runtime: 1274.0 seconds
- total storage: 245.3 MiB

Exact resume command:

```bash
conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.visualize_predictions --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json --models visual_joint,joint_only,persistence --modes teacher_forced,open_loop_joint_feedback_with_exogenous_images --export-video --export-comparison-video --export-montage --resume
```
