# Corsi Prediction Visualization Audit

Created: 2026-06-22 13:34:36

## Scope

Post-hoc visualization of already-trained causal 7-joint motion prediction baselines.
No retraining, dataset regeneration, checkpoint mutation, or metric changes are performed.

## Checkpoint Selection

Rule: use `posthoc_convergence_accuracy_v1/convergence/final_checkpoint_selection.json` when present; otherwise use each run summary best checkpoint. The selection key is validation RMSE only.

| Model | Seed | Source | Best val RMSE | Checkpoint SHA256 | Path |
| --- | ---: | --- | ---: | --- | --- |
| joint_only | 0 | extension | 0.028369 | `0865712aae808e50...` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/joint_only_seed0_extended_continuation_from_best/best.pt` |
| joint_only | 5 | extension | 0.031084 | `881637188a6e954c...` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/joint_only_seed5_extended_continuation_from_best/best.pt` |
| joint_only | 10 | extension | 0.029424 | `f43e62113a02d7c8...` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/joint_only_seed10_extended_continuation_from_best/best.pt` |
| joint_only | 15 | extension | 0.027191 | `3df8c216e96ef2b7...` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/joint_only_seed15_extended_continuation_from_best/best.pt` |
| joint_only | 20 | extension | 0.026252 | `7f241cc162deef1e...` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/joint_only_seed20_extended_continuation_from_best/best.pt` |
| visual_joint | 0 | original | 0.041656 | `4bef2162351c58f1...` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed0_full/best.pt` |
| visual_joint | 5 | extension | 0.035686 | `47186ed7d74d011e...` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed5_extended_continuation_from_best/best.pt` |
| visual_joint | 10 | original | 0.044323 | `1ae859fc54796887...` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed10_full/best.pt` |
| visual_joint | 15 | original | 0.044294 | `2a336d8e197d3093...` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed15_full/best.pt` |
| visual_joint | 20 | original | 0.046827 | `698a1943ecbe3030...` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed20_full/best.pt` |

Representative visual seed: `15` (median validation RMSE among final visual checkpoints).

## Joint And Qpos Mapping

Joint order: `['robot0_joint1', 'robot0_joint2', 'robot0_joint3', 'robot0_joint4', 'robot0_joint5', 'robot0_joint6', 'robot0_joint7']`

Arm qpos indexes: `[0, 1, 2, 3, 4, 5, 6]`

Fixed hand qpos indexes: `[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]`

Fixed hand qpos values from first raw frame: `[1.306515, 1.170595, 1.207557, 0.791145, -0.013315, 0.009923, 1.488669, 1.795799, 1.594214, 1.743769, 1.593907, 1.86001]`

Joint limit source: `robosuite PandaDexRH MuJoCo model jnt_range`

Joint limits: `[[-2.8973000049591064, 2.8973000049591064], [-1.7627999782562256, 1.7627999782562256], [-2.8973000049591064, 2.8973000049591064], [-3.0717999935150146, -0.0697999969124794], [-2.8973000049591064, 2.8973000049591064], [-0.017500000074505806, 3.752500057220459], [-2.8973000049591064, 2.8973000049591064]]`

## Image And Camera

Input image source: raw/canonical `freecam` RGB, shape `[128, 128, 3]`.

Free camera config: `{'lookat': [0.0, 0.0, 0.9], 'distance': 0.489545, 'azimuth': -179.858241, 'elevation': -63.442729}`

## Blocks And Geometry

Block positions are restored from raw episode metadata. Corsi block geom half-size is recorded as `[0.025, 0.025, 0.015]` m.

EE plots use normalized board XY for both `target_ee_xy_norm` and `block_xy`; FK validation separately uses world-frame `target_ee_pose`.

## Tier Availability

Tier A is always available. Tier B is enabled only if `visualizations_v1/audit/renderer_validation.json` passes FK thresholds.
