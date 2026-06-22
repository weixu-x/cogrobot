# Corsi Motion Convergence And Accuracy

Date: 2026-06-22

## Interpretation Limits

- timestep tolerance accuracy is a thresholded continuous-regression metric, not classification accuracy
- token accuracy means segment-endpoint block identity/hit
- teacher-forced full accuracy is not closed-loop robot success
- autoregressive evaluation uses exogenous recorded images and is therefore still not a closed-loop visual-motor execution test
- these metrics evaluate motion prediction, not Corsi working-memory recall

## Loss And Resume Audit

Training loss mask: `training_loss_includes_within_segment_and_segment_boundary_transitions`.
The original checkpoints lack scheduler, AMP scaler, and RNG states, so required extensions are warm-started
from `best.pt` in separate `*_extended_continuation_from_best` run directories.

## Final Checkpoints

| Model | Seed | Original checkpoint | Final checkpoint | Final source | Original best val | Final best val |
| --- | ---: | --- | --- | --- | ---: | ---: |
| visual_joint | 0 | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed0_full/best.pt` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed0_full/best.pt` | original | 0.041656 | 0.041656 |
| visual_joint | 5 | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed5_full/best.pt` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed5_extended_continuation_from_best/best.pt` | extension | 0.043191 | 0.035686 |
| visual_joint | 10 | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed10_full/best.pt` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed10_full/best.pt` | original | 0.044323 | 0.044323 |
| visual_joint | 15 | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed15_full/best.pt` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed15_full/best.pt` | original | 0.044294 | 0.044294 |
| visual_joint | 20 | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed20_full/best.pt` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed20_full/best.pt` | original | 0.046827 | 0.046827 |
| joint_only | 0 | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/joint_only_seed0_full/best.pt` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/joint_only_seed0_extended_continuation_from_best/best.pt` | extension | 0.034439 | 0.028369 |
| joint_only | 5 | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/joint_only_seed5_full/best.pt` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/joint_only_seed5_extended_continuation_from_best/best.pt` | extension | 0.034897 | 0.031084 |
| joint_only | 10 | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/joint_only_seed10_full/best.pt` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/joint_only_seed10_extended_continuation_from_best/best.pt` | extension | 0.034203 | 0.029424 |
| joint_only | 15 | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/joint_only_seed15_full/best.pt` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/joint_only_seed15_extended_continuation_from_best/best.pt` | extension | 0.033231 | 0.027191 |
| joint_only | 20 | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/joint_only_seed20_full/best.pt` | `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/joint_only_seed20_extended_continuation_from_best/best.pt` | extension | 0.032953 | 0.026252 |

## Convergence Decisions

| Model | Seed | Class | Best epoch | Since best | Best val | Final val | Extend reason |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| visual_joint | 0 | UNCERTAIN | 190 | 50 | 0.041656 | 0.042407 | stopped_before_max_by_recorded_early_stopping |
| visual_joint | 5 | STILL_IMPROVING | 294 | 6 | 0.043191 | 0.043422 | reached_epoch_300_and_classified_STILL_IMPROVING |
| visual_joint | 10 | CONVERGED | 138 | 50 | 0.044323 | 0.045152 | stopped_before_max_by_recorded_early_stopping |
| visual_joint | 15 | UNCERTAIN | 176 | 50 | 0.044294 | 0.045420 | stopped_before_max_by_recorded_early_stopping |
| visual_joint | 20 | OVERFITTING | 170 | 50 | 0.046827 | 0.052213 | stopped_before_max_by_recorded_early_stopping |
| joint_only | 0 | STILL_IMPROVING | 300 | 0 | 0.034439 | 0.034439 | reached_epoch_300_and_classified_STILL_IMPROVING |
| joint_only | 5 | STILL_IMPROVING | 299 | 1 | 0.034897 | 0.035775 | reached_epoch_300_and_classified_STILL_IMPROVING |
| joint_only | 10 | STILL_IMPROVING | 296 | 4 | 0.034203 | 0.035989 | reached_epoch_300_and_classified_STILL_IMPROVING |
| joint_only | 15 | STILL_IMPROVING | 299 | 1 | 0.033231 | 0.033886 | reached_epoch_300_and_classified_STILL_IMPROVING |
| joint_only | 20 | STILL_IMPROVING | 296 | 4 | 0.032953 | 0.034511 | reached_epoch_300_and_classified_STILL_IMPROVING |

### Convergence Statistics Over 30 And 50 Epochs

| Model | Seed | Window | Class | Confidence | OLS slope | Theil-Sen slope | Normalized OLS | Normalized Theil-Sen | Relative change | Pred OLS decrease | Pred Theil-Sen decrease | Train-val gap | Val worse while train decreases |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| visual_joint | 0 | 30 | OVERFITTING | normal | -0.00001708 | -0.00000867 | -0.00039154 | -0.00019876 | -0.010457 | 0.011355 | 0.005764 | -0.00130410 | True |
| visual_joint | 0 | 50 | UNCERTAIN | normal | 0.00000120 | -0.00000164 | 0.00002745 | -0.00003778 | 0.006525 | 0.000000 | 0.001851 | -0.00130410 | False |
| visual_joint | 5 | 30 | CONVERGED | normal | -0.00000212 | 0.00001500 | -0.00004745 | 0.00033524 | -0.005115 | 0.001376 | 0.000000 | -0.00132080 | False |
| visual_joint | 5 | 50 | STILL_IMPROVING | normal | -0.00002401 | -0.00002457 | -0.00053298 | -0.00054549 | 0.037297 | 0.026116 | 0.026729 | -0.00132080 | False |
| visual_joint | 10 | 30 | UNCERTAIN | normal | -0.00000597 | -0.00002342 | -0.00012928 | -0.00050724 | 0.003736 | 0.003749 | 0.014710 | -0.00143651 | False |
| visual_joint | 10 | 50 | CONVERGED | normal | 0.00000629 | 0.00000083 | 0.00013627 | 0.00001801 | -0.002840 | 0.000000 | 0.000000 | -0.00143651 | False |
| visual_joint | 15 | 30 | STILL_IMPROVING | normal | -0.00003068 | -0.00001963 | -0.00066936 | -0.00042816 | 0.005996 | 0.019412 | 0.012417 | -0.00151064 | False |
| visual_joint | 15 | 50 | UNCERTAIN | normal | -0.00000611 | -0.00000480 | -0.00013326 | -0.00010465 | 0.005590 | 0.006530 | 0.005128 | -0.00151064 | False |
| visual_joint | 20 | 30 | CONVERGED | normal | 0.00001844 | 0.00001470 | 0.00037120 | 0.00029592 | -0.007626 | 0.000000 | 0.000000 | -0.00198810 | False |
| visual_joint | 20 | 50 | OVERFITTING | normal | 0.00001835 | 0.00002163 | 0.00037051 | 0.00043673 | -0.012354 | 0.000000 | 0.000000 | -0.00198810 | True |
| joint_only | 0 | 30 | STILL_IMPROVING | normal | -0.00006953 | -0.00007516 | -0.00192756 | -0.00208350 | 0.040474 | 0.055899 | 0.060421 | -0.00043390 | False |
| joint_only | 0 | 50 | STILL_IMPROVING | normal | -0.00004526 | -0.00004507 | -0.00124020 | -0.00123491 | 0.060147 | 0.060770 | 0.060511 | -0.00043390 | False |
| joint_only | 5 | 30 | STILL_IMPROVING | normal | -0.00001173 | -0.00001487 | -0.00032253 | -0.00040901 | 0.012209 | 0.009353 | 0.011861 | -0.00058374 | False |
| joint_only | 5 | 50 | STILL_IMPROVING | normal | -0.00003511 | -0.00002944 | -0.00095578 | -0.00080143 | 0.046676 | 0.046833 | 0.039270 | -0.00058374 | False |
| joint_only | 10 | 30 | STILL_IMPROVING | normal | -0.00010997 | -0.00008911 | -0.00299302 | -0.00242524 | 0.056679 | 0.086797 | 0.070332 | -0.00055662 | False |
| joint_only | 10 | 50 | STILL_IMPROVING | normal | -0.00005675 | -0.00005408 | -0.00152970 | -0.00145767 | 0.046166 | 0.074955 | 0.071426 | -0.00055662 | False |
| joint_only | 15 | 30 | STILL_IMPROVING | normal | -0.00005118 | -0.00004909 | -0.00147764 | -0.00141738 | 0.032548 | 0.042851 | 0.041104 | -0.00048060 | False |
| joint_only | 15 | 50 | STILL_IMPROVING | normal | -0.00005021 | -0.00004796 | -0.00142735 | -0.00136342 | 0.047861 | 0.069940 | 0.066808 | -0.00048060 | False |
| joint_only | 20 | 30 | STILL_IMPROVING | normal | -0.00002600 | -0.00003825 | -0.00075252 | -0.00110724 | 0.029195 | 0.021823 | 0.032110 | -0.00045425 | False |
| joint_only | 20 | 50 | STILL_IMPROVING | normal | -0.00002968 | -0.00003825 | -0.00085199 | -0.00109823 | 0.049482 | 0.041748 | 0.053813 | -0.00045425 | False |

Extended runs:

- `visual_joint_seed5_full` -> `visual_joint_seed5_extended_continuation_from_best` warm-start from best epoch 294; final epoch 900, best val 0.035686
- `joint_only_seed0_full` -> `joint_only_seed0_extended_continuation_from_best` warm-start from best epoch 300; final epoch 864, best val 0.028369
- `joint_only_seed5_full` -> `joint_only_seed5_extended_continuation_from_best` warm-start from best epoch 299; final epoch 587, best val 0.031084
- `joint_only_seed10_full` -> `joint_only_seed10_extended_continuation_from_best` warm-start from best epoch 296; final epoch 600, best val 0.029424
- `joint_only_seed15_full` -> `joint_only_seed15_extended_continuation_from_best` warm-start from best epoch 299; final epoch 869, best val 0.027191
- `joint_only_seed20_full` -> `joint_only_seed20_extended_continuation_from_best` warm-start from best epoch 296; final epoch 754, best val 0.026252

## Final Test Metrics

Validation-only checkpoint selection was completed before reading final test accuracy outputs.

| Group | Normalized RMSE mean | Step acc @1deg mean | Step acc @5deg mean |
| --- | ---: | ---: | ---: |
| joint_only_normal | 0.027432 | 0.987385 | 1.000000 |
| visual_joint_normal | 0.040667 | 0.946846 | 1.000000 |
| visual_joint_shuffled_vision | 0.070763 | 0.817385 | 0.999000 |
| visual_joint_zero_vision | 0.305192 | 0.044538 | 0.756308 |
| persistence_normal | 0.173469 | 0.323077 | 0.972308 |

## Step Accuracy Thresholds

### joint_only_normal

| Threshold deg | Mean | SD | Bootstrap 95% CI |
| ---: | ---: | ---: | --- |
| 0.1 | 0.239538 | 0.040795 | [0.212365, 0.271385] |
| 0.25 | 0.699769 | 0.024908 | [0.680604, 0.718923] |
| 0.5 | 0.914923 | 0.010968 | [0.906538, 0.923308] |
| 1.0 | 0.987385 | 0.003021 | [0.985000, 0.989769] |
| 2.0 | 0.998615 | 0.000439 | [0.998308, 0.998923] |
| 5.0 | 1.000000 | 0.000000 | [1.000000, 1.000000] |

### visual_joint_normal

| Threshold deg | Mean | SD | Bootstrap 95% CI |
| ---: | ---: | ---: | --- |
| 0.1 | 0.117538 | 0.131436 | [0.054154, 0.234231] |
| 0.25 | 0.447385 | 0.125927 | [0.371000, 0.560923] |
| 0.5 | 0.754615 | 0.061751 | [0.721538, 0.809231] |
| 1.0 | 0.946846 | 0.013988 | [0.936769, 0.958615] |
| 2.0 | 0.994923 | 0.000918 | [0.994231, 0.995617] |
| 5.0 | 1.000000 | 0.000000 | [1.000000, 1.000000] |

### visual_joint_shuffled_vision

| Threshold deg | Mean | SD | Bootstrap 95% CI |
| ---: | ---: | ---: | --- |
| 0.1 | 0.022000 | 0.019891 | [0.010385, 0.039231] |
| 0.25 | 0.169615 | 0.043967 | [0.138231, 0.207669] |
| 0.5 | 0.479462 | 0.070447 | [0.424615, 0.530000] |
| 1.0 | 0.817385 | 0.042271 | [0.787308, 0.847154] |
| 2.0 | 0.965308 | 0.009050 | [0.957923, 0.970538] |
| 5.0 | 0.999000 | 0.000583 | [0.998615, 0.999538] |

### visual_joint_zero_vision

| Threshold deg | Mean | SD | Bootstrap 95% CI |
| ---: | ---: | ---: | --- |
| 0.1 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| 0.25 | 0.000615 | 0.001173 | [0.000000, 0.001692] |
| 0.5 | 0.005923 | 0.005644 | [0.001615, 0.010231] |
| 1.0 | 0.044538 | 0.032227 | [0.021692, 0.069538] |
| 2.0 | 0.228846 | 0.098520 | [0.153077, 0.304615] |
| 5.0 | 0.756308 | 0.115021 | [0.651769, 0.823788] |

## Token Accuracy

Teacher-forced metrics use endpoint prediction from `e-1` and never the boundary transition from `e`.
- `aggregate_tf_token_nearest_block_acc`: `{"groups": {"joint_only_normal": {"mean": 1.0, "mode": "normal", "model_type": "joint_only", "sd": 0.0, "values": [1.0, 1.0, 1.0, 1.0, 1.0]}, "visual_joint_normal": {"mean": 1.0, "mode": "normal", "model_type": "visual_joint", "sd": 0.0, "values": [1.0, 1.0, 1.0, 1.0, 1.0]}, "visual_joint_shuffled_vision": {"mean": 1.0, "mode": "shuffled_vision", "model_type": "visual_joint", "sd": 0.0, "values": [1.0, 1.0, 1.0, 1.0, 1.0]}, "visual_joint_zero_vision": {"mean": 0.7581818181818181, "mode": "zero_vision", "model_type": "visual_joint", "sd": 0.08722062838875287, "values": [0.85, 0.759090909090909, 0.8272727272727273, 0.6318181818181818, 0.7227272727272728]}}, "persistence": 1.0}`
- `aggregate_tf_token_physical_hit_acc`: `{"groups": {"joint_only_normal": {"mean": 1.0, "mode": "normal", "model_type": "joint_only", "sd": 0.0, "values": [1.0, 1.0, 1.0, 1.0, 1.0]}, "visual_joint_normal": {"mean": 1.0, "mode": "normal", "model_type": "visual_joint", "sd": 0.0, "values": [1.0, 1.0, 1.0, 1.0, 1.0]}, "visual_joint_shuffled_vision": {"mean": 0.9845454545454546, "mode": "shuffled_vision", "model_type": "visual_joint", "sd": 0.014588005941710202, "values": [0.9863636363636363, 0.990909090909091, 0.9954545454545455, 0.9590909090909091, 0.990909090909091]}, "visual_joint_zero_vision": {"mean": 0.3381818181818182, "mode": "zero_vision", "model_type": "visual_joint", "sd": 0.13172410435991388, "values": [0.4318181818181818, 0.2772727272727273, 0.509090909090909, 0.17727272727272728, 0.29545454545454547]}}, "persistence": 1.0}`

### Teacher-Forced Endpoint Error And Token Accuracy

| Group | Endpoint EE err cm | Endpoint XY err cm | Endpoint Z err cm | tf_token_nearest_block_acc | tf_token_physical_hit_acc | Hit 0/5/10mm |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| joint_only_normal | 0.1337 | 0.1275 | 0.0290 | 1.000000 | 1.000000 | 0.990909/1.000000/1.000000 |
| persistence | 0.2427 | 0.2332 | 0.0311 | 1.000000 | 1.000000 | 0.877273/1.000000/1.000000 |
| visual_joint_normal | 0.2179 | 0.2089 | 0.0451 | 1.000000 | 1.000000 | 0.961818/1.000000/1.000000 |
| visual_joint_shuffled_vision | 0.4324 | 0.4234 | 0.0581 | 1.000000 | 0.984545 | 0.853636/0.984545/1.000000 |
| visual_joint_zero_vision | 2.5360 | 2.4948 | 0.3155 | 0.758182 | 0.338182 | 0.200000/0.338182/0.460000 |

### Teacher-Forced Token Accuracy By Length

| Group | Length | tf_token_nearest_block_acc | tf_token_physical_hit_acc |
| --- | ---: | ---: | ---: |
| joint_only_normal | 2 | 1.000000 | 1.000000 |
| joint_only_normal | 3 | 1.000000 | 1.000000 |
| joint_only_normal | 4 | 1.000000 | 1.000000 |
| joint_only_normal | 5 | 1.000000 | 1.000000 |
| joint_only_normal | 6 | 1.000000 | 1.000000 |
| joint_only_normal | 7 | 1.000000 | 1.000000 |
| joint_only_normal | 8 | 1.000000 | 1.000000 |
| joint_only_normal | 9 | 1.000000 | 1.000000 |
| persistence | 2 | 1.000000 | 1.000000 |
| persistence | 3 | 1.000000 | 1.000000 |
| persistence | 4 | 1.000000 | 1.000000 |
| persistence | 5 | 1.000000 | 1.000000 |
| persistence | 6 | 1.000000 | 1.000000 |
| persistence | 7 | 1.000000 | 1.000000 |
| persistence | 8 | 1.000000 | 1.000000 |
| persistence | 9 | 1.000000 | 1.000000 |
| visual_joint_normal | 2 | 1.000000 | 1.000000 |
| visual_joint_normal | 3 | 1.000000 | 1.000000 |
| visual_joint_normal | 4 | 1.000000 | 1.000000 |
| visual_joint_normal | 5 | 1.000000 | 1.000000 |
| visual_joint_normal | 6 | 1.000000 | 1.000000 |
| visual_joint_normal | 7 | 1.000000 | 1.000000 |
| visual_joint_normal | 8 | 1.000000 | 1.000000 |
| visual_joint_normal | 9 | 1.000000 | 1.000000 |
| visual_joint_shuffled_vision | 2 | 1.000000 | 1.000000 |
| visual_joint_shuffled_vision | 3 | 1.000000 | 0.986667 |
| visual_joint_shuffled_vision | 4 | 1.000000 | 0.990000 |
| visual_joint_shuffled_vision | 5 | 1.000000 | 1.000000 |
| visual_joint_shuffled_vision | 6 | 1.000000 | 0.993333 |
| visual_joint_shuffled_vision | 7 | 1.000000 | 0.982857 |
| visual_joint_shuffled_vision | 8 | 1.000000 | 0.975000 |
| visual_joint_shuffled_vision | 9 | 1.000000 | 0.973333 |
| visual_joint_zero_vision | 2 | 0.760000 | 0.460000 |
| visual_joint_zero_vision | 3 | 0.733333 | 0.306667 |
| visual_joint_zero_vision | 4 | 0.680000 | 0.220000 |
| visual_joint_zero_vision | 5 | 0.736000 | 0.312000 |
| visual_joint_zero_vision | 6 | 0.820000 | 0.380000 |
| visual_joint_zero_vision | 7 | 0.782857 | 0.342857 |
| visual_joint_zero_vision | 8 | 0.750000 | 0.370000 |
| visual_joint_zero_vision | 9 | 0.760000 | 0.328889 |

## Full-Trial Accuracy

- `aggregate_tf_full_nearest_block_acc`: `{"groups": {"joint_only_normal": {"mean": 1.0, "mode": "normal", "model_type": "joint_only", "sd": 0.0, "values": [1.0, 1.0, 1.0, 1.0, 1.0]}, "visual_joint_normal": {"mean": 1.0, "mode": "normal", "model_type": "visual_joint", "sd": 0.0, "values": [1.0, 1.0, 1.0, 1.0, 1.0]}, "visual_joint_shuffled_vision": {"mean": 1.0, "mode": "shuffled_vision", "model_type": "visual_joint", "sd": 0.0, "values": [1.0, 1.0, 1.0, 1.0, 1.0]}, "visual_joint_zero_vision": {"mean": 0.195, "mode": "zero_vision", "model_type": "visual_joint", "sd": 0.11374313166077325, "values": [0.325, 0.175, 0.3, 0.075, 0.1]}}, "persistence": 1.0}`
- `aggregate_tf_full_physical_hit_acc`: `{"groups": {"joint_only_normal": {"mean": 1.0, "mode": "normal", "model_type": "joint_only", "sd": 0.0, "values": [1.0, 1.0, 1.0, 1.0, 1.0]}, "visual_joint_normal": {"mean": 1.0, "mode": "normal", "model_type": "visual_joint", "sd": 0.0, "values": [1.0, 1.0, 1.0, 1.0, 1.0]}, "visual_joint_shuffled_vision": {"mean": 0.915, "mode": "shuffled_vision", "model_type": "visual_joint", "sd": 0.08023403267940604, "values": [0.925, 0.95, 0.975, 0.775, 0.95]}, "visual_joint_zero_vision": {"mean": 0.035, "mode": "zero_vision", "model_type": "visual_joint", "sd": 0.041833001326703784, "values": [0.05, 0.0, 0.1, 0.0, 0.025]}}, "persistence": 1.0}`

### Teacher-Forced Full-Trial Accuracy By Length

| Group | Length | tf_full_nearest_block_acc | tf_full_physical_hit_acc |
| --- | ---: | ---: | ---: |
| joint_only_normal | 2 | 1.000000 | 1.000000 |
| joint_only_normal | 3 | 1.000000 | 1.000000 |
| joint_only_normal | 4 | 1.000000 | 1.000000 |
| joint_only_normal | 5 | 1.000000 | 1.000000 |
| joint_only_normal | 6 | 1.000000 | 1.000000 |
| joint_only_normal | 7 | 1.000000 | 1.000000 |
| joint_only_normal | 8 | 1.000000 | 1.000000 |
| joint_only_normal | 9 | 1.000000 | 1.000000 |
| persistence | 2 | 1.000000 | 1.000000 |
| persistence | 3 | 1.000000 | 1.000000 |
| persistence | 4 | 1.000000 | 1.000000 |
| persistence | 5 | 1.000000 | 1.000000 |
| persistence | 6 | 1.000000 | 1.000000 |
| persistence | 7 | 1.000000 | 1.000000 |
| persistence | 8 | 1.000000 | 1.000000 |
| persistence | 9 | 1.000000 | 1.000000 |
| visual_joint_normal | 2 | 1.000000 | 1.000000 |
| visual_joint_normal | 3 | 1.000000 | 1.000000 |
| visual_joint_normal | 4 | 1.000000 | 1.000000 |
| visual_joint_normal | 5 | 1.000000 | 1.000000 |
| visual_joint_normal | 6 | 1.000000 | 1.000000 |
| visual_joint_normal | 7 | 1.000000 | 1.000000 |
| visual_joint_normal | 8 | 1.000000 | 1.000000 |
| visual_joint_normal | 9 | 1.000000 | 1.000000 |
| visual_joint_shuffled_vision | 2 | 1.000000 | 1.000000 |
| visual_joint_shuffled_vision | 3 | 1.000000 | 0.960000 |
| visual_joint_shuffled_vision | 4 | 1.000000 | 0.960000 |
| visual_joint_shuffled_vision | 5 | 1.000000 | 1.000000 |
| visual_joint_shuffled_vision | 6 | 1.000000 | 0.960000 |
| visual_joint_shuffled_vision | 7 | 1.000000 | 0.880000 |
| visual_joint_shuffled_vision | 8 | 1.000000 | 0.800000 |
| visual_joint_shuffled_vision | 9 | 1.000000 | 0.760000 |
| visual_joint_zero_vision | 2 | 0.560000 | 0.240000 |
| visual_joint_zero_vision | 3 | 0.360000 | 0.040000 |
| visual_joint_zero_vision | 4 | 0.120000 | 0.000000 |
| visual_joint_zero_vision | 5 | 0.160000 | 0.000000 |
| visual_joint_zero_vision | 6 | 0.160000 | 0.000000 |
| visual_joint_zero_vision | 7 | 0.160000 | 0.000000 |
| visual_joint_zero_vision | 8 | 0.040000 | 0.000000 |
| visual_joint_zero_vision | 9 | 0.000000 | 0.000000 |

## Autoregressive Diagnostic

Mode name: `open_loop_joint_feedback_with_exogenous_images`.
- `aggregate_ar_token_nearest_block_acc`: `{"groups": {"joint_only_normal": {"mean": 0.22272727272727275, "mode": "normal", "model_type": "joint_only", "sd": 0.02874797872880344, "values": [0.23636363636363636, 0.17727272727272728, 0.2545454545454545, 0.22727272727272727, 0.21818181818181817]}, "visual_joint_normal": {"mean": 0.3009090909090909, "mode": "normal", "model_type": "visual_joint", "sd": 0.06423420473432546, "values": [0.2545454545454545, 0.2590909090909091, 0.32272727272727275, 0.2636363636363636, 0.40454545454545454]}, "visual_joint_shuffled_vision": {"mean": 0.16727272727272727, "mode": "shuffled_vision", "model_type": "visual_joint", "sd": 0.021656598635584726, "values": [0.19545454545454546, 0.17272727272727273, 0.1590909090909091, 0.13636363636363635, 0.17272727272727273]}, "visual_joint_zero_vision": {"mean": 0.10181818181818181, "mode": "zero_vision", "model_type": "visual_joint", "sd": 0.010464422212019394, "values": [0.09090909090909091, 0.10454545454545454, 0.09545454545454546, 0.11818181818181818, 0.1]}}, "persistence": 0.10454545454545454}`
- `aggregate_ar_token_physical_hit_acc`: `{"groups": {"joint_only_normal": {"mean": 0.14090909090909093, "mode": "normal", "model_type": "joint_only", "sd": 0.0376203303573792, "values": [0.16818181818181818, 0.09545454545454546, 0.10454545454545454, 0.17272727272727273, 0.16363636363636364]}, "visual_joint_normal": {"mean": 0.17363636363636362, "mode": "normal", "model_type": "visual_joint", "sd": 0.05886319761156244, "values": [0.1409090909090909, 0.10454545454545454, 0.20909090909090908, 0.1590909090909091, 0.2545454545454545]}, "visual_joint_shuffled_vision": {"mean": 0.06454545454545454, "mode": "shuffled_vision", "model_type": "visual_joint", "sd": 0.014156737729452215, "values": [0.07727272727272727, 0.05, 0.05909090909090909, 0.05454545454545454, 0.08181818181818182]}, "visual_joint_zero_vision": {"mean": 0.02363636363636364, "mode": "zero_vision", "model_type": "visual_joint", "sd": 0.038354587472234525, "values": [0.09090909090909091, 0.0, 0.00909090909090909, 0.0, 0.01818181818181818]}}, "persistence": 0.0}`

### Autoregressive Endpoint Error And Token Accuracy

| Group | Endpoint EE err cm | Endpoint XY err cm | Endpoint Z err cm | ar_token_nearest_block_acc | ar_token_physical_hit_acc | Hit 0/5/10mm |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| joint_only_normal | 9.9578 | 9.9559 | 0.1055 | 0.222727 | 0.140909 | 0.126364/0.140909/0.172727 |
| persistence | 13.0336 | 10.6134 | 7.1259 | 0.104545 | 0.000000 | 0.000000/0.000000/0.000000 |
| visual_joint_normal | 8.7689 | 8.7661 | 0.1215 | 0.300909 | 0.173636 | 0.136364/0.173636/0.194545 |
| visual_joint_shuffled_vision | 10.8935 | 10.8905 | 0.1550 | 0.167273 | 0.064545 | 0.048182/0.064545/0.084545 |
| visual_joint_zero_vision | 10.6080 | 10.5915 | 0.3322 | 0.101818 | 0.023636 | 0.015455/0.023636/0.027273 |

### Autoregressive Token Accuracy By Length

| Group | Length | ar_token_nearest_block_acc | ar_token_physical_hit_acc |
| --- | ---: | ---: | ---: |
| joint_only_normal | 2 | 0.380000 | 0.300000 |
| joint_only_normal | 3 | 0.333333 | 0.266667 |
| joint_only_normal | 4 | 0.340000 | 0.280000 |
| joint_only_normal | 5 | 0.312000 | 0.216000 |
| joint_only_normal | 6 | 0.233333 | 0.126667 |
| joint_only_normal | 7 | 0.165714 | 0.080000 |
| joint_only_normal | 8 | 0.150000 | 0.060000 |
| joint_only_normal | 9 | 0.151111 | 0.088889 |
| persistence | 2 | 0.100000 | 0.000000 |
| persistence | 3 | 0.066667 | 0.000000 |
| persistence | 4 | 0.200000 | 0.000000 |
| persistence | 5 | 0.080000 | 0.000000 |
| persistence | 6 | 0.100000 | 0.000000 |
| persistence | 7 | 0.057143 | 0.000000 |
| persistence | 8 | 0.125000 | 0.000000 |
| persistence | 9 | 0.111111 | 0.000000 |
| visual_joint_normal | 2 | 0.520000 | 0.360000 |
| visual_joint_normal | 3 | 0.493333 | 0.346667 |
| visual_joint_normal | 4 | 0.390000 | 0.260000 |
| visual_joint_normal | 5 | 0.336000 | 0.224000 |
| visual_joint_normal | 6 | 0.313333 | 0.173333 |
| visual_joint_normal | 7 | 0.228571 | 0.131429 |
| visual_joint_normal | 8 | 0.250000 | 0.110000 |
| visual_joint_normal | 9 | 0.222222 | 0.097778 |
| visual_joint_shuffled_vision | 2 | 0.260000 | 0.100000 |
| visual_joint_shuffled_vision | 3 | 0.240000 | 0.146667 |
| visual_joint_shuffled_vision | 4 | 0.170000 | 0.050000 |
| visual_joint_shuffled_vision | 5 | 0.176000 | 0.080000 |
| visual_joint_shuffled_vision | 6 | 0.180000 | 0.053333 |
| visual_joint_shuffled_vision | 7 | 0.188571 | 0.057143 |
| visual_joint_shuffled_vision | 8 | 0.110000 | 0.045000 |
| visual_joint_shuffled_vision | 9 | 0.142222 | 0.057778 |
| visual_joint_zero_vision | 2 | 0.120000 | 0.020000 |
| visual_joint_zero_vision | 3 | 0.040000 | 0.013333 |
| visual_joint_zero_vision | 4 | 0.080000 | 0.020000 |
| visual_joint_zero_vision | 5 | 0.112000 | 0.024000 |
| visual_joint_zero_vision | 6 | 0.073333 | 0.026667 |
| visual_joint_zero_vision | 7 | 0.120000 | 0.017143 |
| visual_joint_zero_vision | 8 | 0.120000 | 0.030000 |
| visual_joint_zero_vision | 9 | 0.111111 | 0.026667 |

### Autoregressive Full-Trial Accuracy By Length

| Group | Length | ar_full_nearest_block_acc | ar_full_physical_hit_acc |
| --- | ---: | ---: | ---: |
| joint_only_normal | 2 | 0.040000 | 0.040000 |
| joint_only_normal | 3 | 0.000000 | 0.000000 |
| joint_only_normal | 4 | 0.000000 | 0.000000 |
| joint_only_normal | 5 | 0.000000 | 0.000000 |
| joint_only_normal | 6 | 0.000000 | 0.000000 |
| joint_only_normal | 7 | 0.000000 | 0.000000 |
| joint_only_normal | 8 | 0.000000 | 0.000000 |
| joint_only_normal | 9 | 0.000000 | 0.000000 |
| persistence | 2 | 0.000000 | 0.000000 |
| persistence | 3 | 0.000000 | 0.000000 |
| persistence | 4 | 0.000000 | 0.000000 |
| persistence | 5 | 0.000000 | 0.000000 |
| persistence | 6 | 0.000000 | 0.000000 |
| persistence | 7 | 0.000000 | 0.000000 |
| persistence | 8 | 0.000000 | 0.000000 |
| persistence | 9 | 0.000000 | 0.000000 |
| visual_joint_normal | 2 | 0.240000 | 0.080000 |
| visual_joint_normal | 3 | 0.160000 | 0.080000 |
| visual_joint_normal | 4 | 0.000000 | 0.000000 |
| visual_joint_normal | 5 | 0.000000 | 0.000000 |
| visual_joint_normal | 6 | 0.000000 | 0.000000 |
| visual_joint_normal | 7 | 0.000000 | 0.000000 |
| visual_joint_normal | 8 | 0.000000 | 0.000000 |
| visual_joint_normal | 9 | 0.000000 | 0.000000 |
| visual_joint_shuffled_vision | 2 | 0.000000 | 0.000000 |
| visual_joint_shuffled_vision | 3 | 0.000000 | 0.000000 |
| visual_joint_shuffled_vision | 4 | 0.000000 | 0.000000 |
| visual_joint_shuffled_vision | 5 | 0.000000 | 0.000000 |
| visual_joint_shuffled_vision | 6 | 0.000000 | 0.000000 |
| visual_joint_shuffled_vision | 7 | 0.000000 | 0.000000 |
| visual_joint_shuffled_vision | 8 | 0.000000 | 0.000000 |
| visual_joint_shuffled_vision | 9 | 0.000000 | 0.000000 |
| visual_joint_zero_vision | 2 | 0.000000 | 0.000000 |
| visual_joint_zero_vision | 3 | 0.000000 | 0.000000 |
| visual_joint_zero_vision | 4 | 0.000000 | 0.000000 |
| visual_joint_zero_vision | 5 | 0.000000 | 0.000000 |
| visual_joint_zero_vision | 6 | 0.000000 | 0.000000 |
| visual_joint_zero_vision | 7 | 0.000000 | 0.000000 |
| visual_joint_zero_vision | 8 | 0.000000 | 0.000000 |
| visual_joint_zero_vision | 9 | 0.000000 | 0.000000 |

### Autoregressive Joint-Limit And Growth Diagnostics

| Group | Joint-limit violation rate | Evaluated steps | RMSE growth timestep entries | RMSE growth rank entries |
| --- | ---: | ---: | ---: | ---: |
| joint_only_normal | 0.000000 | 13000 | 107 | 9 |
| persistence | 0.000000 | 2600 | 107 | 9 |
| visual_joint_normal | 0.000000 | 13000 | 107 | 9 |
| visual_joint_shuffled_vision | 0.000000 | 13000 | 107 | 9 |
| visual_joint_zero_vision | 0.000000 | 13000 | 107 | 9 |

## FK Validation

FK validation passed: `True`; median error 0.042 mm, max error 0.131 mm.

## Failed Seeds Or Unavailable Histories

- no accuracy seed failures recorded
- continuation stopped reasons:
  - joint_only_seed0_extended_continuation_from_best: stopped at epoch 864 because `non_finite_gradient_joint_encoder.net.0.weight_epoch_865`

## Commands And Runtime

Primary resumable command:

```bash
conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.posthoc_suite --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json --resume --devices auto
```

Continuation runtime seconds recorded in lineage: 0.0

## Output Paths

- convergence: `corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/convergence`
- accuracy: `corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/accuracy`
- fk: `corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/fk`
- final visual states: `corsi_artifacts/motion_baseline/posthoc_convergence_accuracy_v1/states`
- audit report: `reports/corsi_motion_convergence_audit.md`
- final report: `reports/corsi_motion_convergence_accuracy.md`

## Finish Checklist

- files changed: see git status and this report's output paths
- tests passed: recorded in final assistant response after test execution
- runs extended: see `continuation_lineage.json`
- final convergence decisions: see `convergence_per_run.json`
- final checkpoints: see `final_checkpoint_selection.json`
- exact resume command: listed above
