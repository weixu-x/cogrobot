# Corsi 7-Joint Motion Baseline Final Report

Date: 2026-06-22

## Executive Summary

The baseline has been revised from the rejected 6-joint target to a 7-joint
Panda arm prediction target:

`q_t = [robot0_joint1, robot0_joint2, robot0_joint3, robot0_joint4, robot0_joint5, robot0_joint6, robot0_joint7]`

The final model is a causal one-step motion-prediction baseline. At timestep
`t`, it consumes `image_t` and `q_t`, keeps recurrent state across the padded
sequence, and predicts normalized `q_{t+1}`. Variable-length sequences are
handled by dynamic padding within each batch plus `valid_mask` and `loss_mask`;
no recurrent reset is applied at Corsi segment boundaries.

This model is a causal visual-proprioceptive motion-prediction baseline.
Because the current dataset contains one continuous motion trajectory without a
separate presentation, retention, and recall phase, its hidden-state geometry
cannot by itself be interpreted as evidence of Corsi working-memory slots or
memory maintenance.

## Environment Gate

The code used the current project checkout, not an external Corsi install:

- `corsi`: `/home/wei2025/Developer/cogrobot/corsi/__init__.py`
- `robosuite`: `/home/wei2025/Developer/cogrobot/robosuite/__init__.py`
- robot: `PandaDexRH`
- controller: `OperationalSpaceController`
- Corsi scene smoke: 9 `corsi_block_*_geom` geoms, all RGBA `[0.0, 0.0, 0.0, 1.0]`

CUDA is available outside the filesystem sandbox. Training and GPU evaluation
were run through escalated `conda run -n robosuite ... --device cuda:*`
commands because sandboxed PyTorch could not initialize CUDA/NVML.

## Canonical Dataset

Raw source:

`corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50`

Canonical output:

`corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12`

Canonical facts:

| Field | Value |
| --- | --- |
| samples | 400 |
| lengths | 2-9, 50 samples each |
| split | train 320, val 40, test 40 |
| split by length | train 40 / val 5 / test 5 for every length |
| image shape | `[3, 128, 128]`, uint8 |
| joint dim | 7 |
| samples per block segment | `K=12` |
| canonical sequence length | `T = Corsi length * 12` |
| canonical fingerprint | `116565bf565b42de0ebeabc97b484d153c08d2c4d17fd0560144887220a5dbf8` |
| raw manifest sha256 | `31be68b089f4e131a6a29ebd239c4a5bb62ebd4def5a2c06c08ff4db4b8673cd` |

Canonicalization samples each raw block-motion segment at 12 normalized times.
Images are selected by nearest raw timestamp and joints are linearly
interpolated by timestamp. Train-only image and joint normalization statistics
are stored in the manifest.

## Model

Two learned baselines were trained:

| Model | Inputs | Parameters |
| --- | --- | ---: |
| visual_joint | RGB image + 7-joint vector | 342,919 |
| joint_only | 7-joint vector only | 138,855 |

The visual-joint model uses a small convolutional visual encoder, an MLP joint
encoder, fusion MLP, and an instrumented LSTM cell exposing `h_t`, `c_t`,
input/forget/candidate/output gates. The joint-only model uses the same
recurrent prediction head without image input. Persistence uses `q_t` as the
prediction for `q_{t+1}`.

Training config:

- config: `corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json`
- seeds: `0, 5, 10, 15, 20`
- optimizer: AdamW, LR `1e-3`, weight decay `1e-5`
- batch size: 16
- max epochs: 300
- min epochs: 100
- patience: 50
- AMP: enabled on CUDA

## Test Results

Metrics are on the held-out test split. RMSE/MSE are in normalized joint space.
MAE is the mean physical joint absolute error in degrees.

| Model / mode | Test RMSE mean | SD | Joint MAE deg mean | SD |
| --- | ---: | ---: | ---: | ---: |
| visual_joint normal | 0.041966 | 0.002249 | 0.160088 | 0.006927 |
| visual_joint shuffled vision | 0.072485 | 0.005647 | 0.283274 | 0.026038 |
| visual_joint zero vision | 0.292132 | 0.024258 | 1.467600 | 0.238032 |
| joint_only normal | 0.032574 | 0.000747 | 0.123552 | 0.004729 |
| persistence | 0.173469 | 0.000000 | 0.755587 | 0.000000 |

Per-seed visual_joint normal:

| Seed | Epochs | Best val RMSE | Test RMSE | Test MAE deg | Runtime sec |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 240 | 0.041656 | 0.039603 | 0.157343 | 3586.0 |
| 5 | 300 | 0.043191 | 0.039832 | 0.149655 | 2681.2 |
| 10 | 188 | 0.044323 | 0.043035 | 0.167846 | 2692.6 |
| 15 | 226 | 0.044294 | 0.042442 | 0.163384 | 3242.5 |
| 20 | 220 | 0.046827 | 0.044919 | 0.162214 | 3402.3 |

Per-seed joint_only normal:

| Seed | Epochs | Best val RMSE | Test RMSE | Test MAE deg | Runtime sec |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 300 | 0.034439 | 0.032330 | 0.122021 | 1050.6 |
| 5 | 300 | 0.034897 | 0.033636 | 0.129756 | 1044.6 |
| 10 | 300 | 0.034203 | 0.032824 | 0.122529 | 1050.0 |
| 15 | 300 | 0.033231 | 0.031586 | 0.117204 | 1050.0 |
| 20 | 300 | 0.032953 | 0.032496 | 0.126250 | 1047.1 |

Interpretation:

- Visual information is being used: shuffled and zeroed vision degrade the
  visual_joint model strongly.
- Joint-only still outperforms visual_joint for this one-step target. The
  current task is dominated by smooth proprioceptive dynamics, so adding images
  does not improve next-joint prediction under this dataset.
- Persistence is much worse than both learned baselines, so the models are not
  just copying `q_t`.

## Length, Rank, And Boundary Diagnostics

Mean test RMSE by Corsi sequence length:

| Length | visual_joint | joint_only |
| ---: | ---: | ---: |
| 2 | 0.021332 | 0.020737 |
| 3 | 0.027095 | 0.025921 |
| 4 | 0.032670 | 0.028732 |
| 5 | 0.032896 | 0.026215 |
| 6 | 0.039249 | 0.030504 |
| 7 | 0.041247 | 0.029957 |
| 8 | 0.049118 | 0.034143 |
| 9 | 0.051600 | 0.041865 |

Mean test RMSE by block rank:

| Rank | visual_joint | joint_only |
| ---: | ---: | ---: |
| 0 | 0.019870 | 0.019839 |
| 1 | 0.024872 | 0.025829 |
| 2 | 0.038588 | 0.030566 |
| 3 | 0.049714 | 0.033917 |
| 4 | 0.051523 | 0.036598 |
| 5 | 0.054243 | 0.036763 |
| 6 | 0.049160 | 0.039166 |
| 7 | 0.059093 | 0.046509 |
| 8 | 0.072211 | 0.063617 |

Boundary transitions are harder than within-segment transitions:

| Model | Within-segment RMSE | Boundary RMSE |
| --- | ---: | ---: |
| visual_joint | 0.039938 | 0.063136 |
| joint_only | 0.029714 | 0.058771 |

Mean per-joint MAE in degrees, joint order `robot0_joint1..7`:

| Model | j1 | j2 | j3 | j4 | j5 | j6 | j7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visual_joint | 0.114540 | 0.160362 | 0.067036 | 0.296211 | 0.121540 | 0.122350 | 0.238580 |
| joint_only | 0.100458 | 0.117859 | 0.049112 | 0.213799 | 0.100505 | 0.105847 | 0.177284 |

## State Export

State traces were exported for all best visual_joint checkpoints:

`corsi_artifacts/motion_baseline/states/`

Files:

- 5 seeds x 3 splits = 15 HDF5 files
- each HDF5 has matching `.json` metadata
- train rows per seed: 21,120
- val rows per seed: 2,640
- test rows per seed: 2,640
- total state directory size: about 565M

Exported fields include:

`seq_id`, `split`, `seed`, `length`, `timestep`, `source_frame_index`,
`source_timestamp`, `rank`, `block_id`, `block_xy`, `segment_progress`,
`boundary`, normalized and physical joints, normalized and physical
predictions/targets, `visual_feature`, `joint_feature`, `fused_feature`,
`h_t`, `c_t`, `input_gate`, `forget_gate`, `candidate`, and `output_gate`.

## Outputs

Main artifacts:

- audit report: `reports/corsi_motion_baseline_audit.md`
- final report: `reports/corsi_motion_baseline_final.md`
- canonical manifest: `corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12/manifest.json`
- aggregate metrics: `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/aggregate_metrics.json`
- learned checkpoints: `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/*_full/best.pt`
- state traces: `corsi_artifacts/motion_baseline/states/visual_joint_seed*_full_*.h5`

Artifact sizes:

| Path | Size |
| --- | ---: |
| canonical dataset | 451M |
| run/checkpoint/metrics directory | 95M |
| state traces | 565M |

## Commands Run

Canonicalization:

```bash
conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.canonicalize --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json --overwrite
```

Formal training examples:

```bash
conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.run_suite --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json --model-type joint_only --device cuda:1 --seeds 0 5 10 15 20 --run-suffix _full
conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.train --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json --model-type visual_joint --seed 20 --run-name visual_joint_seed20_full --device cuda:0
```

Evaluation example:

```bash
conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.evaluate --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json --model-type visual_joint --checkpoint corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed20_full/best.pt --split test --mode normal --batch-size 16 --device cuda:0 --output corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed20_full/test_normal.json
```

State export example:

```bash
conda run -n robosuite python -m corsi.experiments.corsi_motion_baseline.extract_states --config corsi/experiments/corsi_motion_baseline/configs/baseline_7joint_k12.json --checkpoint corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12/visual_joint_seed0_full/best.pt --split test --output corsi_artifacts/motion_baseline/states/visual_joint_seed0_full_test.h5 --batch-size 16 --device cpu
```

Validation:

```bash
conda run -n robosuite python -m pytest tests/test_corsi_motion_baseline.py tests/test_corsi_motion_posthoc.py tests/test_corsi_prediction_visualization.py -q
conda run -n robosuite python -m pytest tests/test_corsi_heatmaps.py tests/test_corsi_attention.py tests/test_corsi_motion_baseline.py -q
```

Results:

- `tests/test_corsi_motion_baseline.py tests/test_corsi_motion_posthoc.py tests/test_corsi_prediction_visualization.py`: 32 passed
- `tests/test_corsi_heatmaps.py tests/test_corsi_attention.py tests/test_corsi_motion_baseline.py`: 20 passed
- minimal `CorsiSceneDemo` smoke: passed with local project paths,
  `PandaDexRH`, `OperationalSpaceController`, 9 black block geoms

## Code Changes

Added:

- `corsi/experiments/corsi_motion_baseline/`
  - `__init__.py`
  - `canonicalize.py`
  - `dataset.py`
  - `model.py`
  - `train.py`
  - `evaluate.py`
  - `extract_states.py`
  - `run_suite.py`
  - `configs/baseline_7joint_k12.json`
- `tests/test_corsi_motion_baseline.py`
- `reports/corsi_motion_baseline_audit.md`
- `reports/corsi_motion_baseline_final.md`

The obsolete scratch experiment line has been removed from the current
worktree. The remaining uncommitted code is the 7-joint
baseline/posthoc/visualization work described above.

## Limitations

- The visual target block is not explicitly cued in the RGB frames; the model
  mostly predicts motion from ongoing proprioceptive dynamics.
- The task is one continuous executed trajectory, not a separated
  presentation-retention-recall Corsi protocol.
- Hidden states and gates are useful for exploratory geometry and dynamics
  analysis, but they should not be labeled as working-memory slots without a
  dataset that separates memory encoding and recall demands.
- Joint-only outperforming visual_joint means the current visual baseline is
  not an accuracy improvement over proprioception-only prediction for this
  one-step target, even though image ablations show the visual branch is used.
