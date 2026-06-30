# Raw Data and Model Inspection - Training Pause 2026-06-30

## Pause State

Training is paused. No CUDA training process or `corsi_memory_recall_v2.train` process was running at the time of this report.

Checkpoint snapshot:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12_expanded800_20260630/pause_snapshot_stage2_direct_interrupt_20260630`

Related audit report:

`reports/stage2_direct_training_audit_20260630.md`

## Raw Dataset

Raw manifest:

`corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_expanded800_20260630/manifest.json`

Raw dataset size: `14G`

Raw manifest schema: `scala_corsi_motion_raw_v1`

Raw episode count: `5562 / 5562 expected`

Failed/skipped episodes: none

Length counts:

| Length | Count |
| ---: | ---: |
| 2 | 102 |
| 3 | 380 |
| 4 | 680 |
| 5 | 880 |
| 6 | 880 |
| 7 | 880 |
| 8 | 880 |
| 9 | 880 |

Split-by-length counts from raw manifest and raw samples match the requested plan:

| Split | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 72 | 300 | 600 | 800 | 800 | 800 | 800 | 800 | 4972 |
| val | 10 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 220 |
| test | 20 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 370 |

Phase 0 self-check report:

`reports/dataset_manifest.json`

Self-check status: `ok`

Important self-check facts:

| Check | Result |
| --- | --- |
| Exact split x length counts | pass |
| L2 train complete ordered no-repeat pairs | `72 / 72` |
| L2 train missing count | `0` |
| Internal repeated block sequences | `0` |
| Cross-split overlaps for lengths 4-9 | `0` for train/val, train/test, val/test |
| L2 overlaps | train/val `10`, train/test `20`, val/test `3` |
| L3 overlaps | `0` for all split pairs |
| `baseline_dataset_design_record.md` present | no |

The L2 overlap is expected from the current generation design because train L2 uses all 72 ordered no-repeat pairs.

## Raw Episode Example

Example raw episode:

`corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_expanded800_20260630/episodes/train_len04_trial0000`

Metadata:

| Field | Value |
| --- | --- |
| split | train |
| length | 4 |
| block_order | `[0, 1, 3, 4]` |
| frame_count | 103 |
| segments | 4 |

Raw `arrays.npz` keys and shapes:

| Key | Shape | Dtype |
| --- | --- | --- |
| rgb | `[103, 128, 128, 3]` | `uint8` |
| joint | `[103, 7]` | `float32` |
| joint_velocity | `[103, 7]` | `float32` |
| ee_pose | `[103, 7]` | `float32` |
| ee_xy | `[103, 2]` | `float32` |
| action | `[103, 12]` | `float32` |
| qpos | `[103, 19]` | `float32` |
| qvel | `[103, 19]` | `float32` |
| block_id | `[103]` | `int64` |
| rank | `[103]` | `int64` |
| timestamp | `[103]` | `float64` |

First segment:

```json
{"segment_id": 0, "rank": 0, "block_id": 0, "start_frame": 0, "end_frame": 23}
```

## Canonical Dataset

Canonical manifest:

`corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12_expanded800_20260630/manifest.json`

Canonical dataset size: `5.8G`

Canonical manifest schema: `scala_corsi_memory_recall_v2_canonical_v1`

Canonical fingerprint:

`94d1813b3f7b4cb5fd2107e9bcdd306683c807dacd18b1429380de1f5575df78`

Canonical split counts:

| Split | Count |
| --- | ---: |
| train | 4972 |
| val | 220 |
| test | 370 |

Canonical split-by-length counts match the raw split-by-length counts exactly.

Canonical model inputs:

```json
["images", "segment_mask", "frame_mask"]
```

Canonical target fields:

```json
[
  "target_tokens",
  "target_xy",
  "block_xy_targets",
  "joint_targets",
  "ee_pose_targets",
  "ee_xy_targets",
  "ee_xy_norm_targets"
]
```

Canonical sample shape:

`["length", 12, 3, 128, 128]`

## Canonical Episode Example

Example canonical episode:

`corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12_expanded800_20260630/episodes/train_len04_trial0000.npz`

For raw `block_order = [0, 1, 3, 4]`, canonical `target_tokens = [0, 1, 3, 4, 9]`.

Canonical arrays:

| Key | Shape | Dtype |
| --- | --- | --- |
| images | `[4, 12, 3, 128, 128]` | `uint8` |
| frame_mask | `[4, 12]` | `bool` |
| segment_mask | `[4]` | `bool` |
| target_tokens | `[5]` | `int64` |
| target_token_mask | `[5]` | `bool` |
| block_id | `[4]` | `int64` |
| rank | `[4]` | `int64` |
| block_xy_targets | `[4, 2]` | `float32` |
| block_xyz_world_targets | `[4, 3]` | `float32` |
| joint_targets | `[4, 12, 7]` | `float32` |
| ee_pose_targets | `[4, 12, 7]` | `float32` |
| ee_xy_targets | `[4, 12, 2]` | `float32` |
| ee_xy_norm_targets | `[4, 12, 2]` | `float32` |
| source_frame_index | `[4, 12]` | `int64` |
| source_timestamp | `[4, 12]` | `float64` |
| segment_start_frame | `[4]` | `int64` |
| segment_end_frame | `[4]` | `int64` |
| segment_progress | `[4, 12]` | `float32` |

## Model Structure

Model file:

`corsi/experiments/corsi_memory_recall_v2/model.py`

Config defaults:

| Field | Default |
| --- | --- |
| image_channels | 3 |
| image_size | 128 |
| k_samples_per_segment | 12 |
| max_sequence_length | 9 |
| num_blocks | 9 |
| eos_token_id | 9 |
| cnn_dim | 64 |
| visual_hidden_dim | 64 |
| motor_hidden_dim | 64 |
| item_dim | 64 |
| memory_dim | 16 |
| memory_write_mode | `lstm` |
| memory_slot_dim | 16 |
| recall_readout_mode | `final` |
| recall_hidden_dim | 64 |
| recall_token_dim | 32 |
| joint_dim | 7 |

Expanded training configs override the relevant memory fields:

| Run | memory_dim | memory_slot_dim | memory_write_mode | recall_readout_mode |
| --- | ---: | ---: | --- | --- |
| dmem64_seed0 | 64 | 16 | `item_context_binding` | `final` |
| dmem16_seed0 | 16 | 16 | `item_context_binding` | `final` |

`build_model(config, stage=...)` accepts `stage` but deletes it before constructing the model. In this code path, `stage` is a no-op for topology.

An actual instantiation comparison showed Stage 1 and Stage 2 have identical state_dict keys and tensor shapes, with identical trainable parameter count:

| Check | Result |
| --- | --- |
| Stage 1-only keys | none |
| Stage 2-only keys | none |
| Shape differences | none |
| Stage 1 trainable params | 295504 |
| Stage 2 trainable params | 295504 |

## Stage 1 vs Stage 2 Training Semantics

Trainer file:

`corsi/experiments/corsi_memory_recall_v2/train.py`

Stage 1/2 difference is objective/call behavior, not model topology.

Stage 1:

Uses `compute_stage1_loss`, an auxiliary reconstruction objective over `joint`, `ee_pose`, and `ee_xy`.

Stage 2:

Uses `compute_stage2_loss`, which wraps autonomous recall loss plus configured auxiliary memory losses.

Stage 1 warm-start into Stage 2 is only used when `warm_start_stage1_checkpoint` is provided by CLI or config.

Warm-start prefixes:

```python
(
    "visual_encoder.",
    "visual_lstm.",
    "motor_lstm.",
    "joint_head.",
    "ee_pose_head.",
    "ee_xy_head.",
)
```

The paused runs did not provide `warm_start_stage1_checkpoint`, and their checkpoint metadata has no `extra.warm_start`.

Conclusion: the paused runs are direct Stage 2 training from random initialization, except D_mem64 later resumed from its own Stage 2 `latest.pt`.

## Paused Checkpoint Metadata

Snapshot size: `27M`

| Run | Stage | Latest epoch | Best full | Best token | Best val loss | Early-stop wait | Warm start |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dmem64_seed0 | 2 | 40 | 0.9954545455 | 0.9993333333 | 0.0169906131 | 4 / 30 | no |
| dmem16_seed0 | 2 | 31 | 0.7727272727 | 0.9600000000 | 0.3119274649 | 1 / 30 | no |

These runs were manually interrupted and did not stop by early stopping.

## Hold Point

Do not resume training, launch diagnostics, or start additional sweep cells until raw data and model structure have been inspected and a human decision is made.
