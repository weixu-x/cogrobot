# Visual Base Version Notes

This file is a short naming guide for the visual baseline experiments under `visual_base/`.

## Dataset Versions

### `freecam_index_v1`

- First official `freecam` visual baseline dataset.
- Task format:
  - input: `freecam` keyframe sequence
  - target: block index sequence
- Main change:
  - introduced fixed train / val dataset roots for the reorganized visual pipeline
- Scale:
  - train: `16`
  - val: `4`

### `freecam_index_v2`

- Scaled `freecam` visual baseline dataset.
- Main changes:
  - sharded export
  - keyframe-only export path
  - merged manifest-driven train / val roots
- Scale:
  - train: `900`
  - val: `100`

## Training Runs

### `freecam_index_v1_baseline`

- First official end-to-end visual baseline run on `v1`.
- Main change:
  - proved the reorganized visual training pipeline runs end to end

### `freecam_index_v2_baseline`

- First scaled baseline run on `v2`.
- Main changes:
  - moved to the larger `900 / 100` dataset
  - exposed and fixed the variable-length evaluation aggregation bug in validation

### `freecam_index_v2_20ep`

- Main long-running baseline on `v2`.
- Main changes:
  - switched to the new checkpoint-capable training loop
  - supports scratch training, full resume, and init-model loading
  - used for continued training beyond the first 20 epochs

### `freecam_index_v2_longer`

- Temporary longer-epoch comparison run.
- Main change:
  - tested training longer before full resume support was used as the primary workflow
- Note:
  - this run was superseded by continuing `freecam_index_v2_20ep` with checkpoint resume

### `freecam_index_v2_order_attention_ss`

- Order-drift-targeted visual baseline extension on `freecam_index_v2`.
- Main changes:
  - added attention-based decoder access to encoder time states
  - added decoder step embedding
  - added scheduled sampling
- Goal:
  - reduce long-sequence order drift without changing the task, camera, or target type
