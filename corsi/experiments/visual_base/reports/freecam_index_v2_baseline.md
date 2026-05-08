# Freecam Index V2 Baseline

## Summary

This is the first non-trivial robosuite visual baseline on the scaled `freecam` dataset.

Task definition:

- input: robosuite `freecam` keyframe sequence
- target: full block-index sequence
- model: per-frame CNN + encoder-decoder LSTM
- train split: `freecam_index_v2_train`
- val split: `freecam_index_v2_val`

Compared with `v1`, this run moves from a tiny preview-style split to a real `900 / 100` dataset.

## Dataset

- Train root: `corsi_artifacts/visual_base/datasets/freecam_index_v2_train`
- Val root: `corsi_artifacts/visual_base/datasets/freecam_index_v2_val`
- Camera: `freecam`
- Train samples: `900`
- Val samples: `100`
- Sequence length range: `2-6`

Export parameters:

- `control_freq=20`
- `fps=20`
- `speed_gain=0.12`
- `dwell_steps=4`
- `arrival_threshold=0.02`
- `target_height=0.04`
- `write_rollout_videos=false`
- `keep_rollout_frames=false`

Important scale-up change:

- dataset export now supports sharded generation
- visual export can skip rollout video writing entirely
- the merged train / val roots are manifest-driven datasets built from shard manifests

## Training Config

- Config: `corsi/experiments/visual_base/configs/freecam_index_v2_baseline.json`
- Output dir: `corsi_artifacts/visual_base/training/freecam_index_v2_baseline`
- Device: `mps`
- Epochs: `12`
- Batch size: `16`

Model family:

- frame encoder: CNN
- temporal model: encoder-decoder LSTM
- output head: index sequence logits over the 9 Corsi blocks

## Variable-Length Learning

This experiment mixes sequence lengths `2-6` in the same train and validation splits.

How this works in the current code:

- each sample stores only the active keyframe sequence for that trial
- batch collation pads sequences to the longest sequence inside the batch
- `targets_pad` uses a pad value ignored by cross-entropy
- `mask` marks which time steps are real and which are padding
- the LSTM still receives the padded batch, but loss and metrics only count active positions

Practically, this means length `2`, `3`, `4`, `5`, and `6` trials can be trained together without pretending they are the same temporal length.

Relevant files:

- `corsi/data/collate_visual.py`
- `corsi/training/train_visual.py`
- `corsi/analysis/metrics.py`

## Bug Fix During V2 Run

The first attempt to train on `v2` exposed a real evaluation bug.

Problem:

- `evaluate_model()` collected greedy-decoded predictions from different validation batches
- those batches could have different padded lengths, because the longest sequence in each batch was not always the same
- the code then tried to `torch.cat()` those tensors directly
- this failed once the larger validation set produced batches with different maximum sequence lengths

Observed failure:

- `RuntimeError: Sizes of tensors must match except in dimension 0`

Fix:

- before concatenation, predictions / targets / masks are now padded to the global maximum step count seen across all validation batches
- metrics are then computed on the padded tensors together with the true `mask`

Impact:

- the bug did not change the training objective itself
- it blocked validation on realistic variable-length data
- after the fix, the `v2` run completed and the metrics below are trustworthy

Relevant file:

- `corsi/training/train_visual.py`

## Result

- Best epoch: `12`
- Best validation loss: `1.2156801896022105`
- Best token accuracy: `0.5089514066496164`
- Best full-sequence accuracy: `0.2199999988079071`
- Estimated span: `2`

Length-wise accuracy at the best epoch:

- Length 2: `0.7222222222222222`
- Length 3: `0.25925925925925924`
- Length 4: `0.10526315789473684`
- Length 5: `0.0`
- Length 6: `0.0`

Error pattern at the best epoch:

- `correct`: `0.22`
- `order_error`: `0.05`
- `wrong_block`: `0.73`

## Interpretation

What `v2` establishes:

- the visual pipeline is now learning real signal rather than only running end to end
- scaled robosuite `freecam` keyframe data is enough to get useful improvement over the tiny `v1` split
- short sequences are being learned substantially better than long ones

What `v2` does not yet establish:

- strong recall on longer sequences (`5-6`)
- a visual span comparable to the coordinate baseline
- robustness under delay / blank intervals

The main qualitative pattern is:

- the model is now clearly competent on short sequences
- performance collapses as sequence length grows
- the next useful baseline comparison is still within the same non-delay visual setup, not yet heatmap or timing variants

## Next Step

The most interpretable next move is:

- keep the same `freecam + keyframe + index` task
- either train longer or lightly tune the baseline
- compare directly against this `v2` run before branching into delay / heatmap / timing

Only after that comparison should `corsi-visual-delay`, `corsi-visual-heatmap`, and `corsi-visual-timing` become the main focus.
