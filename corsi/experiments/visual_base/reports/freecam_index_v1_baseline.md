# Freecam Index V1 Baseline

## Summary

This is the first official robosuite visual baseline built on the reorganized `visual_base` track.

Task definition:

- input: robosuite `freecam` keyframe sequence
- target: full block-index sequence
- model: per-frame CNN + encoder-decoder LSTM
- train split: `freecam_index_v1_train`
- val split: `freecam_index_v1_val`

This run is an initial scale-up baseline, not yet a high-data benchmark.

## Dataset

- Train root: `corsi_artifacts/visual_base/datasets/freecam_index_v1_train`
- Val root: `corsi_artifacts/visual_base/datasets/freecam_index_v1_val`
- Camera: `freecam`
- Train samples: `16`
- Val samples: `4`
- Sequence length range: `2-6`

Export parameters:

- `control_freq=20`
- `fps=20`
- `speed_gain=0.12`
- `dwell_steps=4`
- `arrival_threshold=0.02`
- `target_height=0.04`

## Training Config

- Config: `corsi/experiments/visual_base/configs/freecam_index_v1_baseline.json`
- Output dir: `corsi_artifacts/visual_base/training/freecam_index_v1_baseline`
- Device: `mps`
- Epochs: `8`
- Batch size: `8`

## Result

- Best epoch: `1`
- Best validation loss: `2.2049858570098877`
- Best token accuracy: `0.0`
- Best full-sequence accuracy: `0.0`
- Estimated span: `0`

Length-wise accuracy:

- Length 3: `0.0`
- Length 4: `0.0`
- Length 5: `0.0`

Error pattern at the best epoch:

- `wrong_block`: `1.0`

## Interpretation

What this run established:

- the train / val dataset split format works end to end
- `train_visual.py` can train directly from fixed dataset roots
- the baseline output directory now saves config, dataset info, logs, checkpoint, and summary in one place

What this run did not yet establish:

- useful visual recall performance
- a stable visual benchmark comparable to the coordinate baseline

The current result is expected to be weak because the dataset is still a small initial scale-up run.

## Next Step

The next priority stays the same:

- enlarge the `freecam` baseline dataset further
- retrain the same model family before adding delay / heatmap / timing variants

Do not branch into delay, heatmap, or event timing until the non-delay visual baseline is stronger than this initial run.
