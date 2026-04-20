# Coordinate Baseline: Mixed Span 2-6

## Summary

This experiment is the first formal baseline for the Corsi coordinate-only system.

Task definition:
- input features: `(x, y, dx, dy)`
- target: full block sequence
- model: encoder-decoder LSTM
- span range: 2-6

Result:
- the model reached perfect validation performance under the current synthetic setup

## Run Metadata

- Date: 2026-03-24
- Config: `corsi/experiments/coordinate_base/configs/coord_baseline_mixed_2_6.json`
- Output dir: `corsi_artifacts/coordinate_base/coord_baseline_mixed_2_6/`
- Requested device: `auto`
- Resolved device: `mps`

## Best Result

- Best epoch: `20`
- Best validation loss: `0.0008479162338069467`
- Best token accuracy: `1.0`
- Best full-sequence accuracy: `1.0`
- Estimated span: `6`

## Best Accuracy By Length

- Length 2: `1.0`
- Length 3: `1.0`
- Length 4: `1.0`
- Length 5: `1.0`
- Length 6: `1.0`

## Learning Curve Notes

Early training already improved quickly:
- Epoch 1: full-sequence accuracy `0.6175`
- Epoch 2: full-sequence accuracy `0.8745`
- Epoch 3: full-sequence accuracy `0.941`

Strong performance appeared very early:
- Epoch 9: full-sequence accuracy `0.996`
- Epoch 13: full-sequence accuracy `0.999`
- Epoch 20: full-sequence accuracy `1.0`

The hardest length improved steadily:
- Length 6 accuracy at epoch 1: `0.08056872037914692`
- Length 6 accuracy at epoch 6: `0.8886255924170616`
- Length 6 accuracy at epoch 13: `0.995260663507109`
- Length 6 accuracy at epoch 20: `1.0`

## Interpretation

What this means:
- the minimal coordinate representation is sufficient for the model to solve the current span 2-6 task
- the sequence model and training pipeline are functioning correctly
- the current synthetic setup is likely too easy to be a long-term benchmark on its own

Why this matters:
- we now have a clean baseline before adding delay / maintenance
- we can compare future visual, embodied, and aging versions against a known working reference

## Next Step

The most important next experiment is not a larger model.

The next step should be:
- add `delay / maintenance` conditions

Reason:
- immediate recall is now close to saturated
- delay will test retention rather than only direct replay
- delay also gives a principled place to add aging effects later

## Files

- Summary: `corsi_artifacts/coordinate_base/coord_baseline_mixed_2_6/final_summary.json`
- Device info: `corsi_artifacts/coordinate_base/coord_baseline_mixed_2_6/device_info.json`
- Epoch logs: `corsi_artifacts/coordinate_base/coord_baseline_mixed_2_6/epoch_logs.jsonl`
- Best model: `corsi_artifacts/coordinate_base/coord_baseline_mixed_2_6/best_model.pt`
