# `freecam_index_v2_order_attention_ss`

## Intent

- Target the `length 4 -> 5` accuracy drop in the visual baseline.
- Keep task definition fixed:
  - input: `freecam` keyframe sequence
  - target: block index sequence
- Compare directly against `freecam_index_v2_20ep`.

## Added Functionality

- Added encoder-to-decoder attention over encoder time states.
- Added decoder step embedding.
- Added scheduled sampling to reduce autoregressive drift.
- Added length-specific order-drift metrics in validation summaries.

## Primary Evaluation Criteria

- Higher or matched overall `full_sequence_accuracy`
- Improved `length 5` / `length 6` full-sequence accuracy
- Lower `order_error` rate at `length 5 / 6`
- Later `mean_first_error_pos` at `length 5 / 6`
- Better serial-position accuracy from position `3+`

## Status

- Implementation complete
- Full 80-epoch comparison run: pending
