# Coordinate Model Inspection Guide

This note explains how to inspect a trained coordinate Corsi model and how to read the generated report.

## Purpose

The training metrics tell us whether the model works.

The inspection report tells us:
- what the model input actually looks like
- what the model structure is
- what the decoder sees at each step
- what the model predicts at each step
- how the full predicted sequence compares to the target sequence

This is useful for debugging, explaining, and later comparing delayed / visual / aged variants.

## Script

Use:

```bash
python corsi/analysis/inspect_coord_model.py \
  --checkpoint corsi_artifacts/coordinate_base/coord_baseline_mixed_2_6/best_model.pt \
  --num-samples 4
```

Optional arguments:
- `--device auto|cpu|mps|cuda`
- `--num-samples N`
- `--seed SEED`
- `--output-dir PATH`
- `--top-k K`

## Outputs

The script writes an inspection folder next to the checkpoint by default.

Files:
- `inspection_report.md`
- `inspection_details.json`
- `inspection_summary.json`
- `board_visuals/*.png`
- `board_visuals/*.svg`

## What The Report Contains

### 1. Model Structure

The report prints the actual PyTorch model structure using `str(model)`.

This lets you verify:
- coordinate embedding MLP
- encoder LSTM
- token embedding
- decoder LSTM
- linear output head

### 2. Sample-Level Input

For each sample, the report shows:
- trial id
- sequence length
- target sequence
- predicted sequence
- step-by-step input features

For the current baseline, each input step is:
- `(x, y, dx, dy)`

### 3. Encoder Output Preview

The report shows a short preview of:
- final encoder hidden state
- final encoder cell state

These are not the whole tensors, only a compact preview so the report stays readable.

### 4. Board Visualization

The report renders the 3x3 canonical board as text.

Cell format:
- `block_id:step_positions`

Example:
- `2:1,3` means block `2` appears at step 1 and step 3 in that sequence

The report prints:
- `Target Order`
- `Predicted Order`

This is the easiest way to visually compare input target and model output.

The inspection pipeline now also writes:
- PNG board images
- SVG board images

These make it easier to compare target vs predicted order at a glance.

### 5. Teacher Forcing Decoder Steps

This section shows what happens during supervised decoding.

For each step it records:
- decoder input token
- top predicted blocks
- probability of each top prediction

This answers:
- when the decoder is given the correct previous token, what does it think comes next?

### 6. Greedy Decoder Steps

This section shows actual inference behavior.

For each step it records:
- decoder input token
- top predicted blocks
- chosen block
- hidden state norm
- cell state norm

This answers:
- during free decoding, what is the model actually doing step by step?

## Recommended Reading Order

When you inspect a sample, read it in this order:

1. target sequence
2. input feature table
3. target board
4. predicted board
5. PNG / SVG board images
6. greedy decoder step table
7. teacher forcing step table

This makes it easy to connect:
- the raw input
- the correct answer
- the model's actual decision path

## Why This Matters

Metrics alone are not enough once we move to:
- delay / maintenance
- image input
- attention
- aging manipulations

This inspection pipeline gives us a stable, human-readable debugging baseline before the system becomes more complex.
