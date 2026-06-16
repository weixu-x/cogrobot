# Visual Corsi Roadmap

This document turns the next-stage Corsi plan into an execution checklist.
The goal is that we can open this file and know exactly what to build next.

## Main Principle

Do not change all three axes at once:

- input modality: coordinates -> images
- memory condition: no delay -> blank / maintenance delay
- output format: block index -> spatial heatmap

Recommended order:

1. keep the current coordinate baseline as the control
2. add a synthetic visual-input baseline that still predicts block indices
3. add delay / blank interval on top of the visual baseline
4. compare index prediction and heatmap prediction
5. only then test finer-grained temporal unfolding

This order keeps the experiments interpretable.

## Current Baseline Snapshot

Existing files:

- `corsi/envs/sequence_generator.py`
- `corsi/data/coord_dataset.py`
- `corsi/data/collate.py`
- `corsi/models/lstm_coord.py`
- `corsi/training/train_coord.py`
- `corsi/analysis/metrics.py`

Current data flow:

- sequence generator -> coordinate features `(x, y, dx, dy)`
- coordinate dataset -> padded batch
- encoder-decoder LSTM -> full block-index sequence
- metrics -> token accuracy, full-sequence accuracy, error breakdown

Keep this baseline untouched as the reference condition.

## Phase 0: Freeze The Coordinate Baseline

### Goal

Make the current minimal setup the stable comparison point for all future variants.

### Deliverables

- a clearly named baseline experiment directory
- one smoke-test config
- one standard config for later comparison against visual models

### TODO

- [ ] Re-run the coordinate baseline once before adding visual code.
- [ ] Save the exact config used for comparison.
- [ ] Record the baseline numbers we will compare against later:
  - full-sequence accuracy
  - token accuracy
  - accuracy by sequence length
  - error breakdown
- [ ] Keep one short note explaining what the baseline can and cannot test.

### Files

Likely no new code required if the current baseline is already stable.

Optional documentation updates:

- `corsi/experiments/coordinate_base/reports/coord_baseline_mixed_2_6.md`
- `corsi/WORKLOG.md`

### Exit Criteria

- we have one fixed coordinate baseline to compare every later model against
- no ambiguity remains about which run is the reference run

## Phase 1: Build A Synthetic Visual Dataset

### Goal

Replace direct coordinate input with image frames, but keep the task simple:

- input: presentation frames
- target: block index sequence

This isolates perception + memory without introducing robosuite complexity yet.

### Target Data Flow

`trial -> rendered frames -> CNN encoder -> temporal model -> index sequence`

### Design Choice

Start with 2D synthetic rendered frames, not robosuite camera frames.

Reason:

- easier to debug
- faster to generate
- easier to control appearance
- easier to add delay later

### Dataset Spec

Each training sample should contain:

- `trial_id`
- `sequence`
- `length`
- `layout`
- `frames`
- `frame_mask`
- `event_types`
- `active_block_per_frame`

Recommended tensor shape:

- `frames`: `[T, C, H, W]`
- `frame_mask`: `[T]`
- `targets`: `[L]`

Recommended default event structure for the first version:

- one frame per presented item
- background board stays constant
- one block is highlighted in each frame

Do not add delay yet in this phase.

### Code Tasks

#### 1. Add a frame renderer for presentation sequences

New file:

- `corsi/rendering/frame_sequence_renderer.py`

Implement:

- board renderer that outputs a full RGB frame
- function to highlight exactly one active block
- control over image size, colors, padding, and block appearance
- optional light appearance randomization:
  - background color jitter
  - block color jitter
  - small position jitter

Recommended API:

```python
render_trial_frame_sequence(
    layout,
    sequence,
    image_size=(128, 128),
    block_style="flat",
    add_noise=False,
) -> list[PIL.Image.Image]
```

#### 2. Add a visual dataset

New file:

- `corsi/data/visual_dataset.py`

Implement:

- wrap existing `CorsiTrial` generation
- call the renderer for each trial
- convert frames to numeric arrays or tensors
- return the fields listed in the dataset spec

Recommended class:

```python
class VisualCorsiDataset:
    ...
```

#### 3. Add visual collate logic

New file:

- `corsi/data/collate_visual.py`

Implement:

- pad variable-length frame sequences
- return:
  - `frames_pad`
  - `frame_lengths`
  - `frame_mask`
  - `targets_pad`
  - `target_lengths`

#### 4. Export the new dataset helpers

Modify:

- `corsi/data/__init__.py`

Add exports for:

- `VisualCorsiDataset`
- `collate_visual_batch`

#### 5. Export the new renderer

Modify:

- `corsi/rendering/__init__.py`

Add exports for the frame-sequence renderer.

### Validation Tasks

- [ ] Add a small script that renders 5 to 10 sample trials to disk.
- [ ] Visually inspect that the highlighted block order matches the sequence.
- [ ] Confirm padding and batch shapes are correct.
- [ ] Confirm the first visual dataset can be iterated without training.

### Optional Support Script

New file:

- `corsi/scripts/render_visual_corsi_samples.py`

Purpose:

- render a few dataset examples
- save grids or frame strips
- make debugging much easier before model training

### Exit Criteria

- visual frames are generated reliably
- a batch can be loaded with consistent tensor shapes
- manual inspection confirms that frame order matches target order

## Phase 2: Train A Visual Baseline That Still Predicts Indices

### Goal

Add the first perception-plus-memory model while keeping the output head simple.

Input:

- frame sequence

Output:

- block index sequence

### Model Recommendation

Start with:

- per-frame CNN encoder
- temporal LSTM
- decoder that predicts block indices

Use the coordinate baseline as the conceptual template.

### Proposed Architecture

Option A, simplest:

- `frame_t -> CNN -> feature_t`
- `[feature_1 ... feature_T] -> encoder LSTM`
- final hidden state -> decoder LSTM
- decoder outputs `index_1 ... index_L`

Option B, simpler still:

- `frame_t -> CNN -> feature_t`
- `[feature_1 ... feature_T] -> temporal LSTM`
- per-step classifier over the same length

Use Option A first because it matches the current coordinate model and makes comparison cleaner.

### Code Tasks

#### 1. Add a visual model file

New file:

- `corsi/models/lstm_visual.py`

Implement:

- `VisualLSTMConfig`
- lightweight CNN encoder
- temporal encoder
- token decoder
- greedy decoding

Recommended first CNN:

- 3 to 4 conv blocks
- small model, because the visual task is initially simple

#### 2. Export the visual model

Modify:

- `corsi/models/__init__.py`

Add:

- `VisualLSTMConfig`
- `VisualSeq2SeqLSTM`

#### 3. Add a visual training entry point

New file:

- `corsi/training/train_visual.py`

Implement:

- config parsing
- visual dataset creation
- visual dataloaders
- train loop
- eval loop
- checkpoint saving
- JSONL epoch logging

Reuse as much structure as possible from:

- `corsi/training/train_coord.py`

#### 4. Add visual configs

New files:

- `corsi/experiments/visual_base/configs/visual_smoke_test.json`
- `corsi/configs/visual_baseline_mixed_2_6.json`

Suggested config fields:

- `image_height`
- `image_width`
- `cnn_channels`
- `feature_dim`
- `hidden_dim`
- `batch_size`
- `learning_rate`
- `seq_min`
- `seq_max`
- `train_trials`
- `val_trials`

#### 5. Add an inspection script

New file:

- `corsi/analysis/inspect_visual_model.py`

Implement:

- load checkpoint
- render or save a few validation examples
- print target sequence and predicted sequence
- optionally save the frame strip used as input

### Validation Tasks

- [ ] Overfit a tiny visual batch first.
- [ ] Confirm that the model can reach near-perfect performance on a smoke test.
- [ ] Compare visual baseline performance against coordinate baseline.
- [ ] Inspect failure cases manually to separate perception errors from memory errors.

### Exit Criteria

- smoke-test visual training runs end to end
- the model learns the synthetic frame task
- we have a stable visual baseline before adding delay

## Phase 3: Add Delay / Blank Interval

### Goal

Turn the task from immediate sequence mapping into a real short-term maintenance task.

### New Data Flow

`presentation frames -> blank frames -> recall`

The model should receive no new informative spatial content during the blank interval.

### Important Design Rule

The blank period should not accidentally leak the answer.

During delay frames:

- same empty board
- no highlighted block
- no cue indicating the next target

### Dataset Design

Expand each trial with timing metadata:

- `presentation_steps`
- `delay_steps`
- `recall_steps`
- `event_types`

Recommended event labels:

- `present`
- `delay`
- `recall`

For the first delay version, the model can still decode after the encoder stage.
The important part is that the input sequence now includes blank maintenance steps.

### Code Tasks

#### 1. Add a timing / schedule helper

New file:

- `corsi/envs/trial_schedule.py`

Implement:

- build a per-trial temporal schedule
- support:
  - `no_delay`
  - `short_delay`
  - `long_delay`
- map a logical trial into frame events

Recommended API:

```python
build_trial_schedule(
    sequence_length,
    presentation_frames_per_item=1,
    delay_frames=0,
)
```

#### 2. Extend the frame renderer to support blank frames

Modify:

- `corsi/rendering/frame_sequence_renderer.py`

Add:

- render empty-board frames
- render per-event sequences based on schedule metadata

#### 3. Extend the visual dataset

Modify:

- `corsi/data/visual_dataset.py`

Add support for:

- delay condition
- schedule generation
- blank-frame insertion
- event labels in the returned sample

#### 4. Update the collate function

Modify:

- `corsi/data/collate_visual.py`

Ensure the batch includes:

- frame masks
- event labels
- optional delay-condition metadata

#### 5. Add delay-aware configs

New files:

- `corsi/configs/visual_delay_short_mixed_2_6.json`
- `corsi/configs/visual_delay_long_mixed_2_6.json`

### Experiment Tasks

- [ ] Train `no_delay`, `short_delay`, and `long_delay` under the same base architecture.
- [ ] Compare degradation by sequence length.
- [ ] Compare error breakdown shifts:
  - order error
  - wrong block
  - early collapse
- [ ] Check whether delay hurts long sequences disproportionately.

### Exit Criteria

- delay can be switched on and off by config
- blank frames are verified visually
- we have a clean retention curve across delay lengths

## Phase 4: Add A Heatmap Output Head

### Goal

Reduce dependence on a discrete hand-designed index output and move closer to spatial prediction.

### Recommendation

Do not replace the index head immediately.
First add the heatmap head as a second comparable output mode.

Recommended order:

1. keep index head working
2. add heatmap head
3. compare both heads under the same encoder
4. decide later whether to remove index output

### Heatmap Design

For each recall item, predict a 2D map over board space.

Two practical choices:

- hard target: one-hot heatmap at the target block center
- soft target: Gaussian blob centered on the target block

Prefer Gaussian targets first because training is smoother.

### Code Tasks

#### 1. Add heatmap utilities

New file:

- `corsi/rendering/heatmap_targets.py`

Implement:

- convert a block location into a 2D target map
- configurable heatmap size
- configurable Gaussian sigma

#### 2. Extend the visual dataset with heatmap targets

Modify:

- `corsi/data/visual_dataset.py`

Add optional returned fields:

- `target_heatmaps`
- `heatmap_size`

#### 3. Add a heatmap model head

Modify:

- `corsi/models/lstm_visual.py`

Support:

- `output_mode="index"`
- `output_mode="heatmap"`
- optional dual-head mode later

The heatmap head can start as:

- linear projection from decoder hidden state
- reshape to `[H, W]`

#### 4. Add heatmap losses

Modify:

- `corsi/training/train_visual.py`

Support:

- cross-entropy for index output
- MSE or BCE loss for heatmap output

#### 5. Add heatmap evaluation metrics

Modify:

- `corsi/analysis/metrics.py`

Add:

- nearest-block decoded accuracy from predicted heatmap
- spatial distance error
- full-sequence accuracy after nearest-block decoding

### Experiment Tasks

- [ ] Compare index and heatmap heads under matched settings.
- [ ] Check whether heatmaps improve robustness to small layout perturbations.
- [ ] Inspect whether heatmap errors are spatially local rather than arbitrary.

### Exit Criteria

- heatmap targets are generated correctly
- the model can train with heatmap supervision
- we can compare discrete and spatial recall under one framework

## Phase 5: Add Event-Structured Fine Time, Not Naive Interpolation

### Goal

Test whether finer temporal unfolding helps model short-term spatial memory in a meaningful way.

### Important Caution

Do not simply split one item transition into 5 interpolated coordinates unless there is a clear reason.

Why:

- it increases sequence length
- it makes optimization harder
- it may not make the task more cognitively meaningful

### Better Approach

Represent each item with event-structured micro-steps:

- `onset`
- `dwell`
- `offset`
- `blank`

This better matches the idea that cognition is driven by event boundaries, not arbitrary interpolation points.

### Code Tasks

#### 1. Extend the schedule helper

Modify:

- `corsi/envs/trial_schedule.py`

Add configurable micro-timing:

- `onset_frames`
- `dwell_frames`
- `offset_frames`
- `inter_item_blank_frames`

#### 2. Extend the renderer

Modify:

- `corsi/rendering/frame_sequence_renderer.py`

Add support for:

- onset brightness
- stable dwell highlight
- fade or off phase
- inter-item blank

#### 3. Extend the visual dataset config

Modify:

- `corsi/data/visual_dataset.py`
- `corsi/training/train_visual.py`

Add config-driven control over event timing.

#### 4. Add experiments focused on temporal granularity

New files:

- `corsi/configs/visual_delay_event_timing_v1.json`
- `corsi/experiments/visual_event_timing.md`

### Experiment Questions

- Does finer event timing improve retention or just make training harder?
- Does performance degrade smoothly with longer effective sequences?
- Do error types become more human-like, or only more noisy?

### Exit Criteria

- fine-grained time is represented with meaningful event structure
- we avoid conflating temporal length with cognitive realism

## Phase 6: Bridge To Robosuite Visual Input

### Goal

Move from synthetic visual input to rendered simulation frames.

This should happen only after the synthetic visual version is stable.

### Strategy

Start by generating supervised datasets from robosuite scenes rather than training online control immediately.

### Code Tasks

#### 1. Add a robosuite visual dataset generation script

New file:

- `corsi/scripts/render_robosuite_corsi_dataset.py`

Implement:

- generate or load a sequence
- render camera frames for each presentation event
- save frame tensors or image files plus metadata

#### 2. Add a robosuite-backed dataset wrapper

New file:

- `corsi/data/robosuite_visual_dataset.py`

Implement:

- dataset over saved robosuite frame sequences
- same interface as `VisualCorsiDataset` where possible

#### 3. Add domain-gap experiments

New files:

- `corsi/configs/robosuite_visual_smoke_test.json`
- `corsi/experiments/robosuite_visual_transfer.md`

### Experiment Tasks

- [ ] Compare synthetic-only training vs robosuite-only training.
- [ ] Try synthetic pretraining + robosuite finetuning.
- [ ] Measure how much camera viewpoint and rendering variation matter.

### Exit Criteria

- robosuite frame data can be loaded in the same training framework
- we can measure transfer from simple synthetic perception to simulated perception

## Cross-Cutting Refactors

These are not a separate phase, but they will make later stages much easier.

### Refactor 1: Shared Config Objects

Eventually add a shared config module so `train_coord.py` and `train_visual.py` do not drift too far apart.

Possible new file:

- `corsi/configs/schema.py`

### Refactor 2: Shared Seq2Seq Helpers

If the coordinate and visual models share the same decoder logic, extract common parts.

Possible new file:

- `corsi/models/seq2seq_common.py`

### Refactor 3: Shared Experiment Logging

Reuse:

- checkpoint naming
- epoch log format
- final summary structure

Possible new file:

- `corsi/training/train_utils.py`

## Suggested Execution Order For The Very Next Sessions

If we want the most practical next actions, do them in this order:

### Session 1

- [ ] Create `corsi/rendering/frame_sequence_renderer.py`
- [ ] Create `corsi/data/visual_dataset.py`
- [ ] Create `corsi/data/collate_visual.py`
- [ ] Create `corsi/scripts/render_visual_corsi_samples.py`
- [ ] Render and inspect sample visual trials

### Session 2

- [ ] Create `corsi/models/lstm_visual.py`
- [ ] Create `corsi/training/train_visual.py`
- [ ] Add `visual_smoke_test.json`
- [ ] Overfit a tiny sample
- [ ] Run the first end-to-end visual smoke test

### Session 3

- [ ] Add delay support with `corsi/envs/trial_schedule.py`
- [ ] Extend renderer and dataset for blank frames
- [ ] Run `no_delay` vs `short_delay` vs `long_delay`

### Session 4

- [ ] Add heatmap targets and heatmap head
- [ ] Compare index and heatmap outputs under the same encoder

### Session 5

- [ ] Add event-structured micro-timing
- [ ] Decide whether the fine-time version is useful enough to keep

## Practical Notes For Implementation

### Keep dataset interfaces stable

Try to keep coordinate and visual datasets structurally similar:

- `trial_id`
- `length`
- `targets`
- `layout`

This will make evaluation and analysis code easier to reuse.

### Prefer config switches over separate forks

Examples:

- `input_mode=coord|visual`
- `delay_frames=0|3|10`
- `output_mode=index|heatmap`

This will reduce duplicated scripts.

### Add visual sanity checks early

For any new visual stage, save sample images before training.
It is much cheaper than debugging the model after the fact.

### Keep the baseline metrics comparable

Even after adding heatmaps, always keep a decoded block-sequence view so the results remain comparable with the coordinate baseline.

## Definition Of Success

This roadmap is successful if, after following it, we can answer:

- how much harder the task becomes when perception is added
- how much delay specifically hurts retention
- whether spatial heatmap prediction changes the error structure
- whether finer temporal unfolding gives better cognitive alignment or only extra optimization burden
