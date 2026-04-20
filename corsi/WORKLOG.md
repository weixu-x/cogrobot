# Corsi Worklog

This file tracks implementation progress and design decisions so the project stays easy to modify later.

## Current Status

- The robosuite side already contains a custom Corsi arena and a demo script.
- The current demo is a rule-based pointing demonstration, not a trained memory model.
- The minimal coordinate-based learning system is now running.
- The next research target is to add delay / maintenance conditions on top of the coordinate baseline.

## Task Breakdown

### Phase 1: Environment and Trial Definition

- Define canonical 9-block layout
- Implement variable-length sequence generator
- Define trial metadata format

### Phase 2: Coordinate Dataset

- Build coordinate trial generator
- Implement `Dataset`
- Implement `collate_fn` with padding, lengths, and masks

### Phase 3: Minimal LSTM Model

- Coordinate embedding MLP
- Encoder-decoder LSTM
- Masked cross-entropy loss
- Greedy decoding

### Phase 4: Evaluation

- Token accuracy
- Full-sequence accuracy
- Accuracy by sequence length
- Error type breakdown

### Phase 5+: Delay, Vision, Attention, Aging

- Delay / maintenance experiments
- Image renderer and image dataset
- Visual encoder and motor-conditioned attention
- Aging manipulations

Delay / maintenance means:
- the model first observes the presentation sequence
- then there is a gap before recall starts
- during that gap, the model must keep the sequence active in memory without new informative input

Planned delay conditions:
- `no_delay`: encoder output goes directly into decoding
- `short_delay`: a small number of blank maintenance steps before decoding
- `long_delay`: a larger number of blank maintenance steps before decoding

Purpose:
- measure retention, not just immediate sequence replay
- create a clean entry point for later aging manipulations such as memory leak or hidden-state noise

## Design Decisions

### Decision 1

The cognitive model predicts the full block sequence.

Implementation detail:
- The decoder still generates one token at a time.
- The training target is the entire output sequence.

### Decision 2

The first learning system uses coordinates, not images.

Reason:
- It isolates memory performance from perception and motor noise.
- It is the cleanest baseline before adding visual input or action coupling.

### Decision 3

Robosuite should remain the simulation substrate, not the main home for training code.

Reason:
- Environments belong in `robosuite/`.
- Learning code belongs in `corsi/`.

## Change Log

### 2026-03-24

- Created the standalone `corsi/` project area for cognitive-task code.
- Added project-level documentation and a persistent worklog.
- Added `corsi/envs/sequence_generator.py` for canonical layout, variable-length sequences, and coordinate trial generation.
- Added `corsi/data/coord_dataset.py` for the minimal coordinate-based dataset wrapper.
- Added `corsi/data/collate.py` for variable-length batch padding and masking with delayed PyTorch import.
- Added `corsi/models/lstm_coord.py` with a minimal encoder-decoder LSTM for full-sequence block prediction.
- Added `corsi/analysis/metrics.py` for token accuracy, full-sequence accuracy, length-wise accuracy, span estimation, and error breakdown.
- Added `corsi/training/train_coord.py` as the first end-to-end training entry point for the coordinate-only system.
- Added `corsi/envs/robosuite_corsi.py` so robosuite demos and smoke tests can share the same Corsi pointing setup.
- Refactored `robosuite/demos/demo_test.py` to use the shared robosuite Corsi helpers.
- Added `robosuite/demos/demo_corsi_smoke_test.py` for direct visual smoke tests driven by generated or manual block sequences.
- Added `corsi/training/device.py` for cross-platform PyTorch device resolution.
- Added coordinate training preset configs under `corsi/configs/`.
- Updated `.gitignore` to exclude local training artifacts under `corsi_artifacts/`.
- Ran the first formal coordinate baseline training for `mixed-span 2-6` on Apple MPS.
- Recorded the baseline result in a dedicated experiment note under `corsi/experiments/`.

## Implementation Notes

### Minimal Module 1: Sequence Generator

Current file:
- `corsi/envs/sequence_generator.py`

Implemented:
- canonical 3x3 layout
- random sequence generation
- no immediate repeats
- forward / backward mode
- coordinate mapping
- delta-coordinate computation
- trial collection generation

### Minimal Module 2: Coordinate Dataset

Current file:
- `corsi/data/coord_dataset.py`

Implemented output fields:
- `coords`
- `targets`
- `length`
- `trial_id`
- `layout`
- `mode`

Feature modes:
- `xy`
- `xydxdy`

### Minimal Module 3: Collate Function

Current file:
- `corsi/data/collate.py`

Implemented:
- variable-length padding
- `lengths`
- `mask`
- padded coordinate tensor
- padded target tensor

Note:
- PyTorch is imported lazily inside the collate function because the current environment does not yet have `torch` installed.

### Minimal Module 4: Coordinate LSTM

Current file:
- `corsi/models/lstm_coord.py`

Implemented:
- coordinate embedding MLP
- LSTM encoder
- token embedding with a dedicated `<START>` token
- LSTM decoder
- full-sequence logits over 9 blocks
- greedy decoding

Important detail:
- the task target is the full output sequence
- the decoder still generates one token at a time internally

### Minimal Module 5: Metrics

Current file:
- `corsi/analysis/metrics.py`

Implemented:
- token accuracy
- full-sequence accuracy
- accuracy by sequence length
- estimated span
- sequence error classification

Current error labels:
- `correct`
- `order_error`
- `wrong_block`
- `repeated_block`
- `early_collapse`

### Minimal Module 6: Training Entry Point

Current file:
- `corsi/training/train_coord.py`

Implemented:
- synthetic train / val dataset generation
- dataloaders
- masked cross-entropy loss
- greedy-decoding evaluation
- checkpoint saving
- JSON-style epoch logging

Current limitation:
- runtime training still requires installing PyTorch in the environment

Update:
- PyTorch is now installed in the local virtual environment.
- The minimal coordinate smoke test has already run successfully on CPU.

### Minimal Module 7: Robosuite Smoke Test

Current files:
- `corsi/envs/robosuite_corsi.py`
- `robosuite/demos/demo_corsi_smoke_test.py`

Implemented:
- shared custom hand + robot registration
- shared Corsi robosuite environment construction
- sequence-driven pointing rollout
- online robosuite visualization
- offline video export

Purpose:
- let us run a sequence-level smoke test visually inside robosuite before formal training
- keep the smoke-test behavior aligned with the future action-execution layer

Current note:
- the coordinate baseline uses a fixed canonical layout
- the current robosuite arena still samples physical block positions randomly
- for now the robosuite smoke test validates sequence execution by block index, not exact layout matching

### Minimal Module 8: Device Resolution

Current file:
- `corsi/training/device.py`

Implemented:
- automatic device selection
- explicit `cpu / mps / cuda` requests
- device availability recording

Current behavior:
- `auto` prefers `cuda`, then `mps`, then `cpu`
- training writes `device_info.json` into the output directory

### Minimal Module 9: Training Presets

Current files:
- `corsi/configs/coord_smoke_test.json`
- `corsi/configs/coord_baseline_mixed_2_6.json`
- `corsi/configs/coord_baseline_mixed_2_9.json`

Implemented:
- JSON config loading in `train_coord.py`
- reusable smoke-test preset
- reusable coordinate baseline presets for span 2-6 and 2-9

## Experiment Notes

### Baseline 1: Coordinate Mixed-Span 2-6

Run date:
- 2026-03-24

Config:
- `corsi/configs/coord_baseline_mixed_2_6.json`

Device:
- requested: `auto`
- resolved: `mps`

Result:
- best epoch: `20`
- best full-sequence accuracy: `1.0`
- best token accuracy: `1.0`
- estimated span: `6`

Length-wise best accuracy:
- length 2: `1.0`
- length 3: `1.0`
- length 4: `1.0`
- length 5: `1.0`
- length 6: `1.0`

Interpretation:
- the coordinate-only baseline is fully learnable for span 2-6 under the current synthetic setup
- this gives us a strong clean baseline before introducing delay, visual input, or aging noise

Reference:
- `corsi_artifacts/coord_baseline_mixed_2_6/`
- `corsi/experiments/coord_baseline_mixed_2_6.md`

### 2026-04-13

- Added `corsi/docs/visual_corsi_roadmap.md` as a detailed execution plan for the next research stages.
- Broke the roadmap into visual input, delay / blank interval, heatmap output, event-structured fine time, and robosuite transfer.
- Recorded the intended file-level changes for each stage so implementation can proceed directly from the checklist.
- Corrected the standard Corsi-to-robosuite mapping so the reference figure's block centers, not block bottom-left corners, define the robosuite target positions.
- Rotated the standard board coordinates onto the robosuite table plane so the `agentview` image matches the reference board orientation more closely, with human blocks 7 and 8 appearing in the lower-left region.
- Added a visible scaled outer frame in `CorsiTableArena` matching the reference board aspect ratio `255 x 205`.
- Added `corsi/scripts/make_robosuite_sequence_figure.py` to build a single summary image from exported robosuite reset frames and sequence keyframes, with both 0-based indices and human block ids shown.
- Added `corsi/scripts/capture_mjviewer_camera.py` and stored the user-tuned free-camera parameters as the default online Corsi free-camera preset when `online_render_camera=None`.
- Added a minimal robosuite visual dataset path: per-trial manifests, a batch export script, a visual dataset loader, and a visual collate function so `freecam` keyframes can flow directly into training code.
- Added the first visual baseline stack: `VisualSeq2SeqLSTM`, `train_visual.py`, and a `visual_smoke_test.json` config so robosuite `freecam` keyframes can be trained with a CNN + LSTM + index decoding pipeline.
- Reorganized experiment-facing assets into `corsi/experiments/coordinate_base/` and `corsi/experiments/visual_base/`, while keeping shared code in `envs / data / models / training / analysis / scripts`.
- Moved configs, reports, and roadmap-style docs under those experiment hubs, and grouped `corsi_artifacts/` into `coordinate_base/` and `visual_base/{camera, previews, datasets, training, legacy}`.
- Scaled the visual baseline interface from preview-only roots toward official train / val dataset roots by adding richer dataset manifest metadata, dataset-info export during training, and baseline-specific visual experiment documentation.

### 2026-04-20

- Exported the first official robosuite `freecam` train / val visual baseline splits:
  - `corsi_artifacts/visual_base/datasets/freecam_index_v1_train`
  - `corsi_artifacts/visual_base/datasets/freecam_index_v1_val`
- Added `corsi/experiments/visual_base/configs/freecam_index_v1_baseline.json` as the first fixed-root train / val visual training config.
- Ran the first `freecam` visual baseline training to `corsi_artifacts/visual_base/training/freecam_index_v1_baseline`.
- Recorded the initial baseline report in `corsi/experiments/visual_base/reports/freecam_index_v1_baseline.md`.
- The pipeline is now stable end to end, but the model performance is still weak; the next step remains scaling the baseline dataset before moving on to delay, heatmap, or timing variants.
