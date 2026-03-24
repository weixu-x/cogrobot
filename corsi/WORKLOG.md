# Corsi Worklog

This file tracks implementation progress and design decisions so the project stays easy to modify later.

## Current Status

- The robosuite side already contains a custom Corsi arena and a demo script.
- The current demo is a rule-based pointing demonstration, not a trained memory model.
- The next coding target is the minimal coordinate-based learning system.

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
