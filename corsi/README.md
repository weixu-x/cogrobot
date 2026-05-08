# Corsi Project

This directory contains the learning-oriented code for the embodied Corsi block-tapping project.

## Goals

The project is split into two layers:

- Simulation layer: board geometry, robosuite environments, and robot execution.
- Cognitive layer: sequence generation, datasets, models, training, and analysis.

The core task is not grasping. The robot keeps a fixed pointing hand shape and must reproduce a Corsi block sequence by pointing to the correct blocks in order.

## Directory Layout

- `corsi/envs/`: task logic and reusable robosuite helpers.
- `corsi/scripts/`: shared runnable utilities for robosuite demos, exports, and camera tools.
- `corsi/rendering/`: lightweight renderers and image generation tools.
- `corsi/data/`: reusable dataset definitions and collate functions.
- `corsi/models/`: reusable model definitions.
- `corsi/training/`: shared training runtimes and device helpers.
- `corsi/analysis/`: shared metrics and analysis utilities.
- `corsi/experiments/`: experiment hubs grouped by track.
  Current tracks: `coordinate_base/` and `visual_base/`.
  The current active visual baseline is `robosuite freecam -> keyframe dataset -> CNN + LSTM -> index sequence`.
  For scale-up runs, prefer sharded keyframe-only export instead of rollout-video export.

## Planned Milestones

### Milestone 1

- Fixed 9-block layout
- Coordinate-based sequence dataset
- Variable-length encoder-decoder LSTM
- Full-sequence prediction
- Accuracy by sequence length and estimated span

Current implementation status:
- layout and trial generator: started
- coordinate dataset and collate: started
- minimal coordinate LSTM and training entry point: started
- train entry point supports cross-platform torch device resolution (`auto / cpu / mps / cuda`)

Preset configs:
- `corsi/experiments/coordinate_base/configs/coord_smoke_test.json`
- `corsi/experiments/coordinate_base/configs/coord_baseline_mixed_2_6.json`
- `corsi/experiments/coordinate_base/configs/coord_baseline_mixed_2_9.json`

Example:

```bash
python corsi/experiments/coordinate_base/scripts/train.py \
  --config corsi/experiments/coordinate_base/configs/coord_smoke_test.json
```

### Milestone 2

- Delay / maintenance conditions
- Young vs aged baseline manipulations

### Milestone 3

- Image renderer
- Image-based dataset
- Visual encoder and attention

### Milestone 4

- Block-to-target action interface
- Robosuite execution loop for pointing / tapping

Direct local run:

```bash
.venv/bin/mjpython corsi/scripts/run_robosuite_corsi.py
```

## Working Rule

Whenever we add or change code in `corsi/`, we should also update:

- this `README.md` if the architecture or scope changes
- `corsi/WORKLOG.md` with what was implemented and why
