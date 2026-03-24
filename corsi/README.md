# Corsi Project

This directory contains the learning-oriented code for the embodied Corsi block-tapping project.

## Goals

The project is split into two layers:

- Simulation layer: board geometry, robosuite environments, and robot execution.
- Cognitive layer: sequence generation, datasets, models, training, and analysis.

The core task is not grasping. The robot keeps a fixed pointing hand shape and must reproduce a Corsi block sequence by pointing to the correct blocks in order.

## Directory Layout

- `corsi/envs/`: task logic and sequence generation that do not need robosuite registration.
  Includes reusable robosuite helpers for Corsi pointing demos.
- `corsi/rendering/`: lightweight renderers and image generation tools.
- `corsi/data/`: dataset definitions and collate functions.
- `corsi/models/`: LSTM and later visual / attention models.
- `corsi/training/`: training and inference entry points.
- `corsi/analysis/`: metrics and error analysis.
- `corsi/configs/`: experiment configs.
  Includes ready-to-run coordinate training presets.

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
- `corsi/configs/coord_smoke_test.json`
- `corsi/configs/coord_baseline_mixed_2_6.json`
- `corsi/configs/coord_baseline_mixed_2_9.json`

Example:

```bash
python corsi/training/train_coord.py --config corsi/configs/coord_smoke_test.json
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

## Working Rule

Whenever we add or change code in `corsi/`, we should also update:

- this `README.md` if the architecture or scope changes
- `corsi/WORKLOG.md` with what was implemented and why
