# Experiment Hubs

This directory groups experiment-facing assets by track so they are easy to find without moving reusable core code.

## Tracks

- `coordinate_base/`
  - coordinate-only baseline configs, reports, inspection notes, and entry scripts
- `visual_base/`
  - robosuite visual baseline configs, roadmaps, camera / export entry scripts, and training entry scripts

## Core vs Experiment Split

Reusable code stays in the original shared modules:

- `corsi/envs/`
- `corsi/data/`
- `corsi/models/`
- `corsi/analysis/metrics.py`
- `corsi/training/device.py`
- `corsi/scripts/`

Experiment folders act as launchpads:

- configs
- experiment-specific reports / docs
- small wrapper scripts that call into the shared code
