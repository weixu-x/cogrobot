# Visual Base

This hub collects the robosuite visual baseline assets.

## What Lives Here

- `configs/`: visual training presets
- `docs/`: visual roadmap and design notes
- `reports/`: visual baseline result notes
- `scripts/`: camera, export, preview, and training entry points

## Shared Code Used

- environment + rollout: `corsi/envs/robosuite_corsi.py`
- dataset loader: `corsi/data/robosuite_visual_dataset.py`
- collate: `corsi/data/collate_visual.py`
- model: `corsi/models/lstm_visual.py`
- training runtime: `corsi/training/train_visual.py`
- export utilities: `corsi/scripts/`

## Quick Start

```bash
python corsi/experiments/visual_base/scripts/train.py \
  --config corsi/experiments/visual_base/configs/visual_smoke_test.json
```

## First Official Baseline

- train dataset root: `corsi_artifacts/visual_base/datasets/freecam_index_v1_train`
- val dataset root: `corsi_artifacts/visual_base/datasets/freecam_index_v1_val`
- config: `corsi/experiments/visual_base/configs/freecam_index_v1_baseline.json`
- report: `corsi/experiments/visual_base/reports/freecam_index_v1_baseline.md`
