# Coordinate Base

This hub collects the coordinate-only baseline assets.

## What Lives Here

- `configs/`: coordinate training presets
- `reports/`: saved experiment writeups
- `docs/`: coordinate-specific analysis notes
- `scripts/`: easy entry points for training and model inspection

## Shared Code Used

- dataset: `corsi/data/coord_dataset.py`
- collate: `corsi/data/collate.py`
- model: `corsi/models/lstm_coord.py`
- training runtime: `corsi/training/train_coord.py`
- metrics: `corsi/analysis/metrics.py`

## Quick Start

```bash
python corsi/experiments/coordinate_base/scripts/train.py \
  --config corsi/experiments/coordinate_base/configs/coord_smoke_test.json
```
