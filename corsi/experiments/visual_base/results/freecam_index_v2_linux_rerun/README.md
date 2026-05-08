# freecam_index_v2 Linux rerun results

This directory contains the lightweight, version-controlled result artifacts for
the `freecam_index_v2` visual ablation rerun.

The full training artifacts under `corsi_artifacts/` remain ignored by Git.
Model checkpoints (`*.pt`) and bulky dataset metadata are intentionally not
included here.

Included per experiment:

- `config.json`
- `device_info.json`
- `metrics_best.json`
- `metrics_last.json`
- `final_summary.json`
- `results_summary.json`
- `train_log.jsonl`

Included summary files:

- `summary/ablation_summary.csv`
- `summary/ablation_summary.md`
