# V2 K Ablation Plan - 2026-06-29

## Current State

Current active V2 uses `K=12`.

Evidence:

- `corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json`
- active canonical root: `corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12`
- current eval/run artifacts under `corsi_artifacts/memory_recall_v2/.../corsi_memory_recall_v2_k12`

Current V2 K=20 / K=30 memory-recall results:

NOT AVAILABLE IN CURRENT ARTIFACTS.

Deprecated old artifacts that must not be reused as V2 evidence:

- `corsi_artifacts/motion_base/datasets/freecam_motion_segment_uniform_len3_k20`
- `corsi_artifacts/motion_base/datasets/freecam_motion_segment_uniform_len3_k30`

Those are old motion-base qpos-knot artifacts, not current `scala_corsi_memory_recall_v2_canonical_v1` datasets.

## Post-Approval Generation Plan

Do not generate data or train until explicitly approved.

After approval, create:

- `corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k20.json`
- `corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k30.json`

Keep fixed:

- raw dataset root
- image size 128
- target schema
- split seed
- train/val/test IDs
- model hyperparameters
- training schedule
- evaluator

Change only:

- `k_samples_per_segment`
- `canonical_root`

Planned roots:

- `corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k20`
- `corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k30`

## Validation Gates

For each approved K:

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.validate \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k20.json \
  --check raw

conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.canonicalize \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k20.json \
  --overwrite

conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.validate \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k20.json \
  --check all
```

Repeat for K=30 after editing the config path.

## Fair Comparison Metrics

Compare K=12 / K=20 / K=30 with matched seeds and checkpoint selection:

- full sequence accuracy
- token accuracy
- EOS accuracy
- predicted length accuracy
- per-length exact/token/duplicate metrics for lengths 2-9
- serial-position accuracy
- duplicate sequence rate
- mean unique predicted blocks
- set-overlap Jaccard
- final-memory order probe

Report runtime and memory cost separately because larger K increases per-episode frame count.
