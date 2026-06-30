# Corsi V2 Baseline Progress Log

## 2026-06-30T09:17:49+01:00 phase0-raw-generation
- Command: `conda run -n robosuite python -B -m corsi.data.generate_raw --config configs/corsi_motion_raw_len2_9_expanded800_20260630.json`
- Config changes: new raw config configs/corsi_motion_raw_len2_9_expanded800_20260630.json; split_length_counts train=4972 val=220 test=370
- Key results: raw episodes=5562; failed=0; split counts match targets; reports/dataset_manifest.md status=ok
- Checkpoint: `n/a`
## 2026-06-30T10:01:27+01:00 phase0-canonical-and-selfcheck
- Command: `conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.canonicalize --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_item_context_binding_auxsplit_dmem64_expanded800_20260630.json --overwrite`
- Config changes: new V2 expanded config points raw_dataset_root and canonical_root to expanded800_20260630; early_stopping_patience=30; auxsplit losses memory_order=0.3 memory_identity=0.1 memory_aux_orthogonal=0.01
- Key results: canonical episodes=5562; train/val/test=4972/220/370; fingerprint=94d1813b3f7b4cb5fd2107e9bcdd306683c807dacd18b1429380de1f5575df78; raw/canonical/leakage checks ok; reports/dataset_manifest.md ok
- Checkpoint: `n/a`
