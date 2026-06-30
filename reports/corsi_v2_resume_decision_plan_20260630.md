# Corsi V2 Resume Decision Plan - 2026-06-30

## Current Hold

Training is intentionally paused for raw-data and model inspection.

Do not launch training, diagnostics, or further sweep jobs until a human explicitly chooses one of the routes below.

Authoritative inspection reports:

- `reports/raw_data_model_inspection_pause_20260630.md`
- `reports/stage2_direct_training_audit_20260630.md`

Paused checkpoint snapshot:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12_expanded800_20260630/pause_snapshot_stage2_direct_interrupt_20260630`

## Human Review Checklist

Raw/canonical data:

- Confirm raw dataset root is correct:
  `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_expanded800_20260630`
- Confirm canonical root is correct:
  `corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12_expanded800_20260630`
- Confirm split x length counts are intended:
  train L2=72, L3=300, L4=600, L5-L9=800 each; val L2=10, L3-L9=30 each; test L2=20, L3-L9=50 each.
- Confirm the L2 overlap behavior is acceptable:
  train uses all 72 ordered no-repeat L2 pairs, so L2 train/val and train/test overlap is unavoidable in the current design.
- Confirm canonical input shape:
  `images = [length, 12, 3, 128, 128]`.
- Confirm target token rule:
  `target_tokens = block_order + [EOS=9]`.

Model/training semantics:

- Confirm the model topology is intended:
  CNN visual encoder -> visual LSTM -> motor LSTM -> item projection -> item-context binding memory -> final-memory recall LSTM.
- Confirm `memory_write_mode = item_context_binding`.
- Confirm `recall_readout_mode = final`.
- Confirm Stage 1 and Stage 2 share topology in the current code path:
  `build_model(..., stage=...)` treats `stage` as no-op.
- Confirm whether Stage 1 should be part of the protocol.
  Current paused runs did not use Stage 1.

## Route A: Keep Direct Stage 2 Baseline

Use this route only if direct Stage 2 training from random initialization is acceptable.

Current direct Stage 2 checkpoints:

| Run | Latest epoch | Best full | Best token | Best val loss | Warm start |
| --- | ---: | ---: | ---: | ---: | --- |
| dmem64_seed0 | 40 | 0.9954545455 | 0.9993333333 | 0.0169906131 | no |
| dmem16_seed0 | 31 | 0.7727272727 | 0.9600000000 | 0.3119274649 | no |

Resume D_mem64 seed0 direct Stage 2:

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_item_context_binding_auxsplit_dmem64_expanded800_20260630.json \
  --stage 2 \
  --seed 0 \
  --device cuda:0 \
  --output-root corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12_expanded800_20260630 \
  --run-name stage2_seed0_binding_auxsplit_dmem64_expanded800_20260630 \
  --max-epochs 160 \
  --resume \
  --allow-full-training
```

Resume D_mem16 seed0 direct Stage 2:

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12_expanded800_20260630/configs/memory_recall_v2_k12_item_context_binding_auxsplit_dmem16_expanded800_20260630.json \
  --stage 2 \
  --seed 0 \
  --device cuda:1 \
  --output-root corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12_expanded800_20260630 \
  --run-name stage2_seed0_binding_auxsplit_dmem16_expanded800_20260630 \
  --max-epochs 160 \
  --resume \
  --allow-full-training
```

After D_mem64 seed0 finishes, Phase 2 diagnostics can proceed on its best checkpoint. After all sweep cells finish, Phase 3 summary can be generated.

## Route B: Treat Current Runs as Audit-Only and Rerun Staged Stage 1 -> Stage 2

Use this route if the intended protocol requires Stage 1 presentation grounding before Stage 2 recall training.

The current expanded800 data does not yet have a completed expanded800 Stage 1 checkpoint. First train Stage 1, then use its `best.pt` as `--warm-start-stage1-checkpoint` for Stage 2.

Proposed Stage 1 run for expanded800 seed0:

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_item_context_binding_auxsplit_dmem64_expanded800_20260630.json \
  --stage 1 \
  --seed 0 \
  --device cuda:0 \
  --output-root corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12_expanded800_20260630 \
  --run-name stage1_seed0_binding_auxsplit_expanded800_20260630 \
  --max-epochs 160 \
  --allow-full-training
```

Then Stage 2 D_mem64 seed0 warm-start run:

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_item_context_binding_auxsplit_dmem64_expanded800_20260630.json \
  --stage 2 \
  --seed 0 \
  --device cuda:0 \
  --output-root corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12_expanded800_20260630 \
  --run-name stage2_seed0_binding_auxsplit_dmem64_expanded800_stage1warm_20260630 \
  --max-epochs 160 \
  --warm-start-stage1-checkpoint corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12_expanded800_20260630/stage1_seed0_binding_auxsplit_expanded800_20260630/best.pt \
  --allow-full-training
```

For a staged sweep, run Stage 1 per seed or decide explicitly whether one Stage 1 checkpoint can be reused across seeds. The trainer's warm-start loader only copies grounding prefixes:

```python
(
    "visual_encoder.",
    "visual_lstm.",
    "motor_lstm.",
    "joint_head.",
    "ee_pose_head.",
    "ee_xy_head.",
)
```

Memory and recall parameters remain randomly initialized for Stage 2.

## Route C: Modify Protocol Before More Training

Use this route if inspection finds the raw data, canonicalization, memory architecture, Stage 1 objective, or Stage 2 objective should change.

Recommended actions:

1. Do not resume paused direct Stage 2 runs.
2. Record the rejected reason in `reports/stage2_direct_training_audit_20260630.md`.
3. Patch the data/model/training code or config.
4. Regenerate affected canonical artifacts if the data interface changes.
5. Start a new run namespace so old direct Stage 2 checkpoints remain auditable.

## Current Recommendation

From the code evidence, the paused runs are valid direct Stage 2 runs, but not valid staged Stage 1 -> Stage 2 runs.

If Stage 1 was expected scientifically, choose Route B and keep the current direct Stage 2 checkpoints as audit-only controls.
