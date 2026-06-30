# V2 LSTM Memory Onset Plan - Priority 2

Date: 2026-06-29  
Repository: `/home/wei2025/Developer/cogrobot`  
Status: planning only. No training, canonicalization, architecture change, commit, or push was performed.

## Source Status

The Priority 1 fact base is present and was used:

- `reports/v2_current_model_fact_base_20260629.md`
- `reports/v2_memory_isolates_snapshot_20260626.md`
- current V2 config/artifacts under `corsi_artifacts/memory_recall_v2`

This plan therefore does not rely on the fallback-only `PROJECT_STATE.md` path, except as supporting evidence already cited by the fact base.

## Research Questions

Primary question: at what training budget or epoch range does LSTM memory begin to show useful sequence memory and autonomous recall?

Secondary question: at equal budget, how does LSTM memory compare against slot-compress memory?

The experiment should separate optimization speed from representational capacity. A slow LSTM onset is not the same result as an incapable LSTM memory.

## Fixed Variables

Keep these fixed for all runs in this plan:

| Variable | Fixed value |
|---|---|
| Dataset | current active V2 K=12 canonical dataset |
| Canonical root | `corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12` |
| Canonical fingerprint | `782962d47e55a1e4631919f4042bc1eb31ff0b66c7c27adff27aaf7b994b4485` |
| K | `12` |
| D_mem | `64` |
| Seed | start with `0` |
| Stage 1 warm start | `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage1_seed0_full_20260624/best.pt` |
| Stage 2 loss | current default `combined_v2_loss` weights |
| Evaluation | current evaluator and checkpoint-selection rules |
| Splits | existing deterministic train/val/test split, length-balanced `320 / 40 / 40` |

Do not change CNN architecture, visual LSTM architecture, recall decoder architecture, loss weights, split, K, D_mem, training data, or metrics.

## Existing Evidence

### Slot-Compress D_mem=64

Current production checkpoint family:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624`

Selected checkpoint:

`best_full_sequence.pt`, epoch `63`.

| Split | Full sequence | Token | EOS | Pred length | Duplicate sequence rate |
|---|---:|---:|---:|---:|---:|
| val | 0.325 | 0.627 | 0.875 | 0.875 | 0.525 |
| test | 0.300 | 0.631 | 0.900 | 0.850 | 0.575 |

Per-length exact recall is strong at the short end: val length 2/3 exact is `1.000/0.800`; test length 2/3 exact is `1.000/1.000`. Exact recall is still zero for long sequences: val lengths 7-9 and test lengths 5-9.

Final-memory evidence is strong for slot-compress D_mem=64:

| Checkpoint | Val final-memory order probe | Test final-memory order probe | Known-length exact |
|---|---:|---:|---:|
| best full | 0.882 | 0.877 | 0.675 |
| best val loss | 0.886 | 0.895 | 0.675 |

Interpretation: slot-compress D_mem=64 already stores ordered identity well enough for a probe, while autonomous recall remains limited by duplicate collapse and readout.

### Existing LSTM D_mem=64

Existing run:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_dmem64_full_20260624`

Config:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_dmem64.json`

This is LSTM memory because `memory_write_mode` is omitted and the model default is `lstm`; the config sets `D_mem=64`, `memory_dim=64`, and `memory_noise_std=0.0`.

Best-full metrics:

| Split | Full sequence | Token | EOS | Pred length | Duplicate sequence rate |
|---|---:|---:|---:|---:|---:|
| val | 0.050 | 0.392 | 1.000 | 0.975 | 0.825 |
| test | 0.050 | 0.427 | 1.000 | 0.975 | 0.750 |

Final-memory probe evidence for the LSTM run is weak on held-out splits:

| Split | Final-memory order probe | Known-length exact |
|---|---:|---:|
| train | 0.934 | 0.725 |
| val | 0.332 | 0.050 |
| test | 0.432 | 0.125 |

The segment/item block probes are still `1.0` on train/val/test, so the upstream presentation representation is not the obvious bottleneck. The LSTM memory appears to learn something on train, but the held-out final-memory order probe and autonomous recall are much weaker than slot-compress.

### Existing Milestone Evidence

The existing summaries contain per-epoch validation metrics, but the run did not retain numbered milestone checkpoints for full test/probe analysis. Existing history is enough to show learning trend, not enough for the full onset experiment.

LSTM D_mem=64 history:

| Milestone | Train loss | Val loss | Val token | Val full | Val length-2 exact | Val length-3 exact |
|---:|---:|---:|---:|---:|---:|---:|
| epoch 10 | 1.891 | 1.896 | 0.223 | 0.000 | 0.000 | 0.000 |
| epoch 20 | 1.875 | 1.895 | 0.235 | 0.000 | 0.000 | 0.000 |
| epoch 40 | 1.824 | 1.873 | 0.258 | 0.000 | 0.000 | 0.000 |
| epoch 80 | 1.465 | 1.533 | 0.392 | 0.050 | 0.400 | 0.000 |

Slot-compress D_mem=64 history:

| Milestone | Train loss | Val loss | Val token | Val full | Val length-2 exact | Val length-3 exact | Val length-4 exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| epoch 10 | 1.897 | 2.039 | 0.396 | 0.050 | 0.400 | 0.000 | 0.000 |
| epoch 20 | 1.468 | 1.647 | 0.465 | 0.075 | 0.600 | 0.000 | 0.000 |
| epoch 40 | 0.743 | 1.217 | 0.619 | 0.200 | 1.000 | 0.400 | 0.200 |
| epoch 80 | 0.213 | 1.298 | 0.654 | 0.275 | 1.000 | 0.400 | 0.600 |

Interpretation: the existing LSTM run is not absent, but it is insufficient. It shows slow optimization onset by epoch 80, but it lacks retained milestone checkpoints, multi-seed evidence, and 120/160-epoch evidence. Slot-compress learns much earlier at the same budget.

## What "LSTM Memory Starts To Work" Means

Use two levels of onset. Do not call the LSTM viable just because loss decreases.

Optimization onset means:

- train loss and val loss decrease across milestones without numerical instability;
- val token accuracy rises above the weak scaffold/random region, using `0.30` as the first meaningful line and `0.40` as stronger evidence;
- serial-position accuracy improves at positions 0 and 1, not just EOS or predicted length;
- EOS and predicted-length accuracy remain interpretable, rather than masking repeated-block failure.

Useful sequence-memory onset means at least three of these hold at a milestone and do not immediately reverse at the next milestone:

- val token accuracy `>= 0.45`;
- val length-2 exact accuracy `>= 0.60`;
- val length-3 exact accuracy is nonzero, preferably `>= 0.20`;
- test length-2 exact accuracy is nonzero and tracks val, not just one val accident;
- duplicate sequence rate falls below `0.75` and mean unique predicted blocks increases;
- final-memory order probe on val/test rises above `0.50`;
- known-length final-memory exact rises above `0.20`;
- serial-position accuracy for positions 0 and 1 is at least `0.60`;
- set-overlap improves while duplicate count decreases.

Strong onset means LSTM reaches useful sequence-memory onset and approaches slot-compress at the same milestone, for example within roughly `0.10` absolute token accuracy or with comparable length 2-3 exact accuracy.

## Training Schedule

Use one continuous LSTM trajectory with staged stops, not independent fresh runs:

1. Train to epoch 10.
2. Snapshot latest and best checkpoints.
3. Evaluate milestone checkpoints.
4. Resume the same run to epoch 20.
5. Repeat for epochs 40 and 80.
6. Extend to 120 and then 160 only if the 80-epoch result shows learning signal or plausible delayed onset.

This gives the preferred "one long-cap run" while preserving actual milestone checkpoints. The current trainer only retains `latest.pt`, `best.pt`, `best_full_sequence.pt`, `best_token.pt`, and `best_val_loss.pt`; therefore each milestone stop must copy those files before resuming.

Planned milestones:

| Milestone | Purpose |
|---:|---|
| 10 | smoke learning signal |
| 20 | early onset check |
| 40 | medium check |
| 80 | comparable to current Stage 2 baseline |
| 120 | long-run continuation if epoch 80 still improves |
| 160 | final delayed-onset check if epoch 120 improves |

Stop early before epoch 80 only for a hard failure: non-finite loss, corrupt checkpoints, evaluator failure that cannot be fixed without changing the experiment, or completely flat loss/behavior through epoch 40. Based on the existing LSTM history, the default expectation is to run through epoch 80 once approved.

## Slot-Compress Control

For the first batch, reuse existing slot-compress D_mem=64 evidence for training-history comparison because matching epoch 10/20/40/80 validation history already exists in:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624/summary.json`

Do a new staged slot-compress control only if one of these becomes true:

- LSTM shows useful sequence-memory onset and needs a fully matched retained-checkpoint comparison;
- the report requires test/probe metrics at every slot-compress milestone, which the old history cannot provide;
- any code/evaluator changes before execution make old slot-compress metrics non-comparable.

If rerun, name it:

`stage2_seed0_slot_compress_dmem64_onset_control_20260629`

## Checkpoint Selection At Each Milestone

For each memory mode and milestone, evaluate:

- `latest.pt` copied at that milestone;
- `best_full_sequence.pt` copied at that milestone;
- `best_token.pt` copied at that milestone;
- `best_val_loss.pt` copied at that milestone;
- `best.pt` only as the primary alias for `best_full_sequence.pt`;
- final-memory probe checkpoint selection if the diagnostic supports it.

For Stage 2, current primary checkpoint selection maximizes:

`(val_full_sequence_accuracy, val_token_accuracy, val_predicted_length_accuracy, -val_loss)`

Keep this rule unchanged.

## Required Metrics At Every Milestone

Record these for both val and test where applicable:

- train loss;
- val loss;
- total loss components if available;
- full-sequence accuracy;
- token accuracy;
- per-length exact accuracy for lengths 2-9;
- per-length token accuracy;
- serial-position accuracy;
- duplicate sequence rate;
- duplicate count;
- mean unique predicted blocks;
- set-overlap metrics;
- EOS accuracy;
- predicted-length accuracy;
- final-memory order probe if available;
- final-memory known-length exact probe if available;
- final-memory length probe if available.

The current `evaluate.py` path can write autonomous recall metrics, per-length token metrics, duplicate metrics, set-overlap metrics, and loss components for newly evaluated checkpoints. The existing final-memory probe metrics live in `*_minimal_localization.json`, but I do not see a stable repo CLI for regenerating those files. Before execution, either expose that diagnostic as a repeatable command or explicitly mark final-memory probe metrics as unavailable for new milestones.

## Decision Rules

Outcome A: LSTM learns early and catches up.  
Criteria: useful sequence-memory onset by epoch 40 or 80, length 2-3 exact recall is stable on val/test, duplicate rate falls, and LSTM is close to slot-compress at equal budget. Interpretation: LSTM is viable; prioritize longer LSTM runs and then add seeds.

Outcome B: LSTM learns slowly but steadily.  
Criteria: loss, token accuracy, serial-position accuracy, duplicate rate, or final-memory probe improve through epoch 80, but LSTM remains materially below slot-compress. Interpretation: LSTM may be viable but needs a longer schedule or curriculum; extend to 120/160 before rejecting it.

Outcome C: LSTM loss improves but behavior does not.  
Criteria: train/val loss improves, but token accuracy, length 2-3 exact, duplicate rate, set-overlap, and serial-position accuracy stay flat. Interpretation: the objective may be learning shortcuts or weak auxiliaries; inspect loss/output mismatch before more long runs.

Outcome D: LSTM final-memory probe improves but autonomous recall fails.  
Criteria: final-memory order probe or known-length exact rises, while autonomous exact remains near zero and duplicate collapse remains high. Interpretation: memory encoding is working; recall readout is the bottleneck.

Outcome E: LSTM final-memory probe does not improve.  
Criteria: final-memory order probe remains near current held-out levels and autonomous recall remains weak. Interpretation: memory write/storage is the bottleneck.

Outcome F: slot-compress works much earlier and LSTM never shows onset.  
Criteria: slot-compress remains much better at matched milestones and LSTM has no useful sequence-memory onset by 160. Interpretation: slot-compress is currently the better mechanism; LSTM needs architecture or training intervention before it can be the final path.

## Artifact Naming

LSTM run:

`stage2_seed0_lstm_memory_dmem64_onset_20260629`

LSTM run directory:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629`

LSTM milestone directory pattern:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_010`

Evaluation file pattern:

`corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629_epoch010_latest_val_eval.json`

Slot-compress control, only if needed:

`stage2_seed0_slot_compress_dmem64_onset_control_20260629`

## Minimal Execution Batch

Smallest first run after approval:

- LSTM memory, D_mem=64, K=12, seed 0;
- staged caps at 10, 20, 40, and 80 epochs;
- snapshot and evaluate milestone checkpoints at each stop;
- reuse old slot-compress D_mem=64 history/control unless the LSTM shows onset or fully matched test/probe milestones are required.

Do not run the 120/160 continuation until the epoch-80 milestone is reviewed.

## Proposed First Execution Commands

These commands intentionally do not run in this planning thread.

First training stop, epoch 10:

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_dmem64.json \
  --stage 2 \
  --device auto \
  --seed 0 \
  --run-name stage2_seed0_lstm_memory_dmem64_onset_20260629 \
  --output-root corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12 \
  --max-epochs 10 \
  --allow-full-training \
  --warm-start-stage1-checkpoint corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage1_seed0_full_20260624/best.pt
```

Snapshot the epoch-10 retained checkpoints before resuming:

```bash
mkdir -p corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_010
cp corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/latest.pt corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_010/latest.pt
cp corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/best_full_sequence.pt corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_010/best_full_sequence.pt
cp corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/best_token.pt corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_010/best_token.pt
cp corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/best_val_loss.pt corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_010/best_val_loss.pt
cp corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/summary.json corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_010/summary.json
```

Evaluate the epoch-10 latest checkpoint:

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.evaluate \
  --config corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_dmem64.json \
  --checkpoint corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_010/latest.pt \
  --split val \
  --batch-size 16 \
  --device auto \
  --output corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629_epoch010_latest_val_eval.json
```

Repeat the same evaluator command for `--split test`, and for `best_full_sequence.pt`, `best_token.pt`, and `best_val_loss.pt`. Then resume the same run with `--resume --max-epochs 20`, and repeat the snapshot/evaluation cycle for epoch 20, 40, and 80.

Resume command pattern:

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_dmem64.json \
  --stage 2 \
  --device auto \
  --seed 0 \
  --run-name stage2_seed0_lstm_memory_dmem64_onset_20260629 \
  --output-root corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12 \
  --max-epochs 20 \
  --resume \
  --allow-full-training \
  --warm-start-stage1-checkpoint corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage1_seed0_full_20260624/best.pt
```

Change only `--max-epochs` to `40` and then `80` for later staged stops.

## Expected Artifact Paths

Primary run artifacts:

- `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/summary.json`
- `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/latest.pt`
- `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/best_full_sequence.pt`
- `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/best_token.pt`
- `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/best_val_loss.pt`

Milestone copies:

- `.../stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_010/*.pt`
- `.../stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_020/*.pt`
- `.../stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_040/*.pt`
- `.../stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_080/*.pt`

Evaluation outputs:

- `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629_epoch010_latest_val_eval.json`
- same pattern for `test`, `best_full_sequence`, `best_token`, and `best_val_loss`;
- same pattern for `epoch020`, `epoch040`, and `epoch080`.

## Estimated Runtime

Existing D_mem=64 Stage 2 runs appear to take about 19 minutes for 80 epochs on this machine, based on checkpoint timestamps. Estimate:

- epoch 10 stop: about 2-4 minutes;
- staged training through epoch 80: about 20-30 minutes;
- core val/test evaluation across milestone checkpoints: additional tens of minutes depending on how many checkpoint/split combinations are evaluated;
- final-memory probe diagnostics, if exposed, should be budgeted separately.

If the 80-epoch LSTM run qualifies for continuation, each extra 40 epochs should be roughly another 10-15 minutes of training on the same hardware.

## Stop / Continue Criteria

Continue from 10 to 20 unless there is a hard execution failure.

Continue from 20 to 40 if either train loss or val token accuracy improves, or if serial-position accuracy starts to move.

Continue from 40 to 80 unless all of these are true: val token accuracy remains below `0.30`, length-2 exact is still zero, duplicate rate is not improving, and train/val loss are flat.

At epoch 80:

- extend to 120 if val token is still rising, length-2 exact appears, duplicate rate is decreasing, or final-memory probe improves;
- extend to 160 only if epoch 120 improves on at least one behavioral or probe metric;
- stop the LSTM lane if loss improves but all autonomous/probe behavior is flat, and classify as Outcome C or E;
- run a fresh staged slot-compress control only if LSTM shows useful onset or if existing slot-compress artifacts are not sufficient for the comparison table.

## What Not To Run Yet

Do not run:

- K=0, K=6, K=20, or K=30;
- D_mem=16, D_mem=32, or D_mem=128;
- Stage 1 epoch ablations;
- no-pretrain/warm-start ablations;
- loss ablations;
- architecture changes;
- new canonical K datasets;
- commits or pushes.

## Approval Text To Start

Send this exact approval if the next action should start the run:

`Approved: start the Priority 2 LSTM memory onset run only, with K=12, D_mem=64, seed=0, staged milestones 10/20/40/80, current Stage 1 warm start, current default Stage 2 loss, no new datasets, no architecture changes, no commits, and no pushes.`
