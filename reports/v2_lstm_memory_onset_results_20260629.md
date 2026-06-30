# V2 LSTM Memory Onset Results

Date: 2026-06-29  
Run: `stage2_seed0_lstm_memory_dmem64_onset_20260629`  
Status: completed approved continuation through epoch 200. No dataset, architecture, loss-weight, commit, or push changes were made.

## Artifacts

Run root:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629`

Milestone checkpoint snapshots:

- `milestones/epoch_010`
- `milestones/epoch_020`
- `milestones/epoch_040`
- `milestones/epoch_080`
- `milestones/epoch_120`
- `milestones/epoch_160`
- `milestones/epoch_200`

Evaluation JSONs:

`corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629_epoch*_eval.json`

Total retained checkpoint files: `35`  
Total eval JSON files: `56`

## Aggregate Results

Best-full checkpoint at each milestone:

| Milestone | Selected epoch | Val full | Val token | Val EOS | Val length | Val dup | Test full | Test token | Test EOS | Test length | Test dup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 79 | 0.275 | 0.642 | 1.000 | 0.925 | 0.625 | 0.250 | 0.635 | 0.925 | 0.875 | 0.725 |
| 120 | 115 | 0.275 | 0.658 | 0.975 | 0.950 | 0.675 | 0.350 | 0.654 | 0.800 | 0.725 | 0.650 |
| 160 | 139 | 0.300 | 0.654 | 0.975 | 0.875 | 0.625 | 0.350 | 0.692 | 0.925 | 0.850 | 0.650 |
| 200 | 139 | 0.300 | 0.654 | 0.975 | 0.875 | 0.625 | 0.350 | 0.692 | 0.925 | 0.850 | 0.650 |

Latest checkpoint at each long milestone:

| Milestone | Latest epoch | Val full | Val token | Val loss | Val dup | Test full | Test token | Test loss | Test dup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 79 | 0.275 | 0.642 | 1.417 | 0.625 | 0.250 | 0.635 | 1.344 | 0.725 |
| 120 | 119 | 0.250 | 0.638 | 1.430 | 0.700 | 0.300 | 0.681 | 1.263 | 0.675 |
| 160 | 159 | 0.275 | 0.650 | 1.675 | 0.700 | 0.350 | 0.673 | 1.351 | 0.625 |
| 200 | 199 | 0.225 | 0.631 | 1.935 | 0.675 | 0.350 | 0.696 | 1.410 | 0.625 |

## Best Checkpoint

The best full-sequence checkpoint after all continuations is:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_160/best_full_sequence.pt`

It was selected at training epoch `139`, not at epoch 200.

Metrics:

| Split | Full | Token | EOS | Pred length | Duplicate rate | Loss |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.300 | 0.654 | 0.975 | 0.875 | 0.625 | 1.529 |
| test | 0.350 | 0.692 | 0.925 | 0.850 | 0.650 | 1.284 |

Best-token checkpoint is epoch `122`: val full/token `0.275/0.662`, test full/token `0.350/0.677`. Best-val-loss checkpoint is epoch `93`: val full/token `0.250/0.615`, test full/token `0.275/0.623`.

## Per-Length Behavior

Best-full epoch 139 per-length exact/token:

| Length | Val exact | Val token | Val dup | Test exact | Test token | Test dup |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| 3 | 0.800 | 0.950 | 0.000 | 1.000 | 1.000 | 0.000 |
| 4 | 0.400 | 0.800 | 0.600 | 0.600 | 0.920 | 0.400 |
| 5 | 0.000 | 0.700 | 0.800 | 0.000 | 0.770 | 1.000 |
| 6 | 0.200 | 0.770 | 0.800 | 0.200 | 0.540 | 0.800 |
| 7 | 0.000 | 0.620 | 1.000 | 0.000 | 0.650 | 1.000 |
| 8 | 0.000 | 0.510 | 0.800 | 0.000 | 0.690 | 1.000 |
| 9 | 0.000 | 0.400 | 1.000 | 0.000 | 0.460 | 1.000 |

## Interpretation

The LSTM memory clearly starts working by epoch 80 and continues improving through roughly epoch 139. It catches up to, and on test full/token surpasses, the existing slot-compress D_mem=64 best-full checkpoint:

| Model | Val full | Val token | Val dup | Test full | Test token | Test dup |
|---|---:|---:|---:|---:|---:|---:|
| slot-compress D_mem=64 current best-full | 0.325 | 0.627 | 0.525 | 0.300 | 0.631 | 0.575 |
| LSTM D_mem=64 onset best-full | 0.300 | 0.654 | 0.625 | 0.350 | 0.692 | 0.650 |

The tradeoff is that LSTM still has higher duplicate-collapse rates than slot-compress. Longer training after epoch 160 does not improve validation full-sequence selection: epoch 200 latest has lower train loss but worse val loss and lower val full accuracy. This is optimization/overfit behavior, not a reason to reject LSTM memory.

Outcome classification: between Outcome A and Outcome B. LSTM is viable and catches up with enough budget, but it remains duplicate-limited and should not be judged from short 80-epoch evidence alone.

Recommended next step: run seed replication for LSTM D_mem=64 around a 160-epoch cap, and expose/regenerate final-memory probe diagnostics for the onset run before changing architecture.
