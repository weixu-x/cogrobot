# V2 LSTM Binding Diagnostics - 2026-06-29

Scope: frozen probes for checkpoint epoch 139. Only probe heads are trained.

## Artifacts

- JSON: `reports/v2_lstm_phase1_binding_diagnostics_20260629.json`
- Report: `reports/v2_lstm_phase1_binding_diagnostics_20260629.md`
- Config: `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_dmem64.json`
- Checkpoint: `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_160/best_full_sequence.pt`

## 1.1 Write-Step Linear Probes

| state | current item acc | write position acc | states |
| --- | --- | --- | --- |
| h_t | 0.982 | 0.914 | 220 |
| c_t | 0.973 | 0.964 | 220 |
| [h_t;c_t] | 0.982 | 0.959 | 220 |

## 1.2 Short-Sequence Behavior

| length | count | exact | token | duplicate |
| --- | --- | --- | --- | --- |
| 2 | 5 | 1.000 | 1.000 | 0.000 |
| 3 | 5 | 1.000 | 1.000 | 0.000 |
| 4 | 5 | 0.600 | 0.920 | 0.400 |
| 5 | 5 | 0.000 | 0.767 | 1.000 |
| 6 | 5 | 0.200 | 0.543 | 0.800 |
| 7 | 5 | 0.000 | 0.650 | 1.000 |
| 8 | 5 | 0.000 | 0.689 | 1.000 |
| 9 | 5 | 0.000 | 0.460 | 1.000 |

## 3 Behavior Sanity Check

| first | middle | last | edge | U-score |
| --- | --- | --- | --- | --- |
| 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

| adjacent | far | adjacent fraction | distance-dependent | distance counts |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0.000 | False | {} |

## 1.1/1.2 Final-State Probes

| state | order token | known-L exact | length acc | L2 exact | L3 exact |
| --- | --- | --- | --- | --- | --- |
| final h | 0.541 | 0.200 | 0.550 | 1.000 | 0.400 |
| final c | 0.586 | 0.325 | 0.700 | 1.000 | 0.600 |
| final [h;c] | 0.595 | 0.325 | 0.675 | 1.000 | 0.600 |

## 1.3 Identity-Frozen Control

All item embeddings are replaced by the train-set mean item embedding before memory write-in.

| state | write position acc | final length acc | states |
| --- | --- | --- | --- |
| h_t | 1.000 | 1.000 | 220 |
| c_t | 1.000 | 1.000 | 220 |
| [h_t;c_t] | 1.000 | 1.000 | 220 |

## Decision Readout

Trajectory [h_t;c_t] test probes: current item 0.982, write position 0.959.
Final [h;c] test probes: ordered identity token 0.595, length 0.675.
Short-sequence behavior: test length-2 exact 1.000, length-3 exact 1.000.
Identity-frozen control: [h_t;c_t] write-position probe 1.000.
Decision: current item and step are present in the write trajectory, and short sequences are solved. This does not match a catastrophic item-write failure or a pure capacity squeeze.
The main bottleneck is after per-step writing: ordered identity is much stronger in trajectory states than in the final compressed state.
The constant-identity control can still encode write position, so position dynamics are not broken by themselves.
