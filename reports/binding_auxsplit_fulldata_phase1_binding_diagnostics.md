# V2 LSTM Binding Diagnostics - 2026-06-29

Scope: frozen probes for checkpoint epoch 69. Only probe heads are trained.

## Artifacts

- JSON: `reports/binding_auxsplit_fulldata_phase1_binding_diagnostics.json`
- Report: `reports/binding_auxsplit_fulldata_phase1_binding_diagnostics.md`
- Config: `corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_item_context_binding_auxsplit_dmem64.json`
- Checkpoint: `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_item_context_binding_auxsplit_dmem64_20260629/best_full_sequence.pt`

## 1.1 Write-Step Linear Probes

| state | current item acc | write position acc | states |
| --- | --- | --- | --- |
| h_t | 0.864 | 0.905 | 220 |
| c_t | 0.118 | 0.182 | 220 |
| [h_t;c_t] | 0.859 | 0.914 | 220 |

## 1.2 Short-Sequence Behavior

| length | count | exact | token | duplicate |
| --- | --- | --- | --- | --- |
| 2 | 5 | 1.000 | 1.000 | 0.000 |
| 3 | 5 | 0.800 | 0.900 | 0.200 |
| 4 | 5 | 0.600 | 0.920 | 0.200 |
| 5 | 5 | 0.000 | 0.667 | 0.400 |
| 6 | 5 | 0.000 | 0.543 | 0.600 |
| 7 | 5 | 0.000 | 0.750 | 1.000 |
| 8 | 5 | 0.200 | 0.689 | 0.800 |
| 9 | 5 | 0.000 | 0.540 | 1.000 |

## 3 Behavior Sanity Check

| first | middle | last | edge | U-score |
| --- | --- | --- | --- | --- |
| 0.900 | 0.586 | 0.675 | 0.787 | 0.202 |

| adjacent | far | adjacent fraction | distance-dependent | distance counts |
| --- | --- | --- | --- | --- |
| 36 | 21 | 0.632 | True | {"1": 36, "2": 8, "3": 3, "4": 2, "5": 5, "6": 2, "7": 1} |

## 1.1/1.2 Final-State Probes

| state | order token | known-L exact | length acc | L2 exact | L3 exact |
| --- | --- | --- | --- | --- | --- |
| final h | 0.886 | 0.725 | 0.650 | 1.000 | 1.000 |
| final c | 0.127 | 0.000 | 0.125 | 0.000 | 0.000 |
| final [h;c] | 0.868 | 0.675 | 0.600 | 1.000 | 1.000 |

## 1.3 Identity-Frozen Control

All item embeddings are replaced by the train-set mean item embedding before memory write-in.

| state | write position acc | final length acc | states |
| --- | --- | --- | --- |
| h_t | 1.000 | 1.000 | 220 |
| c_t | 0.182 | 0.125 | 220 |
| [h_t;c_t] | 1.000 | 1.000 | 220 |

## Decision Readout

Trajectory [h_t;c_t] test probes: current item 0.859, write position 0.914.
Final [h;c] test probes: ordered identity token 0.868, length 0.600.
Short-sequence behavior: test length-2 exact 1.000, length-3 exact 0.800.
Identity-frozen control: [h_t;c_t] write-position probe 1.000.
Decision: probes show mixed evidence. Treat the problem as write/readout binding plus optimization until a capacity sweep shows a clear length cliff that shifts with D_mem.
The constant-identity control can still encode write position, so position dynamics are not broken by themselves.
