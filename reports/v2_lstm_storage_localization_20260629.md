# V2 LSTM Storage Localization - 2026-06-29

Scope: frozen linear probes over presentation item embeddings and LSTM memory states for `stage2_seed0_lstm_memory_dmem64_onset_20260629`.
Only probe heads are trained on frozen features; no checkpoint model weights are updated.

## Artifacts

- JSON: `reports/v2_lstm_storage_localization_20260629.json`
- Report: `reports/v2_lstm_storage_localization_20260629.md`
- Config: `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_dmem64.json`
- Run root: `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629`

## P1.1 Item Embedding Probe (test split)

| Checkpoint | source epoch | item block acc | items |
| --- | --- | --- | --- |
| epoch080_best_full | 79 | 1.000 | 220 |
| epoch120_best_full | 115 | 1.000 | 220 |
| epoch139_best_full | 139 | 1.000 | 220 |
| epoch160_latest | 159 | 1.000 | 220 |
| epoch200_latest | 199 | 1.000 | 220 |
| epoch122_best_token | 122 | 1.000 | 220 |
| epoch093_best_val_loss | 93 | 1.000 | 220 |

## P1.2 Memory Write Trajectory Probe (test split, M_t = [h_t;c_t])

| Checkpoint | prefix token | prefix exact | current item | past-set exact | past-set recall |
| --- | --- | --- | --- | --- | --- |
| epoch080_best_full | 0.701 | 0.545 | 0.995 | 1.000 | 1.000 |
| epoch120_best_full | 0.732 | 0.577 | 0.982 | 1.000 | 1.000 |
| epoch139_best_full | 0.743 | 0.591 | 0.982 | 0.995 | 0.999 |
| epoch160_latest | 0.752 | 0.605 | 0.986 | 0.995 | 0.999 |
| epoch200_latest | 0.752 | 0.627 | 0.973 | 1.000 | 1.000 |
| epoch122_best_token | 0.733 | 0.582 | 0.986 | 1.000 | 1.000 |
| epoch093_best_val_loss | 0.729 | 0.582 | 0.995 | 1.000 | 1.000 |

## P1.3 Final h / c / [h;c] Probe (test split)

| Checkpoint | state | order token | known-L exact | length |
| --- | --- | --- | --- | --- |
| epoch080_best_full | h | 0.523 | 0.250 | 0.550 |
| epoch080_best_full | c | 0.545 | 0.300 | 0.550 |
| epoch080_best_full | [h;c] | 0.545 | 0.275 | 0.600 |
| epoch120_best_full | h | 0.532 | 0.225 | 0.650 |
| epoch120_best_full | c | 0.545 | 0.300 | 0.625 |
| epoch120_best_full | [h;c] | 0.568 | 0.300 | 0.625 |
| epoch139_best_full | h | 0.545 | 0.200 | 0.550 |
| epoch139_best_full | c | 0.573 | 0.300 | 0.725 |
| epoch139_best_full | [h;c] | 0.582 | 0.325 | 0.625 |
| epoch160_latest | h | 0.595 | 0.325 | 0.575 |
| epoch160_latest | c | 0.614 | 0.300 | 0.750 |
| epoch160_latest | [h;c] | 0.609 | 0.350 | 0.625 |
| epoch200_latest | h | 0.586 | 0.375 | 0.500 |
| epoch200_latest | c | 0.627 | 0.350 | 0.700 |
| epoch200_latest | [h;c] | 0.609 | 0.375 | 0.625 |
| epoch122_best_token | h | 0.541 | 0.250 | 0.675 |
| epoch122_best_token | c | 0.545 | 0.300 | 0.625 |
| epoch122_best_token | [h;c] | 0.555 | 0.325 | 0.625 |
| epoch093_best_val_loss | h | 0.536 | 0.225 | 0.650 |
| epoch093_best_val_loss | c | 0.559 | 0.325 | 0.625 |
| epoch093_best_val_loss | [h;c] | 0.564 | 0.300 | 0.625 |

## P1.4 Flattened Memory-State Probe (test split)

| Checkpoint | state | order token | known-L exact | length |
| --- | --- | --- | --- | --- |
| epoch080_best_full | all_h | 0.791 | 0.600 | 1.000 |
| epoch080_best_full | all_c | 0.727 | 0.450 | 1.000 |
| epoch080_best_full | all_[h;c] | 0.745 | 0.450 | 1.000 |
| epoch120_best_full | all_h | 0.768 | 0.525 | 1.000 |
| epoch120_best_full | all_c | 0.718 | 0.525 | 1.000 |
| epoch120_best_full | all_[h;c] | 0.723 | 0.475 | 1.000 |
| epoch139_best_full | all_h | 0.764 | 0.500 | 1.000 |
| epoch139_best_full | all_c | 0.741 | 0.525 | 1.000 |
| epoch139_best_full | all_[h;c] | 0.723 | 0.450 | 1.000 |
| epoch160_latest | all_h | 0.750 | 0.475 | 1.000 |
| epoch160_latest | all_c | 0.695 | 0.475 | 1.000 |
| epoch160_latest | all_[h;c] | 0.736 | 0.475 | 1.000 |
| epoch200_latest | all_h | 0.723 | 0.500 | 1.000 |
| epoch200_latest | all_c | 0.736 | 0.500 | 1.000 |
| epoch200_latest | all_[h;c] | 0.695 | 0.425 | 1.000 |
| epoch122_best_token | all_h | 0.764 | 0.500 | 1.000 |
| epoch122_best_token | all_c | 0.736 | 0.500 | 1.000 |
| epoch122_best_token | all_[h;c] | 0.705 | 0.400 | 1.000 |
| epoch093_best_val_loss | all_h | 0.777 | 0.575 | 1.000 |
| epoch093_best_val_loss | all_c | 0.718 | 0.500 | 1.000 |
| epoch093_best_val_loss | all_[h;c] | 0.745 | 0.500 | 1.000 |

## Decision Tree Readout

For epoch-139 best-full, item embedding probe is 1.000, so presentation/segment grounding is not the bottleneck.
The memory trajectory is only partial: prefix token 0.743, prefix exact 0.591, current item 0.982, past-set exact 0.995.
Final c is stronger than final h on order token (0.573 vs 0.545), and [h;c] is 0.582. This means the LSTM cell state exposes more serial identity than h, but not enough to call storage solved.
Flattened all_[h;c] order token is 0.723. If this is materially above final [h;c], information exists in the trajectory more than in the final state, supporting attention over memory states.
