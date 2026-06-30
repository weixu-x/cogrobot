# V2 LSTM No-Training Diagnostics - 2026-06-29

Scope: frozen main-model diagnostics for `stage2_seed0_lstm_memory_dmem64_onset_20260629`.
The final-memory probes train only linear readouts on frozen `final_memory` features; no main model weights are updated.

## Artifacts

- JSON: `reports/v2_lstm_no_training_diagnostics_20260629.json`
- Report: `reports/v2_lstm_no_training_diagnostics_20260629.md`
- Config: `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_dmem64.json`
- Run root: `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629`

## P0.1 Final-Memory Probe (test split)

| Checkpoint | source epoch | order token | known-L exact | length |
| --- | --- | --- | --- | --- |
| epoch080_best_full | 79 | 0.518 | 0.250 | 0.575 |
| epoch093_best_val_loss | 93 | 0.523 | 0.225 | 0.675 |
| epoch120_best_full | 115 | 0.536 | 0.225 | 0.650 |
| epoch122_best_token | 122 | 0.514 | 0.225 | 0.650 |
| epoch139_best_full | 139 | 0.555 | 0.200 | 0.575 |
| epoch160_latest | 159 | 0.573 | 0.225 | 0.550 |
| epoch200_latest | 199 | 0.573 | 0.350 | 0.550 |

Per-position final-memory order probe accuracy on test:

| Checkpoint | p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epoch080_best_full | 1.000 | 0.775 | 0.371 | 0.267 | 0.400 | 0.050 | 0.333 | 0.300 | 0.600 |
| epoch093_best_val_loss | 1.000 | 0.800 | 0.400 | 0.333 | 0.240 | 0.100 | 0.267 | 0.300 | 0.800 |
| epoch120_best_full | 1.000 | 0.825 | 0.457 | 0.333 | 0.240 | 0.150 | 0.200 | 0.300 | 0.800 |
| epoch122_best_token | 1.000 | 0.775 | 0.429 | 0.300 | 0.160 | 0.150 | 0.200 | 0.400 | 0.800 |
| epoch139_best_full | 1.000 | 0.900 | 0.429 | 0.400 | 0.240 | 0.200 | 0.133 | 0.300 | 0.800 |
| epoch160_latest | 1.000 | 0.800 | 0.600 | 0.367 | 0.320 | 0.150 | 0.200 | 0.400 | 0.800 |
| epoch200_latest | 0.975 | 0.825 | 0.714 | 0.300 | 0.320 | 0.200 | 0.133 | 0.400 | 0.400 |

## P0.2 No-Repeat Constrained Decoding (test split)

| Checkpoint | normal full | normal token | normal dup | no-repeat full | no-repeat token | no-repeat dup | full delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| epoch080_best_full | 0.250 | 0.635 | 0.725 | 0.300 | 0.581 | 0.000 | 0.050 |
| epoch093_best_val_loss | 0.275 | 0.623 | 0.675 | 0.350 | 0.600 | 0.000 | 0.075 |
| epoch120_best_full | 0.350 | 0.654 | 0.650 | 0.375 | 0.631 | 0.000 | 0.025 |
| epoch122_best_token | 0.350 | 0.677 | 0.650 | 0.400 | 0.654 | 0.000 | 0.050 |
| epoch139_best_full | 0.350 | 0.692 | 0.650 | 0.400 | 0.688 | 0.000 | 0.050 |
| epoch160_latest | 0.350 | 0.673 | 0.625 | 0.400 | 0.642 | 0.000 | 0.050 |
| epoch200_latest | 0.350 | 0.696 | 0.625 | 0.425 | 0.692 | 0.000 | 0.075 |

## P0.3 Oracle Length / EOS (test split)

| Checkpoint | normal full | oracle EOS full | oracle length full | no-repeat+oracle length full | oracle length len-acc | no-repeat+oracle dup |
| --- | --- | --- | --- | --- | --- | --- |
| epoch080_best_full | 0.250 | 0.250 | 0.250 | 0.325 | 1.000 | 0.000 |
| epoch093_best_val_loss | 0.275 | 0.275 | 0.275 | 0.350 | 1.000 | 0.000 |
| epoch120_best_full | 0.350 | 0.350 | 0.350 | 0.375 | 1.000 | 0.000 |
| epoch122_best_token | 0.350 | 0.350 | 0.350 | 0.400 | 1.000 | 0.000 |
| epoch139_best_full | 0.350 | 0.350 | 0.350 | 0.400 | 1.000 | 0.000 |
| epoch160_latest | 0.350 | 0.350 | 0.350 | 0.400 | 1.000 | 0.000 |
| epoch200_latest | 0.350 | 0.350 | 0.350 | 0.425 | 1.000 | 0.000 |

## Interpretation

For the selected epoch-139 best-full checkpoint, final-memory order token accuracy is 0.555 and known-length exact is 0.200 on test. This is in the 0.5-0.6 band, so held-out ordered identity is still weak in the LSTM final memory; memory write/storage remains a bottleneck.
No-repeat decoding gives epoch-139 test full 0.400 versus normal 0.350. That is a real but modest upper-bound gain, so duplicate policy is part of the failure, but not the whole failure.
Oracle length alone gives epoch-139 test full 0.350; no-repeat plus oracle length gives 0.400. Length/EOS correction alone does not improve exact recall here; the limiting errors are content/order and repeat policy, with final-memory storage still not clean by the probe.

Note: `oracle_eos` sets EOS at the target EOS index only; `oracle_length` decodes exactly L block steps from block logits and then appends EOS.
