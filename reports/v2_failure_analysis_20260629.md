# V2 Failure Analysis - 2026-06-29

## Evidence Source

This report uses artifacts visible in the main checkout, not the earlier isolated `/tmp` worktree.

Primary checkpoint:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624/best_full_sequence.pt`

Primary eval files:

- `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_val_eval.json`
- `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_test_eval.json`

## Aggregate Metrics

| Split | Epoch | Full seq | Token | EOS | Pred length | Errors S/O/I/T |
|---|---:|---:|---:|---:|---:|---|
| Val | 63 | 0.325 | 0.627 | 0.875 | 0.875 | 61 / 8 / 14 / 6 |
| Test | 63 | 0.300 | 0.631 | 0.900 | 0.850 | 58 / 12 / 14 / 3 |

`S/O/I/T` = substitution / omission / insertion / transposition counts.

## Per-Length Metrics

### Val, best-full checkpoint

| Length | Count | Exact | Token | EOS | Pred length | Dup rate | Mean dup count | Mean unique | Set Jaccard |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 2.000 | 1.000 |
| 3 | 5 | 0.800 | 0.950 | 1.000 | 1.000 | 0.000 | 0.000 | 3.000 | 0.900 |
| 4 | 5 | 0.400 | 0.760 | 0.800 | 0.800 | 0.200 | 0.200 | 4.000 | 0.653 |
| 5 | 5 | 0.200 | 0.700 | 0.800 | 0.800 | 0.400 | 0.400 | 4.800 | 0.681 |
| 6 | 5 | 0.200 | 0.743 | 0.800 | 0.800 | 0.600 | 1.000 | 5.200 | 0.757 |
| 7 | 5 | 0.000 | 0.575 | 0.800 | 0.800 | 1.000 | 1.400 | 5.800 | 0.689 |
| 8 | 5 | 0.000 | 0.556 | 0.800 | 0.800 | 1.000 | 2.800 | 5.600 | 0.661 |
| 9 | 5 | 0.000 | 0.300 | 1.000 | 1.000 | 1.000 | 2.600 | 6.400 | 0.711 |

### Test, best-full checkpoint

| Length | Count | Exact | Token | EOS | Pred length | Dup rate | Mean dup count | Mean unique | Set Jaccard |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 2.000 | 1.000 |
| 3 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 3.000 | 1.000 |
| 4 | 5 | 0.400 | 0.840 | 0.800 | 0.800 | 0.200 | 0.200 | 4.000 | 0.760 |
| 5 | 5 | 0.000 | 0.700 | 0.800 | 0.800 | 0.800 | 1.200 | 4.000 | 0.613 |
| 6 | 5 | 0.000 | 0.629 | 1.000 | 1.000 | 0.800 | 1.000 | 5.000 | 0.512 |
| 7 | 5 | 0.000 | 0.575 | 0.800 | 0.600 | 0.800 | 1.000 | 6.000 | 0.718 |
| 8 | 5 | 0.000 | 0.444 | 1.000 | 1.000 | 1.000 | 1.600 | 6.400 | 0.719 |
| 9 | 5 | 0.000 | 0.440 | 0.800 | 0.600 | 1.000 | 2.400 | 6.600 | 0.733 |

## Serial-Position Token Accuracy

| Position | Val | Test |
|---:|---:|---:|
| 0 | 0.850 | 0.775 |
| 1 | 0.850 | 0.925 |
| 2 | 0.629 | 0.629 |
| 3 | 0.433 | 0.433 |
| 4 | 0.440 | 0.520 |
| 5 | 0.250 | 0.350 |
| 6 | 0.333 | 0.133 |
| 7 | 0.400 | 0.200 |
| 8 | 0.000 | 0.200 |

## Failure Pattern

The model solves length 2 and most length 3 sequences. Performance drops sharply from length 4 onward and exact accuracy is zero for test lengths 5-9. Token accuracy degrades more gradually, which means the model often recovers part of the sequence but fails exact ordered recall.

The most visible long-sequence failure is duplicate collapse:

- val duplicate rate reaches `1.0` at lengths 7-9;
- test duplicate rate reaches `0.8` at lengths 5-7 and `1.0` at lengths 8-9;
- predicted sets still overlap targets moderately, so the model often knows many involved blocks but repeats or misorders them.

EOS and predicted-length accuracy are not the sole bottleneck. Many long examples have correct EOS/length but still fail exact order because of substitutions, duplicate repeats, and serial-position degradation.

## Evaluator Instrumentation Added

The evaluator/analysis path now exposes:

- loss components when evaluating newly with the current code;
- `per_length_token_accuracy`;
- `per_length_metrics`;
- `duplicate_metrics`;
- `set_overlap_metrics`;
- row-level `target_blocks`, `predicted_blocks`, `duplicate_count`, `set_overlap_jaccard`, and `failure_flags`.

Existing keys such as `per_length_accuracy`, `serial_position_accuracy`, `error_taxonomy`, and `rows` remain available.
