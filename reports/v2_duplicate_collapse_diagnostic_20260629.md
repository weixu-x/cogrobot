# V2 Duplicate Collapse / Recall Readout Diagnostic - 2026-06-29

## Answer

The current evidence points more to autonomous recall readout failure than memory storage failure.

The slot-compress `D_mem=64` model has strong final-memory ordered-identity probes, but the autonomous decoder still repeats blocks and collapses on long sequences.

## Final-Memory Evidence

From `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_minimal_localization.json`:

| Checkpoint | Val block-order probe | Test block-order probe | Known-length exact |
|---|---:|---:|---:|
| best full | 0.882 | 0.877 | 0.675 |
| best val loss | 0.886 | 0.895 | 0.675 |

Segment/item representations are even stronger: item-embedding and item-motor block probes are `1.0` on train/val/test in the same diagnostic file.

## Autonomous Recall Evidence

Best-full checkpoint:

| Split | Full seq | Token | EOS | Pred length |
|---|---:|---:|---:|---:|
| Val | 0.325 | 0.627 | 0.875 | 0.875 |
| Test | 0.300 | 0.631 | 0.900 | 0.850 |

This gap between final-memory probe accuracy and autonomous exact recall is the key evidence for a recall-readout bottleneck.

## Duplicate Collapse By Length

### Val duplicate rate

| Length | Dup rate | Mean dup count | Mean unique predicted blocks | Exact |
|---:|---:|---:|---:|---:|
| 2 | 0.000 | 0.000 | 2.000 | 1.000 |
| 3 | 0.000 | 0.000 | 3.000 | 0.800 |
| 4 | 0.200 | 0.200 | 4.000 | 0.400 |
| 5 | 0.400 | 0.400 | 4.800 | 0.200 |
| 6 | 0.600 | 1.000 | 5.200 | 0.200 |
| 7 | 1.000 | 1.400 | 5.800 | 0.000 |
| 8 | 1.000 | 2.800 | 5.600 | 0.000 |
| 9 | 1.000 | 2.600 | 6.400 | 0.000 |

### Test duplicate rate

| Length | Dup rate | Mean dup count | Mean unique predicted blocks | Exact |
|---:|---:|---:|---:|---:|
| 2 | 0.000 | 0.000 | 2.000 | 1.000 |
| 3 | 0.000 | 0.000 | 3.000 | 1.000 |
| 4 | 0.200 | 0.200 | 4.000 | 0.400 |
| 5 | 0.800 | 1.200 | 4.000 | 0.000 |
| 6 | 0.800 | 1.000 | 5.000 | 0.000 |
| 7 | 0.800 | 1.000 | 6.000 | 0.000 |
| 8 | 1.000 | 1.600 | 6.400 | 0.000 |
| 9 | 1.000 | 2.400 | 6.600 | 0.000 |

Duplicate collapse clearly increases with sequence length in the current best-full checkpoint.

## Interpretation

The model often retains much of the set of target blocks, but it fails to emit a clean no-repeat ordered sequence during autonomous recall. This is why token accuracy remains nonzero for long sequences while exact accuracy drops to zero.

The current recall loop initializes from final memory and then consumes the same learned recall token at each step. It receives no previous predicted token, no ground-truth previous token, no explicit no-repeat state, no rank embedding, and no target length. That makes the decoder itself responsible for step identity, ordering, stopping, and duplicate suppression.

## Recommended Analysis-Only Diagnostics

Before changing architecture, run:

1. no-repeat constrained decoding as an analysis-only upper bound;
2. oracle-known-length decoding as an analysis-only EOS/length upper bound;
3. per-length duplicate histogram on val/test rows;
4. set-overlap vs order-error separation;
5. final-memory probe comparison for best-full and best-val-loss checkpoints.

Any constrained decoding should be reported as diagnostic only, not as the model's main cognitive result.
