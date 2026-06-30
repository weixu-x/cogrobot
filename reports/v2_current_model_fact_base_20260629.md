# V2 Current Model Fact Base - Priority 1

Date: 2026-06-29  
Repository: `/home/wei2025/Developer/cogrobot`  
Scope: current Corsi Memory Recall V2 fact base only. No training, K-generation, evaluator instrumentation, architecture redesign, commit, or push was performed.

## 1. Current Active V2 Lane

The active lane is `corsi_memory_recall_v2_k12`, a segmented RGB memory-recall model. It uses canonical data at `corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12` with schema `scala_corsi_memory_recall_v2_canonical_v1` and fingerprint `782962d47e55a1e4631919f4042bc1eb31ff0b66c7c27adff27aaf7b994b4485`. The current production checkpoint family is `stage2_seed0_slot_compress_dmem64_full_20260624`; the selected checkpoint is `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624/best_full_sequence.pt`.

Evidence: `PROJECT_STATE.md`, `reports/v2_memory_isolates_snapshot_20260626.md`, `corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12/manifest.json`, `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624/summary.json`.

## 2. Current Config

| Item | Value |
|---|---|
| Source raw dataset | `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50` |
| Canonical dataset | `corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12` |
| Image input | `3x128x128`, layout `L,K,C,H,W` |
| K samples per segment | `12` |
| Length range | `2..9` |
| Blocks / EOS / padding | `9` blocks, EOS `9`, ignore `-100` |
| Target rule | `block_order followed by EOS` |
| Train / val / test | `320 / 40 / 40`; per length `40 / 5 / 5` |
| Current generated run config | `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_slot_compress_dmem64.json` |
| CNN / visual / motor / item | `64 / 64 / 64 / 64` |
| Memory | `slot_compress`, `D_mem=64`, slot dim `16` |
| Recall | hidden `64`, learned recall-token dim `32` |
| Warm start | Stage 2 loaded Stage 1 grounding checkpoint `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage1_seed0_full_20260624/best.pt` |

The checked-in config `corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json` defines data/canonical fields. The current best run adds `memory_dim=64`, `memory_write_mode=slot_compress`, and `memory_slot_dim=16` in `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_slot_compress_dmem64.json`. Remaining model defaults are dataclass defaults in `corsi/experiments/corsi_memory_recall_v2/model.py:14-34` and are copied by `build_model` at `model.py:579-613`.

Canonical data-processing flow: `canonicalize.py` validates raw RGB as `[T,128,128,3]`, uses existing `segments.json` boundaries, samples `K=12` timestamps per segment, transposes images to `[L,K,C,H,W]`, interpolates joint/EE targets, copies block `xy_norm` targets, and creates `target_tokens = block_order + [EOS]` (`canonicalize.py:186-353`). Splits are deterministic and length-balanced by `deterministic_split` (`canonicalize.py:356-392`). The dataset loader exposes only `images`, `segment_mask`, and `frame_mask` as `model_inputs` and keeps tokens, block IDs, rank, length, and pose/XY fields as targets or metadata (`dataset.py:118-155`).

## 3. Current Best Checkpoint

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624/best_full_sequence.pt` exists and is the current selected checkpoint. It is Stage 2, seed 0, epoch `63`, warm-started from Stage 1 epoch `27` with `30` grounding keys.

The associated eval JSONs are:

- `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_val_eval.json`
- `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_test_eval.json`
- duplicate/probe diagnostics: `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_minimal_localization.json`

## 4. Existing Experiment Recap

| Run label | Run | Val full | Val token | Val EOS | Val length | Test full | Test token | Test EOS | Test length |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `original Stage 2 scaffold` | `stage2_seed0_full_20260624` | 0.000 | 0.204 | 1.000 | 0.250 | 0.000 | 0.192 | 1.000 | 0.250 |
| `Stage 1 warm-start` | `stage2_seed0_warmstart_full_20260624` | 0.000 | 0.308 | 1.000 | 0.975 | 0.000 | 0.273 | 1.000 | 1.000 |
| `order prefix auxiliary` | `stage2_seed0_orderaux_full_20260624` | 0.025 | 0.458 | 1.000 | 0.975 | 0.100 | 0.469 | 1.000 | 0.925 |
| `D_mem=64 LSTM memory` | `stage2_seed0_dmem64_full_20260624` | 0.050 | 0.392 | 1.000 | 0.975 | 0.050 | 0.427 | 1.000 | 0.975 |
| `slot-compress plus final-memory auxiliaries` | `stage2_seed0_slot_compress_aux_full_20260624` | 0.075 | 0.450 | 1.000 | 0.925 | 0.025 | 0.465 | 1.000 | 0.925 |
| `slot-compress D_mem=16` | `stage2_seed0_slot_compress_full_20260624` | 0.200 | 0.573 | 0.875 | 0.750 | 0.200 | 0.650 | 0.925 | 0.800 |
| `slot-compress D_mem=64 current` | `stage2_seed0_slot_compress_dmem64_full_20260624` | 0.325 | 0.627 | 0.875 | 0.875 | 0.300 | 0.631 | 0.900 | 0.850 |
| `direct final-memory order/length auxiliary` | `stage2_seed0_orderaux_finalmem_full_20260624` | 0.075 | 0.412 | 0.950 | 0.925 | 0.050 | 0.415 | 0.900 | 0.850 |
| `balanced prefix auxiliary` | `stage2_seed0_orderaux_balanced_full_20260624` | 0.125 | 0.442 | 1.000 | 0.875 | 0.075 | 0.462 | 1.000 | 0.950 |

The current best production model is the slot-compress `D_mem=64` run. `best_val_loss.pt` in the same run has higher test full-sequence accuracy (`0.350`) than `best_full_sequence.pt` (`0.300`), but it is not the selected production checkpoint because the selection rule is validation full-sequence first.

## 5. Checkpoint Selection Logic

For Stage 2, `best_full_sequence.pt` is selected by the tuple `(val_full_sequence_accuracy, val_token_accuracy, val_predicted_length_accuracy, -val_loss)`. `best_token.pt` and `best_val_loss.pt` are tracked with different tuple orderings. Code: `corsi/experiments/corsi_memory_recall_v2/train.py:243-274`; primary selection is set at `train.py:35` and applied at `train.py:696-819`.

Current best-full selection score from `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624/summary.json` is `[0.325, 0.6269230769230769, 0.875, -1.1997230748335521]`.

## 6. Val/Test Final Metrics

| Split | Epoch | N | Full sequence | Token | EOS | Predicted length | Error taxonomy S/O/I/T | Duplicate sequence rate |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| val | 63 | 40 | 0.325 | 0.627 | 0.875 | 0.875 | 61 / 8 / 14 / 6 | 0.525 |
| test | 63 | 40 | 0.300 | 0.631 | 0.900 | 0.850 | 58 / 12 / 14 / 3 | 0.575 |

`S/O/I/T` means substitution / omission / insertion / transposition. Exact/token/EOS/length/taxonomy come from the best-full eval JSONs. Duplicate rates come from `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_minimal_localization.json` under `collapse.best_full_sequence_val/test`, which points back to the same eval JSONs.

## 7. Length 2-9 Performance

| Length | Count/split | Val exact | Val token | Val EOS | Val pred length | Val duplicate | Test exact | Test token | Test EOS | Test pred length | Test duplicate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| 3 | 5 | 0.800 | 0.950 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| 4 | 5 | 0.400 | 0.760 | 0.800 | 0.800 | 0.200 | 0.400 | 0.840 | 0.800 | 0.800 | 0.200 |
| 5 | 5 | 0.200 | 0.700 | 0.800 | 0.800 | 0.400 | 0.000 | 0.700 | 0.800 | 0.800 | 0.800 |
| 6 | 5 | 0.200 | 0.743 | 0.800 | 0.800 | 0.600 | 0.000 | 0.629 | 1.000 | 1.000 | 0.800 |
| 7 | 5 | 0.000 | 0.575 | 0.800 | 0.800 | 1.000 | 0.000 | 0.575 | 0.800 | 0.600 | 0.800 |
| 8 | 5 | 0.000 | 0.556 | 0.800 | 0.800 | 1.000 | 0.000 | 0.444 | 1.000 | 1.000 | 1.000 |
| 9 | 5 | 0.000 | 0.300 | 1.000 | 1.000 | 1.000 | 0.000 | 0.440 | 0.800 | 0.600 | 1.000 |

Length-level exact, token, EOS, predicted-length, and duplicate values are aggregated from existing row-level fields in the current best-full eval JSONs. The eval JSONs do not contain top-level `per_length_token_accuracy`, so the exact checked paths are recorded in `metrics_unavailable_with_checked_paths` in `reports/v2_current_fact_metrics_20260629.json`.

## 8. Duplicate / EOS / Length / Substitution / Omission Failures

| Split | EOS error rate | Length error rate | Duplicate sequence rate | Substitutions | Omissions | Insertions | Transpositions | Mean unique predicted blocks | Mean set overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.125 | 0.125 | 0.525 | 61 | 8 | 14 | 6 | 4.600 | 0.797 |
| test | 0.100 | 0.150 | 0.575 | 58 | 12 | 14 | 3 | 4.625 | 0.798 |

Failure counts are current-artifact counts, not new evaluation. Edit taxonomy comes from `rows[].errors` and top-level `error_taxonomy` in `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_val_eval.json` and `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_test_eval.json`. Duplicate/set-overlap metrics come from `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_minimal_localization.json`.

Serial-position accuracy shows the long-sequence degradation directly:

| Position | Val accuracy | Val count | Test accuracy | Test count |
|---:|---:|---:|---:|---:|
| 0 | 0.850 | 40 | 0.775 | 40 |
| 1 | 0.850 | 40 | 0.925 | 40 |
| 2 | 0.629 | 35 | 0.629 | 35 |
| 3 | 0.433 | 30 | 0.433 | 30 |
| 4 | 0.440 | 25 | 0.520 | 25 |
| 5 | 0.250 | 20 | 0.350 | 20 |
| 6 | 0.333 | 15 | 0.133 | 15 |
| 7 | 0.400 | 10 | 0.200 | 10 |
| 8 | 0.000 | 5 | 0.200 | 5 |

## 9. Stage 1 Loss

Stage 1 is presentation-frame grounding only:

```text
1.0 * joint_loss + 0.5 * ee_pose_loss + 0.5 * ee_xy_loss
```

It uses `pred_joint`, `pred_ee_pose`, and `pred_ee_xy` against the standardized per-frame `targets["joint"]`, `targets["ee_pose"]`, and `targets["ee_xy"]`, masked by `model_inputs["frame_mask"]`. Code: `losses.py:307-340`.

## 10. Stage 2 Loss

Stage 2 uses `combined_v2_loss`, the autonomous recall objective plus weak coordinate and auxiliary presentation losses. Default weights in `losses.py:12-21` are:

```text
seq=1.0
memory_order=0.3
memory_order_final=0.0
memory_length=0.0
coord=0.05
joint=0.1
ee_pose=0.05
ee_xy=0.05
```

The current best run has `loss_weights={}` in `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624/summary.json`, so these code defaults apply.

## 11. Loss Term Table

| Term | Code location | Weight | Target tensor | Supervision stage | Meaning |
|---|---|---:|---|---|---|
| `seq_loss` | `corsi/experiments/corsi_memory_recall_v2/losses.py:37`, `:228` | 1.0 | `targets["tokens"]` | Stage 2 | Cross-entropy over autonomous recall block+EOS logits, ignoring `-100` padding. |
| `memory_order_loss` | `losses.py:82`, `:232` | 0.3 | `targets["tokens"]`, `token_mask`, segment mask | Stage 2 | Prefix CE from each memory update to block IDs seen so far; excludes EOS/padding. |
| `final_memory_order_loss` | `losses.py:141`, `:244` | 0.0 | `targets["tokens"]`, `token_mask` | Stage 2 optional | CE from final memory to full ordered block sequence; disabled by default. |
| `final_memory_length_loss` | `losses.py:178`, `:255` | 0.0 | `targets["tokens"]`, `token_mask` | Stage 2 optional | CE for valid block-token count, excluding EOS; disabled by default. |
| `coord_loss` | `losses.py:54`, `:266` | 0.05 | `targets["target_xy"]` or `targets["block_xy"]` | Stage 2 | Weak MSE on recall `coord` for non-EOS, non-padding block steps. |
| `joint_loss` | `losses.py:199`, `:280`, `:321` | Stage 1: 1.0; Stage 2: 0.1 | `targets["joint"]` | Stages 1 and 2 | Masked per-frame MSE for presentation `pred_joint`. |
| `ee_pose_loss` | `losses.py:199`, `:280`, `:324` | Stage 1: 0.5; Stage 2: 0.05 | `targets["ee_pose"]` | Stages 1 and 2 | Masked per-frame MSE for presentation `pred_ee_pose`. |
| `ee_xy_loss` | `losses.py:199`, `:280`, `:326` | Stage 1: 0.5; Stage 2: 0.05 | `targets["ee_xy"]` | Stages 1 and 2 | Masked per-frame MSE for presentation `pred_ee_xy`. |

## 12. Output Behavior

Presentation accepts only `images`, `segment_mask`, and `frame_mask` as model inputs. The dataset loader returns these under `model_inputs` and places block IDs, ranks, length, tokens, joint/EE targets, and block XY under targets/metadata, not model inputs (`dataset.py:118-155`).

Presentation outputs:

- `pred_joint [B,L,K,7]`
- `pred_ee_pose [B,L,K,7]`
- `pred_ee_xy [B,L,K,2]`
- `item_embeddings [B,L,item_dim]`
- `item_visual`, `item_motor`, and optional traces

Code: `model.py:194-264`.

Memory outputs:

- `final_memory`
- `memory_before_noise`
- `memory_order_logits [B,L,P,num_blocks]`
- `final_memory_order_logits [B,P,num_blocks]`
- `final_memory_length_logits [B,P]`

Code: `model.py:266-417`.

Recall outputs:

- `logits [B,T,num_tokens]` for block IDs plus EOS
- `coord [B,T,2]`
- `recall_inputs`, which are repeated learned recall tokens, not previous ground-truth or predicted block IDs

Code: `model.py:418-457` and `model.py:484-499`.

Movement/presentation frames are not per-frame block-ID prediction in the current code. Per-frame presentation heads predict joint/EE quantities only (`model.py:246-248`). Block-ID supervision appears at autonomous recall steps via `logits`, and at memory update/serial-position auxiliaries via `memory_order_logits`; it is not a `K`-frame block classifier.

## 13. What The Current Model Has Learned

The model has learned the short end of the task and a strong presentation-to-memory representation:

- Segment/item states are linearly block-decodable at 1.0 val/test accuracy in the current localization artifact.
- Final memory carries ordered block identity: best-full final-memory order probe is `0.882` val and `0.877` test, with known-length exact `0.675` on both splits.
- Autonomous recall solves lengths 2-3 well: test exact is `1.000` for lengths 2 and 3.
- It maintains moderate token accuracy overall: `0.627` val and `0.631` test.

## 14. What Currently Limits Performance

The main bottleneck is autonomous recall readout, not upstream RGB/visual/motor encoding. The final-memory probe is much stronger than autonomous exact recall: known-length exact probe exceeds autonomous exact by `0.350` val and `0.375` test. Duplicate-collapse remains substantial: duplicate sequence rate is `0.525` val and `0.575` test. Long-sequence exact accuracy is zero for val lengths 7-9 and test lengths 5-9.

## 15. What Remains Unknown

| Item | Status | Checked path(s) |
|---|---|---|
| Top-level `per_length_metrics` / `per_length_token_accuracy` in best-full eval JSONs | NOT AVAILABLE IN CURRENT ARTIFACTS; derived from rows instead | `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_val_eval.json`, `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_test_eval.json` |
| Top-level `duplicate_metrics` / `failure_taxonomy` in best-full eval JSONs | NOT AVAILABLE IN CURRENT ARTIFACTS; duplicate metrics are in minimal-localization JSON | `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_val_eval.json`, `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_test_eval.json`, `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_minimal_localization.json` |
| Full test-split causal sanity checks for current best checkpoint | NOT AVAILABLE IN CURRENT ARTIFACTS; val JSON has one-batch sanity checks only | `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_test_eval.json`, `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_val_eval.json` |
| Single JSON source for all architecture fields | NOT AVAILABLE IN CURRENT ARTIFACTS; effective config is config JSON plus generated run config plus model defaults | `corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json`, `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_slot_compress_dmem64.json`, `corsi/experiments/corsi_memory_recall_v2/model.py` |

## 16. Priority 2 Experiment / Instrumentation Queue

Do not execute these in this Priority 1 thread.

1. Add evaluator output that writes top-level `per_length_metrics`, `per_length_token_accuracy`, `duplicate_metrics`, and `failure_taxonomy` for all current evals.
2. Run a recall-readout / duplicate-suppression isolate from the current slot-compress `D_mem=64` checkpoint family.
3. Evaluate K=12/20/30 only in a separate Priority 2 thread.
4. Run Stage 1 epoch ablation only in a separate Priority 2 thread.
5. Run no-pretrain vs warm-start ablation only in a separate Priority 2 thread.
6. Run loss ablations for `memory_order`, `coord`, final-memory order, and length losses only in separate Priority 2 work.
7. Add full val/test causal sanity-check artifacts for current best checkpoints.

## 17. Priority 3 Paper-Narrative Queue

Do not expand these into paper claims until Priority 2 evidence exists.

1. Separate the factual statement "final memory is decodable" from the stronger claim "autonomous recall succeeds".
2. Frame current failure as a recall-readout / duplicate-collapse limit rather than an upstream visual encoding failure.
3. Report Corsi length effects only with current artifact tables or post-Priority-2 re-evaluation.
4. Keep motor-shell claims separate from cognitive recall metrics.
5. Compare V1 and V2 only after V2 instrumentation is complete and current artifacts are re-exported with consistent metric fields.

## Artifact Paths Used

- `corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12/manifest.json`
- `corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json`
- `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_slot_compress_dmem64.json`
- `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624/summary.json`
- `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624/best_full_sequence.pt`
- `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_val_eval.json`
- `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_test_eval.json`
- `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_minimal_localization.json`
- `reports/v2_memory_isolates_snapshot_20260626.md`

## Companion Machine-Readable Files

- `reports/v2_current_fact_config_20260629.json`
- `reports/v2_current_fact_metrics_20260629.json`
