# V2 Current Model Audit - 2026-06-29

## Scope

This audit was redone directly in the main checkout `/home/wei2025/Developer/cogrobot` so local generated V2 artifacts under `corsi_artifacts/memory_recall_v2` are visible.

No full training, dataset generation, V1 artifact modification, old motion-baseline workflow modification, artifact deletion, commit, or push was performed.

## Current Active Configuration

| Item | Value |
|---|---|
| Dataset | `corsi_memory_recall_v2_k12` |
| Canonical root | `corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12` |
| Schema | `scala_corsi_memory_recall_v2_canonical_v1` |
| Fingerprint | `782962d47e55a1e4631919f4042bc1eb31ff0b66c7c27adff27aaf7b994b4485` |
| Image size | 128x128 |
| K samples per segment | 12 |
| Blocks | 9 |
| EOS token | 9 |
| Max length | 9 |
| Splits | train/val/test = 320/40/40 |

K=20/K=30 current V2 artifacts:

NOT AVAILABLE IN CURRENT ARTIFACTS.

Deprecated old K=20/K=30 motion-base artifacts must not be reused as current V2 evidence.

## Current Best Checkpoint

Primary current checkpoint:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624/best_full_sequence.pt`

Primary eval JSONs:

- `stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_val_eval.json`
- `stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_test_eval.json`

## What The Model Has Learned

The current slot-compress `D_mem=64` model has learned strong block identity in item/final-memory representations but only partial autonomous recall.

Evidence:

- item-embedding and item-motor block probes are `1.0` on train/val/test;
- final-memory block-order probe is val/test `0.882/0.877` for best-full;
- known-length exact final-memory probe is `0.675`;
- autonomous recall is much lower: val/test exact full sequence `0.325/0.300`.

Conclusion: the strongest current evidence points to a final-memory-to-autonomous-recall bottleneck, not upstream RGB/presentation storage as the primary remaining limitation.

## Model Output Contract

Model inputs:

- `images [B,L,K,3,128,128]`
- `segment_mask [B,L]`
- `frame_mask [B,L,K]`

Recall-stage outputs:

- `logits [B,T,10]`: block IDs `0..8` plus EOS `9`
- `coord [B,T,2]`: weak block XY target for non-EOS recall steps

Presentation-stage auxiliary outputs:

- `pred_joint [B,L,K,7]`
- `pred_ee_pose [B,L,K,7]`
- `pred_ee_xy [B,L,K,2]`

The model is not predicting block ID at every physical movement frame. Block/EOS logits are autonomous recall-step outputs.

## Loss Definition

Stage 1:

```text
loss = 1.0 * joint_loss + 0.5 * ee_pose_loss + 0.5 * ee_xy_loss
```

Stage 2 default:

```text
loss = 1.0 * seq_loss
     + 0.3 * memory_order_loss
     + 0.05 * coord_loss
     + 0.1 * joint_loss
     + 0.05 * ee_pose_loss
     + 0.05 * ee_xy_loss
```

Disabled by default:

```text
memory_order_final = 0.0
memory_length = 0.0
```

This audit adds component-returning support and aggregation for future train/eval outputs without changing default scalar loss behavior.

## Current Results

| Split | Epoch | Full seq | Token | EOS | Pred length |
|---|---:|---:|---:|---:|---:|
| Val | 63 | 0.325 | 0.627 | 0.875 | 0.875 |
| Test | 63 | 0.300 | 0.631 | 0.900 | 0.850 |

## Per-Length Performance

### Exact full-sequence accuracy

| Length | Val | Test |
|---:|---:|---:|
| 2 | 1.000 | 1.000 |
| 3 | 0.800 | 1.000 |
| 4 | 0.400 | 0.400 |
| 5 | 0.200 | 0.000 |
| 6 | 0.200 | 0.000 |
| 7 | 0.000 | 0.000 |
| 8 | 0.000 | 0.000 |
| 9 | 0.000 | 0.000 |

### Token accuracy

| Length | Val | Test |
|---:|---:|---:|
| 2 | 1.000 | 1.000 |
| 3 | 0.950 | 1.000 |
| 4 | 0.760 | 0.840 |
| 5 | 0.700 | 0.700 |
| 6 | 0.743 | 0.629 |
| 7 | 0.575 | 0.575 |
| 8 | 0.556 | 0.444 |
| 9 | 0.300 | 0.440 |

### Duplicate sequence rate

| Length | Val | Test |
|---:|---:|---:|
| 2 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 |
| 4 | 0.200 | 0.200 |
| 5 | 0.400 | 0.800 |
| 6 | 0.600 | 0.800 |
| 7 | 1.000 | 0.800 |
| 8 | 1.000 | 1.000 |
| 9 | 1.000 | 1.000 |

## Serial-Position Accuracy

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

## Failure Interpretation

Short sequences are mostly solved. Exact recall collapses on longer sequences, while token accuracy and set overlap remain nonzero. This means the model often carries useful block information but fails to emit the whole ordered sequence exactly.

The dominant long-sequence pattern is duplicate collapse:

- duplicate rate is near zero for lengths 2-3;
- duplicate rate rises at length 4;
- duplicate rate reaches 0.8-1.0 on most test sequences of length 5-9;
- exact accuracy is zero on test lengths 5-9.

EOS and predicted-length errors matter, but they do not fully explain the failure. Many long examples have acceptable EOS/length behavior and still fail because of repeats, substitutions, and serial-position degradation.

## Stage 1 / Stage 2 Evidence

Stage 1 is visual-motor grounding. It trains RGB presentation encoding against joint/EE auxiliary targets.

Stage 2 is autonomous sequence recall. It trains RGB presentation -> final memory -> block/EOS recall.

Stage 2 warm-start loads only:

- `visual_encoder.`
- `visual_lstm.`
- `motor_lstm.`
- `joint_head.`
- `ee_pose_head.`
- `ee_xy_head.`

Recorded evidence for pretraining is limited: warm-start improved partial token/length metrics in one seed/config but did not produce nonzero full-sequence accuracy before later memory/readout isolates. Pretraining necessity is therefore:

UNKNOWN - needs audit

## K Ablation Status

Current K is 12. K=20/K=30 current V2 results are:

NOT AVAILABLE IN CURRENT ARTIFACTS.

Plan: see `reports/v2_k_ablation_plan_20260629.md`.

## Loss / Output Ablation Status

The current ID+XY output is justified as:

- ID logits: behavioral target;
- XY: weak spatial grounding and diagnostic geometry;
- joint/EE: presentation grounding auxiliaries, not cognitive output.

Plan: see `reports/v2_loss_output_ablation_plan_20260629.md`.

## Answered Supervisor Questions

| Question | Answer |
|---|---|
| Image input 64 or 128? | 128x128 |
| Current K? | 12 |
| Current best result? | slot-compress `D_mem=64`, best-full val/test full `0.325/0.300` |
| Length 2-9 results? | now available above |
| Serial-position accuracy? | now available above |
| Main limit? | autonomous recall readout / duplicate collapse |
| Does final memory contain ordered identity? | yes, probe val/test about `0.88` |
| Is pretraining proven sufficient? | UNKNOWN - needs audit |
| Do we have current K=20/K=30 V2 results? | NOT AVAILABLE IN CURRENT ARTIFACTS. |

## Recommendation

Do not redesign the architecture yet.

Next evidence-first step:

1. Re-evaluate current checkpoints with the enhanced evaluator to write explicit duplicate/set-overlap/loss-component fields.
2. Run analysis-only no-repeat and oracle-known-length decoding diagnostics.
3. Use the planned loss/output ablations to identify which supervision terms change duplicate collapse.
4. Only then decide whether a recall-readout change is justified.
