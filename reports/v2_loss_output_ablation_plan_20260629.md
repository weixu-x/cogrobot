# V2 Loss / Output Ablation Plan - 2026-06-29

## Scope

This is a config-only ablation plan. Do not change the architecture, input contract, dataset, recall loop, or add new loss terms for the first ablation pass.

## Default Stage 2 Loss

| Term | Default weight | Meaning |
|---|---:|---|
| `seq` | 1.0 | block/EOS recall CE |
| `memory_order` | 0.3 | prefix order CE from memory-update states |
| `coord` | 0.05 | weak recall-step block XY MSE |
| `joint` | 0.1 | presentation-frame joint auxiliary MSE |
| `ee_pose` | 0.05 | presentation-frame EE pose auxiliary MSE |
| `ee_xy` | 0.05 | presentation-frame EE XY auxiliary MSE |
| `memory_order_final` | 0.0 | direct final-memory order loss, disabled |
| `memory_length` | 0.0 | direct final-memory length loss, disabled |

Block ID logits are the behavioral output. XY is justified as weak spatial grounding and diagnostic geometry, not as the cognitive success criterion.

## Minimum Ablation Set

| ID | Name | Config change | Diagnostic meaning |
|---|---|---|---|
| A | seq only | `seq=1.0`, all other weights `0.0` | Tests pure sequence learning from RGB memory. |
| B | seq + coord | A + `coord=0.05` | Tests whether weak XY supervision helps recall or only adds pressure. |
| C | seq + memory_order | A + `memory_order=0.3` | Tests whether prefix memory-order supervision helps serial memory. |
| D | seq + joint/EE aux | A + `joint=0.1`, `ee_pose=0.05`, `ee_xy=0.05` | Tests whether visual-motor grounding helps recall. |
| E | current default | default weights | Reference objective. |
| F | current + final_memory_order | E + `memory_order_final=0.3` | Tests direct final-memory order supervision; risky because prior isolate worsened behavior. |
| G | current + memory_length | E + `memory_length=0.3` | Tests whether explicit length supervision reduces EOS/length errors. |

Do not remove heads in code. Set weights to zero so model capacity and forward contract remain fixed.

## Metrics

For each condition, report:

- full sequence accuracy
- token accuracy
- EOS accuracy
- predicted length accuracy
- per-length metrics for lengths 2-9
- serial-position accuracy
- duplicate sequence rate
- mean duplicate count
- mean unique predicted blocks
- set-overlap Jaccard
- substitution / omission / insertion / transposition
- loss components
- causal sanity checks

## Artifact Names

Use stable names:

- `stage2_seed{seed}_lossA_seq_only_20260629`
- `stage2_seed{seed}_lossB_seq_coord_20260629`
- `stage2_seed{seed}_lossC_seq_memory_order_20260629`
- `stage2_seed{seed}_lossD_seq_aux_20260629`
- `stage2_seed{seed}_lossE_current_default_20260629`
- `stage2_seed{seed}_lossF_current_final_memory_order_20260629`
- `stage2_seed{seed}_lossG_current_memory_length_20260629`

Stop each run at the same planned budget. Do not extend weak conditions just to make them competitive.
