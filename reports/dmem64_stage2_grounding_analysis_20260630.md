# D_mem64 Direct Stage 2 Grounding Analysis - 2026-06-30

This is a frozen checkpoint analysis. It does not resume model training and does not update weights.

Checkpoint: `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12_expanded800_20260630/pause_snapshot_stage2_direct_interrupt_20260630/dmem64_seed0/latest.pt`
Checkpoint stage/epoch: `2` / `40`

## Motion Aux Heads

| Split | Field | Norm scalar MAE | Norm scalar RMSE | Physical scalar MAE | Physical scalar RMSE | Physical vector L2 mean | Physical vector L2 RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | joint | 0.118874 | 0.196277 | 0.010671 | 0.015487 | 0.034605 | 0.040976 |
| train | ee_pose | 0.277146 | 0.453160 | 0.002733 | 0.003777 | 0.008776 | 0.009992 |
| train | ee_xy | 0.056685 | 0.075600 | 2.239464 | 2.949380 | 3.516884 | 4.171053 |
| val | joint | 0.119451 | 0.199584 | 0.010703 | 0.015700 | 0.034634 | 0.041538 |
| val | ee_pose | 0.273817 | 0.450681 | 0.002709 | 0.003760 | 0.008693 | 0.009948 |
| val | ee_xy | 0.056630 | 0.075687 | 2.235653 | 2.949023 | 3.515448 | 4.170549 |
| test | joint | 0.117281 | 0.195433 | 0.010545 | 0.015428 | 0.034233 | 0.040819 |
| test | ee_pose | 0.273084 | 0.445615 | 0.002704 | 0.003736 | 0.008670 | 0.009884 |
| test | ee_xy | 0.056443 | 0.075214 | 2.230768 | 2.936653 | 3.500550 | 4.153055 |

## Item Embedding Block Separability

| Split | Segments | Nearest-centroid acc | Ridge linear-probe acc |
| --- | ---: | ---: | ---: |
| train | 31444 | 1.000000 | 1.000000 |
| val | 1280 | 1.000000 | 1.000000 |
| test | 2140 | 1.000000 | 1.000000 |

PCA figure:

`reports/figures/dmem64_item_embedding_pca_20260630.png`

## Interpretation

- Stage 1 would warm-start the visual/motor grounding prefixes.
- Low aux errors indicate those prefixes and aux heads were learned during direct Stage 2.
- High item-probe accuracy indicates item embeddings already linearly separate the 9 block identities.
- If item-probe accuracy is high but aux errors are poor, motion grounding is not necessary for recall in this setup.
