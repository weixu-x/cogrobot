# Corsi V2 Expanded800 Dataset Manifest

- Status: `ok`
- Raw manifest: `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_expanded800_20260630/manifest.json`
- Raw samples: `5562`
- `baseline_dataset_design_record.md` found in checkout: `False`

## Self-Checks

| check | result |
| --- | --- |
| split x length counts | PASS |
| length-2 train complete 72 | PASS |
| L>=4 cross-split overlap zero | PASS |
| internal block repeats | PASS |
| raw failed/skipped episodes | PASS |

## Split x Length Counts

| split | length | actual | target |
| --- | --- | --- | --- |
| train | 2 | 72 | 72 |
| train | 3 | 300 | 300 |
| train | 4 | 600 | 600 |
| train | 5 | 800 | 800 |
| train | 6 | 800 | 800 |
| train | 7 | 800 | 800 |
| train | 8 | 800 | 800 |
| train | 9 | 800 | 800 |
| val | 2 | 10 | 10 |
| val | 3 | 30 | 30 |
| val | 4 | 30 | 30 |
| val | 5 | 30 | 30 |
| val | 6 | 30 | 30 |
| val | 7 | 30 | 30 |
| val | 8 | 30 | 30 |
| val | 9 | 30 | 30 |
| test | 2 | 20 | 20 |
| test | 3 | 50 | 50 |
| test | 4 | 50 | 50 |
| test | 5 | 50 | 50 |
| test | 6 | 50 | 50 |
| test | 7 | 50 | 50 |
| test | 8 | 50 | 50 |
| test | 9 | 50 | 50 |

## Cross-Split Block-Order Overlap

| length | split pair | overlap count |
| --- | --- | --- |
| 2 | train/val | 10 |
| 2 | train/test | 20 |
| 2 | val/test | 3 |
| 3 | train/val | 0 |
| 3 | train/test | 0 |
| 3 | val/test | 0 |
| 4 | train/val | 0 |
| 4 | train/test | 0 |
| 4 | val/test | 0 |
| 5 | train/val | 0 |
| 5 | train/test | 0 |
| 5 | val/test | 0 |
| 6 | train/val | 0 |
| 6 | train/test | 0 |
| 6 | val/test | 0 |
| 7 | train/val | 0 |
| 7 | train/test | 0 |
| 7 | val/test | 0 |
| 8 | train/val | 0 |
| 8 | train/test | 0 |
| 8 | val/test | 0 |
| 9 | train/val | 0 |
| 9 | train/test | 0 |
| 9 | val/test | 0 |

## Canonical Dataset

- Manifest: `corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12_expanded800_20260630/manifest.json`
- Samples: `5562`
- Fingerprint: `94d1813b3f7b4cb5fd2107e9bcdd306683c807dacd18b1429380de1f5575df78`
