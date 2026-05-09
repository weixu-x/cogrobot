#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="corsi_artifacts/visual_base/training/attention_ablation"
V2_ROOT="corsi_artifacts/visual_base/datasets/freecam_index_v2_val"
V3_ROOT="corsi_artifacts/visual_base/datasets/freecam_index_v3_test"

for CKPT in "$RESULT_ROOT"/*/best_model.pt; do
  RUN_DIR="$(dirname "$CKPT")"
  RUN_NAME="$(basename "$RUN_DIR")"

  python evaluate_visual.py \
    --checkpoint "$CKPT" \
    --data-root "$V2_ROOT" \
    --mode free_running \
    --save-attention \
    --output-dir "$RUN_DIR/eval_v2_free_running"

  python evaluate_visual.py \
    --checkpoint "$CKPT" \
    --data-root "$V3_ROOT" \
    --mode free_running \
    --save-attention \
    --output-dir "$RUN_DIR/eval_v3_free_running"

  python evaluate_visual.py \
    --checkpoint "$CKPT" \
    --data-root "$V3_ROOT" \
    --mode teacher_forced \
    --save-attention \
    --output-dir "$RUN_DIR/eval_v3_teacher_forced"

  echo "evaluated $RUN_NAME"
done
