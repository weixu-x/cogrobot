#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

RESULT_ROOT="corsi_artifacts/visual_base/training/attention_ablation"
V2_ROOT="corsi_artifacts/visual_base/datasets/freecam_index_v2_val"
V3_ROOT="corsi_artifacts/visual_base/datasets/freecam_index_v3_val"
V3_LEN_ROOTS=(
  "corsi_artifacts/visual_base/datasets/freecam_index_v3_val_len7"
  "corsi_artifacts/visual_base/datasets/freecam_index_v3_val_len8"
  "corsi_artifacts/visual_base/datasets/freecam_index_v3_val_len9"
)
HAS_MATPLOTLIB=0
if python -c "import matplotlib" >/dev/null 2>&1; then
  HAS_MATPLOTLIB=1
fi

run_eval() {
  local ckpt="$1"
  local data_root="$2"
  local mode="$3"
  local output_dir="$4"
  local title="$5"

  python evaluate_visual.py \
    --checkpoint "$ckpt" \
    --data-root "$data_root" \
    --mode "$mode" \
    --save-attention \
    --output-dir "$output_dir"

  if [[ "$HAS_MATPLOTLIB" -eq 1 ]]; then
    python -m corsi.analysis.plot_attention_results \
      --attention-npz "$output_dir/attention_weights.npz" \
      --output-dir "$output_dir/plots" \
      --title "$title"

    python -m corsi.analysis.plot_error_results \
      --error-csv "$output_dir/error_taxonomy.csv" \
      --output-dir "$output_dir/plots" \
      --title "$title"
  else
    echo "skipping plots for $title because matplotlib is not installed"
  fi
}

CHECKPOINTS=("$RESULT_ROOT"/*/best_model.pt)
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "no checkpoints found under $RESULT_ROOT"
  exit 1
fi

for CKPT in "${CHECKPOINTS[@]}"; do
  RUN_DIR="$(dirname "$CKPT")"
  RUN_NAME="$(basename "$RUN_DIR")"

  run_eval "$CKPT" "$V2_ROOT" free_running "$RUN_DIR/eval_v2_free_running" "$RUN_NAME v2 free-running"
  run_eval "$CKPT" "$V3_ROOT" free_running "$RUN_DIR/eval_v3_free_running" "$RUN_NAME v3 free-running"
  run_eval "$CKPT" "$V3_ROOT" teacher_forced "$RUN_DIR/eval_v3_teacher_forced" "$RUN_NAME v3 teacher-forced"

  for V3_LEN_ROOT in "${V3_LEN_ROOTS[@]}"; do
    LEN_NAME="${V3_LEN_ROOT##*_val_}"
    run_eval "$CKPT" "$V3_LEN_ROOT" free_running "$RUN_DIR/eval_v3_${LEN_NAME}_free_running" "$RUN_NAME v3 ${LEN_NAME} free-running"
  done

  echo "evaluated $RUN_NAME"
done

python corsi/experiments/visual_base/scripts/summarize_attention_ablation.py
python corsi/experiments/visual_base/scripts/write_attention_analysis_report.py
