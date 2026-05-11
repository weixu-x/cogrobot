#!/usr/bin/env bash
set -euo pipefail

V2_ROOT="corsi_artifacts/visual_base/datasets/freecam_index_v2_val"
V3_ROOT="corsi_artifacts/visual_base/datasets/freecam_index_v3_val"
V3_LEN_ROOTS=(
  "corsi_artifacts/visual_base/datasets/freecam_index_v3_val_len7"
  "corsi_artifacts/visual_base/datasets/freecam_index_v3_val_len8"
  "corsi_artifacts/visual_base/datasets/freecam_index_v3_val_len9"
)
OUTPUT_ROOT="corsi_artifacts/visual_base/results/checkpoint_diagnostics"
HAS_MATPLOTLIB=0
if python -c "import matplotlib" >/dev/null 2>&1; then
  HAS_MATPLOTLIB=1
fi

CHECKPOINTS=(
  "attention_step_scheduled_sampling:corsi_artifacts/visual_base/training/freecam_index_v2_order_attention_ss/best_model.pt"
)

run_eval() {
  local name="$1"
  local ckpt="$2"
  local data_root="$3"
  local mode="$4"
  local output_dir="$5"
  local title="$6"

  python evaluate_visual.py \
    --checkpoint "$ckpt" \
    --data-root "$data_root" \
    --mode "$mode" \
    --save-attention \
    --output-dir "$output_dir" || {
      echo "failed to evaluate $name $mode on $(basename "$data_root"); continuing"
      return 0
    }

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

  echo "evaluated $name $mode on $(basename "$data_root")"
}

for SPEC in "${CHECKPOINTS[@]}"; do
  NAME="${SPEC%%:*}"
  CKPT="${SPEC#*:}"
  if [[ ! -f "$CKPT" ]]; then
    echo "skipping missing checkpoint: $CKPT"
    continue
  fi

  RUN_DIR="$OUTPUT_ROOT/$NAME"
  run_eval "$NAME" "$CKPT" "$V2_ROOT" free_running "$RUN_DIR/eval_v2_free_running" "$NAME v2 free-running"
  run_eval "$NAME" "$CKPT" "$V3_ROOT" free_running "$RUN_DIR/eval_v3_free_running" "$NAME v3 free-running"
  run_eval "$NAME" "$CKPT" "$V3_ROOT" teacher_forced "$RUN_DIR/eval_v3_teacher_forced" "$NAME v3 teacher-forced"

  for V3_LEN_ROOT in "${V3_LEN_ROOTS[@]}"; do
    LEN_NAME="${V3_LEN_ROOT##*_val_}"
    run_eval "$NAME" "$CKPT" "$V3_LEN_ROOT" free_running "$RUN_DIR/eval_v3_${LEN_NAME}_free_running" "$NAME v3 ${LEN_NAME} free-running"
    run_eval "$NAME" "$CKPT" "$V3_LEN_ROOT" teacher_forced "$RUN_DIR/eval_v3_${LEN_NAME}_teacher_forced" "$NAME v3 ${LEN_NAME} teacher-forced"
  done
done

python corsi/experiments/visual_base/scripts/summarize_attention_ablation.py \
  --result-root "$OUTPUT_ROOT" \
  --output-csv "$OUTPUT_ROOT/checkpoint_diagnostics_summary.csv"

python corsi/experiments/visual_base/scripts/write_attention_analysis_report.py \
  --summary-csv "$OUTPUT_ROOT/checkpoint_diagnostics_summary.csv" \
  --output-md "$OUTPUT_ROOT/analysis_report.md"
