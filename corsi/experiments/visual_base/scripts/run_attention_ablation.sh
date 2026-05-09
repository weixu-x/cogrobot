#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="corsi/experiments/visual_base/configs/attention_ablation"
CONFIGS=(
  "$CONFIG_DIR/global.json"
  "$CONFIG_DIR/local_distance_03.json"
  "$CONFIG_DIR/local_distance_05.json"
  "$CONFIG_DIR/local_gaussian_10.json"
  "$CONFIG_DIR/local_gaussian_15.json"
  "$CONFIG_DIR/local_window_1.json"
  "$CONFIG_DIR/local_window_2.json"
  "$CONFIG_DIR/noisy_global_01.json"
  "$CONFIG_DIR/decay_global_005.json"
  "$CONFIG_DIR/decay_global_01.json"
  "$CONFIG_DIR/response_suppression_10.json"
  "$CONFIG_DIR/cognitive_full.json"
)

for CONFIG in "${CONFIGS[@]}"; do
  python corsi/train_visual.py --config "$CONFIG"
done
