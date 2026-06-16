# Weekly Report Timeline - Corsi / robosuite

Date: 2026-05-12

Scope:
- Branches checked: `master`, `codex/set-initial-values-for-gripper`, `corsi-visual-heatmap`, `corsi-visual-scaleup`, `corsi-ee-xy-dataset-preview`.
- Sources used: git history, `corsi/WORKLOG.md`, experiment reports, committed result summaries, and local `corsi_artifacts/` metrics.
- Note: several latest metrics are from local artifacts and current working-tree changes, not necessarily committed.

## 1. One-Slide Summary

- Built a Corsi memory-task research stack on top of robosuite: sequence generation, datasets, coordinate baseline, visual baseline, attention variants, heatmap output, retention-delay hooks, and end-effector XY prediction.
- Coordinate baseline is solved: span 2-6 reaches `100%` full-sequence accuracy.
- Early visual CNN+LSTM baseline became trainable after dataset scale-up: best non-attention baseline reached `69%` full-sequence accuracy, but length 5-6 remained difficult.
- Attention variants changed the result materially: best v2 attention runs reached `98-100%` full-sequence accuracy on length 2-6.
- Generalization to longer length 7-9 is now the main differentiator: local Gaussian attention is strongest in current local evaluations.
- Retention delay and heatmap output are implemented but not yet strong.
- EE-XY target work shows a clear modeling lesson: autoregressive XY failed, step-conditioned XY solved length 2-3.

## 2. Timeline By Date

### 2026-01-27 to 2026-03-23: robosuite demo base

Work completed:
- Added demo/debug files and made `demo_test.py` video output configurable.
- Added InspireRightHand initial `qpos` setup in the gripper branch.

PPT message:
- Established the robosuite demo path and robot-hand initialization needed before formal Corsi experiments.

### 2026-03-24: coordinate Corsi baseline

Work completed:
- Created standalone `corsi/` project area.
- Added canonical 9-block layout, variable-length sequence generator, coordinate dataset, padding/mask collate, coordinate encoder-decoder LSTM, metrics, and training entry point.
- Added robosuite Corsi smoke-test helpers.

Experiment:
- `coord_baseline_mixed_2_6`
- Input: `(x, y, dx, dy)`
- Target: full block sequence
- Span: 2-6

Result:
- Best epoch: `20`
- Token accuracy: `1.000`
- Full-sequence accuracy: `1.000`
- Estimated span: `6`
- Accuracy by length 2-6: all `1.000`

Conclusion:
- Coordinate-only memory pipeline is correct and saturated.
- This is a useful reference, but too easy as a long-term benchmark.
- Next difficulty should come from vision, delay, or embodied action signals.

### 2026-04-20: visual baseline and dataset scale-up

Work completed:
- Reorganized experiments into `coordinate_base/` and `visual_base/`.
- Added visual dataset loader, visual collate, visual CNN+LSTM model, visual training pipeline, camera tools, export tools, and shard merge tools.
- Added keyframe-only robosuite freecam dataset export with rollout video disabled for scale-up.

Visual dataset evolution:
- `freecam_index_v1`: 16 train / 4 val, proof of wiring only.
- `freecam_index_v2`: 900 train / 100 val, sequence lengths 2-6.
- v2 validation split is balanced enough that long-sequence failure is not just sample shortage.

Main results:
- `freecam_index_v1_baseline`: full-sequence accuracy `0.000`.
- `freecam_index_v2_baseline` at 12 epochs: full-sequence accuracy `0.220`, estimated span `2`.
- `freecam_index_v2_20ep` resumed to 80 epochs: best epoch `75`, token accuracy `0.780`, full-sequence accuracy `0.690`, estimated span `4`.

Best non-attention v2 length-wise accuracy:
- Length 2: `1.000`
- Length 3: `0.963`
- Length 4: `0.947`
- Length 5: `0.333`
- Length 6: `0.056`

Capacity comparison:
- Hidden 256: full-sequence accuracy `0.580`.
- 2-layer LSTM: full-sequence accuracy `0.480`.
- Fair dropout/LR rerun: baseline `0.680`, hidden 256 `0.680`, 2-layer `0.370`.

Conclusion:
- Visual baseline is real and learnable.
- Length 2-4 are mostly solved; length 5-6 are the bottleneck.
- Bigger recurrent models alone do not solve the problem.
- Main failure pattern is long-sequence generalization / autoregressive error accumulation.

### 2026-05-08: visual ablation rerun results

Work completed:
- Added committed lightweight result artifacts for v2 ablation reruns.
- Compared attention, step embedding, and scheduled sampling combinations.

Key v2 ablation results:

| Variant | Attention | Step emb. | Scheduled sampling | Full-seq acc. | Token acc. | Span |
| --- | --- | --- | --- | ---: | ---: | ---: |
| attention only | yes | no | no | `0.990` | `0.997` | `6` |
| attention + scheduled sampling | yes | no | yes | `0.980` | `0.987` | `6` |
| attention + step | yes | yes | no | `0.920` | `0.974` | `6` |
| attention + step + scheduled sampling | yes | yes | yes | `0.990` | `0.995` | `6` |
| baseline LSTM | no | no | no | `0.570` | `0.752` | `4` |
| step + scheduled sampling, no attention | no | yes | yes | `0.690` | `0.808` | `4` |

Conclusion:
- Attention is the decisive improvement for the v2 length 2-6 task.
- Step embedding and scheduled sampling help somewhat without attention, but do not close the gap.
- Attention reduces wrong-block and repeat errors and recovers length 5-6 performance.

### 2026-05-09 to 2026-05-11: attention diagnostics and longer-span evaluation

Work completed:
- Added attention module variants and diagnostics:
  - global attention
  - local distance
  - local Gaussian
  - local window
  - noisy global
  - decay global
  - response suppression
  - cognitive combined variants
- Added error taxonomy, attention analysis, selected checkpoint diagnostics, and longer-span evaluations.
- Added v3 dataset family with length 2-9 evaluation:
  - v3 train: 1440 samples
  - v3 val: 160 samples
  - dedicated length 7/8/9 validation sets: 20 each

Selected local evaluation results:
- `local_gaussian_10`: v3 full-seq `0.988`; len7 `1.000`; len8 `0.950`; len9 `0.950`.
- `local_distance_05`: v3 full-seq `0.956`; len7 `1.000`; len8 `0.900`; len9 `0.800`.
- `cognitive_full`: v3 full-seq `0.894`; len7 `0.850`; len8 `0.750`; len9 `0.650`.
- `freecam_index_v2_order_attention_ss`: v3 full-seq `0.725`; len7 `0.400`; len8 `0.350`; len9 `0.100`.

Conclusion:
- v2 success does not automatically imply length 7-9 generalization.
- Local Gaussian attention is currently the best longer-span candidate.
- Attention locality and diagonal/local mass are useful diagnostic signals.
- Strong decay / response suppression can hurt extrapolation.

### 2026-05-11: heatmap output support

Work completed:
- Added heatmap target generation, heatmap losses, heatmap decoding, nearest-block decode, and unit tests.
- Added heatmap output mode to visual LSTM and training.
- Added heatmap configs and visualization tooling.

Current local results:
- `visual_heatmap_len2_only`: full-sequence accuracy `0.111`, token accuracy `0.111`.
- `visual_heatmap_len2_only_encoder_summary`: full-sequence accuracy `0.111`, token accuracy `0.194`.

Conclusion:
- Heatmap path is implemented and testable, but the current training setup is not yet competitive.
- It should be treated as infrastructure completed, not a final modeling result.

### 2026-05-11: retention delay baseline

Work completed:
- Added retention modes:
  - `hold_state`
  - `encoder_blanks`
- Added delay metadata, hidden-trace hooks, and result summaries.

Current local results:
- `retention_h128_blank_d2`: token accuracy `0.430`, full-sequence accuracy `0.140`.
- `retention_h128_blank_d8`: token accuracy `0.092`, full-sequence accuracy `0.000`.

Conclusion:
- Blank recurrent delay strongly degrades recall.
- The current model is not robust to retention intervals without explicit training or architectural support.
- This is a good aging / working-memory manipulation axis.

### 2026-05-11 to 2026-05-12: EE-XY visual target work

Work completed:
- Added EE-XY dataset export, merge, preview, and visualization scripts.
- Added target types for continuous XY output.
- Added step-conditioned decoder mode in current working tree.
- Added support for exhaustive length 2-3 splits and sampled length 4-5 splits.

Dataset validation:
- Preview dataset: 5 trials, validation passed.
- Exhaustive length 2-3 train: 514 samples, validation passed.
- Exhaustive length 2-3 val: 134 samples, validation passed, train/val overlap `0`.
- Current working tree adds sampled length 4-5 plan:
  - train: 1300 samples
  - val: 260 samples
  - full valid sequence counts: length 4 = 4608, length 5 = 36864

XY model results:
- Autoregressive XY length 2-3: full-sequence accuracy `0.000`, repeat-error rate about `0.993`.
- Tiny overfit without step-conditioned decoder: full-sequence accuracy `0.000`.
- Step-conditioned XY length 2-3: token accuracy `0.997`, full-sequence accuracy `0.993`.
- Step-conditioned tiny overfit: token accuracy `1.000`, full-sequence accuracy `1.000`.

Conclusion:
- Continuous EE-XY prediction should not depend on previous discrete block-token feedback.
- Step index + encoder summary is the right conditioning for XY output.
- Next useful experiment is scaling step-conditioned XY from length 2-3 to sampled length 4-5.

## 3. Test / Verification Status

Existing Corsi test coverage:
- `tests/test_corsi_attention.py`
- `tests/test_corsi_heatmaps.py`

Local verification attempt:
- `python -m pytest tests/test_corsi_attention.py tests/test_corsi_heatmaps.py -q`
  - failed because `pytest` is not installed.
- `python -m unittest tests.test_corsi_attention tests.test_corsi_heatmaps`
  - result: 2 passed, 4 skipped, 1 error.
  - error reason: current interpreter lacks `numpy`, so `test_corsi_heatmaps.py` could not import.

PPT wording:
- Test files were added for attention and heatmap utilities.
- Current shell environment cannot fully execute them because `pytest` and `numpy` are missing.
- This is an environment dependency issue, not necessarily a code failure.

## 4. Recommended PPT Structure

Slide 1 - Weekly focus:
- From symbolic Corsi baseline to visual memory, attention, delay, and EE-XY targets.

Slide 2 - Timeline:
- 03-24 coordinate baseline solved.
- 04-20 visual baseline and scale-up.
- 05-08 attention ablation results.
- 05-09/11 attention diagnostics and longer-span evaluation.
- 05-11 heatmap / delay / EE-XY extensions.

Slide 3 - Coordinate baseline:
- 100% full-sequence accuracy, span 6.
- Conclusion: pipeline works, task saturated.

Slide 4 - Visual baseline:
- v1 failed because too small.
- v2 reached 69% full-seq and span 4.
- Long sequences 5-6 are the bottleneck.

Slide 5 - Attention breakthrough:
- Attention variants reach 98-100% on v2.
- Attention is more important than width/depth.

Slide 6 - Longer-span diagnosis:
- v3 length 7-9 exposes generalization differences.
- Local Gaussian attention currently best.

Slide 7 - Negative / diagnostic experiments:
- Retention blanks degrade memory.
- Heatmap path implemented but weak.
- These are useful next research axes.

Slide 8 - EE-XY direction:
- EE-XY dataset validated.
- Autoregressive XY failed.
- Step-conditioned XY reaches 99%+ on length 2-3.

Slide 9 - Next week:
- Run sampled length 4-5 step-conditioned EE-XY.
- Promote best local Gaussian attention setup into the main reference.
- Re-run tests in an environment with `pytest`, `numpy`, and `torch`.
- Decide whether delay should be trained from scratch or initialized from attention checkpoints.

## 5. Short Chinese PPT Bullets

本周主要工作:
- 整理并扩展 Corsi 记忆任务实验线，从坐标输入推进到 robosuite 视觉输入、注意力机制、延迟保持、热力图输出和末端执行器 XY 输出。
- 坐标基线已完全跑通，span 2-6 全序列准确率达到 `100%`。
- robosuite freecam 视觉基线从小数据集 `0%` 提升到 v2 数据集 `69%`，但 length 5-6 仍是瓶颈。
- 注意力机制显著提升 v2 表现，最佳配置达到 `99%` 左右全序列准确率，估计 span 提升到 `6`。
- 更长序列 length 7-9 上，local Gaussian attention 当前泛化最好。
- EE-XY 方向完成数据导出和验证；自回归 XY 失败，step-conditioned XY 在 length 2-3 上达到 `99%+`。

关键结论:
- 简单加大 LSTM 容量不能解决视觉长序列问题。
- 注意力机制是当前视觉 Corsi 的核心改进。
- delay / heatmap 是已具备工程入口但仍需继续优化的方向。
- 连续 XY 输出需要 step-conditioned decoder，而不是沿用离散 token 自回归结构。
