# 实验报告记录 — V2 LSTM Associative Memory 绑定故障：诊断与修复

日期：2026-06-29

实验线：`corsi_memory_recall_v2` / `corsi_memory_recall_v2_k12`

主报告：`reports/v2_lstm_associative_memory_binding_final_report_20260629.md`

## 1. 背景

V2 LSTM associative memory 的原始失败表现为：sequence length / EOS 信号相对容易保留，但 ordered block identity 在 recall 阶段丢失或退化。Phase 1/3 probes 已经表明，问题不是单纯的 item write-in 缺失：write trajectory `[h_t;c_t]` 中可以线性解码 current item 和 write position。主要瓶颈是将有序内容压缩为单一 final memory vector 后再供 recall 使用。

## 2. 已完成的关键结论

- 原始 LSTM `D_mem=64` checkpoint 的 write trajectory `[h_t;c_t]` 上 current item probe 为 `0.982`，write position probe 为 `0.959`。
- 原始 checkpoint 的 final compressed `[h;c]` ordered identity probe 较弱，为 `0.595`。
- `memory_attention` readout 直接访问 per-step write trajectory 后，在 full-data 上达到 full/token/length `1.000/1.000/1.000`。
- `memory_attention` 胜出说明这是有效 readout repair/control condition；它不证明 final compressed memory 本身已经成为强 ordered-memory substrate。

## 3. 主要 artifacts

- Phase 1 baseline report: `reports/v2_lstm_phase1_binding_diagnostics_20260629.md`
- Phase 3 attention report: `reports/v2_lstm_attention_phase3_binding_diagnostics_20260629.md`
- Phase 2B repair summary: `reports/v2_lstm_phase2b_memory_attention_repair_20260629.md`
- Final report: `reports/v2_lstm_associative_memory_binding_final_report_20260629.md`
- Experiment (a) probe report: `reports/binding_auxsplit_fulldata_phase1_binding_diagnostics.md`
- Experiment (a) probe JSON: `reports/binding_auxsplit_fulldata_phase1_binding_diagnostics.json`

## 8. 讨论 / 判定实验

### 8.1 Attention 胜出版的定位

`memory_attention` checkpoint 在行为上已经满分，但 final compressed `[h;c]` ordered identity probe 仍弱。因此该 regime 更适合作为性能上界和 readout-control condition，而不是“压缩态工作记忆已经解决”的证据。

### 8.2 压缩态 regime 的意义

如果目标是解释 development/ageing 或 human-like working-memory error signatures，压缩态 / 容量受限 regime 仍然重要。满分 attention checkpoint 没有错误，天然无法呈现 U-shape 或 transposition gradient。

### 8.3 判定实验 (a)：0.25 binding checkpoint 已补测

目的：判断 full-data explicit-binding + auxsplit checkpoint 的 low full-sequence score 是单纯失败，还是 human-like graded degradation。

实际使用 checkpoint：

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_item_context_binding_auxsplit_dmem64_20260629/best_full_sequence.pt`

Checkpoint metadata:

- epoch: `69`
- validation selection score: full `0.250`, token `0.638`, length `0.800`
- diagnostic test full-sequence: `0.325`

Behavior metrics on test:

| Metric | Value |
| --- | ---: |
| full-sequence accuracy | `0.325` |
| token accuracy | `0.704` |
| predicted-length accuracy | `0.850` |
| duplicate sequence rate | `0.525` |

Per-length behavior:

| Length | Exact | Token | Duplicate |
| --- | ---: | ---: | ---: |
| 2 | `1.000` | `1.000` | `0.000` |
| 3 | `0.800` | `0.900` | `0.200` |
| 4 | `0.600` | `0.920` | `0.200` |
| 5 | `0.000` | `0.667` | `0.400` |
| 6 | `0.000` | `0.543` | `0.600` |
| 7 | `0.000` | `0.750` | `1.000` |
| 8 | `0.200` | `0.689` | `0.800` |
| 9 | `0.000` | `0.540` | `1.000` |

Serial-position shape:

| First | Middle | Last | Edge | U-score |
| --- | ---: | ---: | ---: | ---: |
| `0.900` | `0.586` | `0.675` | `0.787` | `0.202` |

Transposition gradient:

| Adjacent | Far | Adjacent fraction | Distance-dependent | Distance counts |
| ---: | ---: | ---: | --- | --- |
| `36` | `21` | `0.632` | `True` | `{"1":36,"2":8,"3":3,"4":2,"5":5,"6":2,"7":1}` |

Probe readout:

| Representation | Current item | Write position | Ordered identity | Length |
| --- | ---: | ---: | ---: | ---: |
| write trajectory `[h_t;c_t]` | `0.859` | `0.914` | n/a | n/a |
| final compressed `[h;c]` | n/a | n/a | `0.868` | `0.600` |

判定：

- 不是单纯失败：token accuracy `0.704`，短序列 preserved，final compressed `[h;c]` ordered identity probe `0.868`。
- 存在 graded / human-like 信号：first 高于 middle，last 有回升，U-score `0.202`；transposition distance-dependent 为 `True`，adjacent fraction `0.632`。
- 但不是干净的主 substrate 结论：U-score 弱于基线参考 `0.354`，adjacent fraction 弱于基线参考 `0.736`，且 long-length duplicate rate 偏高。

最终结论：**mixed / partially graded**。该 checkpoint 支持“压缩态 regime 值得继续作为候选 substrate”，但证据不足以直接淘汰 attention 对照并把 explicit-binding 压缩态定为主模型。下一步应先做判定实验 (b)：attention-lesion / blank-delay，再决定主 substrate。

## 9. 后续

1. 跑 attention-lesion：mask early memory states、final-state-only、shuffle memory-state order。
2. 跑 blank-delay / maintenance-only condition，区分 trajectory rereading 与 compressed maintenance。
3. 若 lesion 后 attention 显著坍向 final-state probe 水平，则 attention 更像 rereading 上界；若 explicit-binding 的 graded error 仍稳定，压缩态 regime 更适合作为 human-like substrate。
