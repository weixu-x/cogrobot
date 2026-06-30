# Corsi Memory Recall V2 阶段性结果汇报

日期：2026-06-30

范围：`corsi_memory_recall_v2` 主实验线，覆盖 V2 数据/模型建立、Stage 0 gates、Stage 1/2 训练、memory-isolate、LSTM binding 诊断、readout repair、explicit-binding 误差结构补测。

## 1. 一句话结论

V2 已经从“RGB 到动作/记忆链路是否可行”的验证，推进到“记忆表征与 recall readout 的机制分离”。当前最强结果是：LSTM write trajectory 中已经有足够的 item/order 信息；原始失败主要发生在 final-vector compression/readout 阶段。`memory_attention` 直接读取 per-step write states 后，在 full-data 上达到 full/token/length `1.000/1.000/1.000`。同时，explicit-binding 压缩态 checkpoint 呈现 partial graded / human-like 错误结构，仍是后续工作记忆 substrate 的候选。

## 2. 当前 V2 实验设置

| 项目 | 当前值 |
| --- | --- |
| 数据集 | `corsi_memory_recall_v2_k12` |
| schema | `scala_corsi_memory_recall_v2_canonical_v1` |
| fingerprint | `782962d47e55a1e4631919f4042bc1eb31ff0b66c7c27adff27aaf7b994b4485` |
| 输入 | segmented RGB `[B,L,K,3,128,128]` |
| 图像尺寸 | `128x128` |
| K | `12` frames per segment |
| block tokens | `0..8` |
| EOS token | `9` |
| sequence length | 2-9 |
| split | train/val/test = 320/40/40 |

当前没有可用的 V2 K=20 / K=30 正式结果。旧 motion-base K=20/K=30 artifacts 不能作为当前 V2 证据复用。

## 3. 实验推进逻辑

### 3.1 V2 scaffold 和 Stage 0 gates：先证明任务链路可行

最初 V2 目标是把任务从 motion prediction 转成 memory recall：模型看每个 block 的视觉片段，最终输出 block ID 序列 + EOS。

Stage 0 gates 证明四件事：

| Gate | 结果 | 说明 |
| --- | --- | --- |
| Motor shell reachability | pass, 9/9 | robot/control shell 可达，不是动作层基本失效 |
| Oracle recall length-9 | exact `1.0` | 如果给 oracle item memory，recall 上界可达 |
| Frozen probe from oracle `M` | token `1.0`, exact `1.0` | 诊断 memory 表征可被线性读出 |
| CNN/Visual/Motor tiny overfit | final loss `0.00147` | visual/motor encoder 至少可在小样本上拟合 |

这一步的意义：如果后面 recall 失败，优先怀疑 memory/readout，而不是 robot shell 或数据 schema。

### 3.2 原始 Stage 2 和 warm-start：能学长度，但不能精确 recall

早期 Stage 2 full training 的 full-sequence accuracy 为 `0.0`。Stage 1 warm-start 以后，token/length 有改善，但 full-sequence 仍为 `0.0`。

| Run | Val full | Val token | Val length | Test full | Test token | Test length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original Stage 2 scaffold | `0.000` | `0.204` | `0.250` | n/a | n/a | n/a |
| Stage 1 warm-start | `0.000` | `0.308` | `0.975` | `0.000` | `0.273` | `1.000` |

解释：

- Stage 1 pretraining 确实帮助了 grounding / length 相关信号。
- 但 full recall 没有恢复，说明问题不只是视觉编码或 pretraining 不足。
- 后续改动应集中在 memory 表征和 recall readout。

### 3.3 Order auxiliary 和 slot-compress：把 ordered identity 写进 final memory

为了判断 final memory 是否包含 ordered identity，后续做了 memory-isolate。

| Run / checkpoint | Val full | Val token | Val length | Test full | Test token | Test length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| orderaux balanced | `0.125` | `0.442` | `0.875` | `0.075` | `0.462` | `0.950` |
| slot-compress `D_mem=64`, best full | `0.325` | `0.627` | `0.875` | `0.300` | `0.631` | `0.850` |
| slot-compress `D_mem=64`, best val-loss | `0.250` | `0.677` | `0.975` | `0.350` | `0.654` | `0.875` |

Production final-memory probe for slot-compress `D_mem=64`:

| Checkpoint | Val order token | Test order token | Known-length exact |
| --- | ---: | ---: | ---: |
| best full | `0.882` | `0.877` | `0.675` |
| best val-loss | `0.886` | `0.895` | `0.675` |

这一步的结论：

- `slot_compress + D_mem=64` 明显改善 final memory 的 ordered identity。
- full-sequence 从 0 提升到约 `0.30-0.35`，说明 memory write 机制确实影响行为。
- 但是 long-sequence exact 仍然崩，duplicate 仍高，问题转移到 autonomous recall readout / duplicate suppression。

### 3.4 当前 slot-compress 现象：短序列解决，长序列 duplicate collapse

slot-compress best-full 的 per-length exact：

| Length | Val exact | Test exact |
| ---: | ---: | ---: |
| 2 | `1.000` | `1.000` |
| 3 | `0.800` | `1.000` |
| 4 | `0.400` | `0.400` |
| 5 | `0.200` | `0.000` |
| 6 | `0.200` | `0.000` |
| 7 | `0.000` | `0.000` |
| 8 | `0.000` | `0.000` |
| 9 | `0.000` | `0.000` |

slot-compress best-full 的 duplicate rate：

| Length | Val duplicate | Test duplicate |
| ---: | ---: | ---: |
| 2 | `0.000` | `0.000` |
| 3 | `0.000` | `0.000` |
| 4 | `0.200` | `0.200` |
| 5 | `0.400` | `0.800` |
| 6 | `0.600` | `0.800` |
| 7 | `1.000` | `0.800` |
| 8 | `1.000` | `1.000` |
| 9 | `1.000` | `1.000` |

解释：

- 模型不是完全不知道 block identity；token accuracy 和 set overlap 仍非零。
- full-sequence exact 严苛，长序列常因一两个位置、重复 token 或 EOS/length tradeoff 失败。
- 主要现象变成 duplicate collapse，而不是视觉表征失败。

### 3.4.1 Duplicate 输出合规层决策：采用推理时 masking（档 1）

当前对 duplicate 的决策是：baseline 阶段只采用 **inference-time output masking** 作为输出合规层，不在此阶段引入训练时 penalty / inhibition。

机制：

- autonomous decode 时维护“已输出 block”集合。
- 每一步 softmax 前，把已输出 block 的 logit 设为 `-inf`。
- 这样从数学上禁止同一序列内重复输出。

为什么选 masking，而不是训练时反重复 penalty：

- masking 是事后合规层，不改权重、不改 `final_memory`、不改 probe 表示。
- baseline 的认知内容应由 binding / compressed-memory 机制决定，而不是由工程性的反重复损失决定。
- 若此阶段加入训练时 penalty，后续无法区分 U-shape / transposition 是容量瓶颈自然涌现，还是 penalty 调出来的。
- masking 影响范围可控：duplicate rate 应接近 0；final-state order probe 理论上不变；token / U-shape / transposition 只会因重复出口被堵后重新落点而轻微变化。

预期影响：

| 指标 | 预期 | 原因 |
| --- | --- | --- |
| duplicate rate | `->≈0` | 已输出 block 被 mask，数学上不能重复 |
| final-state order probe | 不变 | probe 读 memory 表示，masking 不动表示 |
| token accuracy | 可能小幅上升 | 原来浪费在重复上的概率可能改判到次高 block |
| U-shape / transposition | 可能微变 | 被禁重复后的替代落点可能改变错误类型 |

关键分型：

| 观察 | 判定 | 含义 | 后续 |
| --- | --- | --- | --- |
| duplicate 降到约 0，但 token/full/U-shape 基本不变 | 情况 A | duplicate 只是底层读错后的出口之一；堵住重复后错误转为替换/错位 | duplicate 线收工，保留 masking 作合规层，baseline 校准回到 `D_mem` / 拐点 |
| duplicate rate 本身没降 | 情况 B | masking 未生效，是实现或评估路径 bug | 排查 autonomous decode 路径、checkpoint/eval 缓存、已选集合跨步累积 |

当前下一步：

1. 跑加 output-masking 的 autonomous decode 评估。
2. 并排比较修复前 / 后四个数：duplicate rate、token accuracy、U-score、final-order probe。
3. 若情况 A 成立，则不再把 duplicate 当成独立建模问题，转入 `D_mem` 拐点校准。
4. 若情况 B 成立，则先修 masking 实现，再重新分型。

后续预案：

- 档 1 masking 保留为 baseline 输出合规层。
- 只有在 thesis 写作或 ageing ablation 阶段，才考虑升级到 inhibition-of-return / output suppression（档 3）。
- 档 3 会进入学习动态并改变内部表示，因此不能在当前 baseline 锁定前使用。

### 3.5 LSTM onset / storage localization：write trajectory 有信息，final 压缩弱

LSTM `D_mem=64` 的后续诊断显示：

| Probe | 结果 |
| --- | ---: |
| item embedding block decoding | `1.000` |
| memory trajectory `[h_t;c_t]` current item | `0.982` |
| memory trajectory `[h_t;c_t]` write position | `0.959` |
| memory trajectory past-set exact | `0.995` |
| memory trajectory ordered-prefix token | `0.743` |
| memory trajectory ordered-prefix exact | `0.591` |
| final `[h;c]` ordered identity | about `0.58-0.60` |

行为上，原始 LSTM checkpoint：

| Length | Exact | Token | Duplicate |
| ---: | ---: | ---: | ---: |
| 2 | `1.000` | `1.000` | `0.000` |
| 3 | `1.000` | `1.000` | `0.000` |
| 4 | `0.600` | `0.920` | `0.400` |
| 5 | `0.000` | `0.767` | `1.000` |
| 6 | `0.200` | `0.543` | `0.800` |
| 7 | `0.000` | `0.650` | `1.000` |
| 8 | `0.000` | `0.689` | `1.000` |
| 9 | `0.000` | `0.460` | `1.000` |

这一步改变了问题定义：

原先怀疑是 item/order binding 在 write-in 阶段失败。probe 结果显示更准确的说法是：**per-step write trajectory 中已经有 item/order 信息，但 final-vector compression/readout 没有稳定暴露 ordered identity。**

因此下一步不应该先盲目扫 `D_mem`，而是先改 recall readout。

### 3.6 Memory attention：直接读 write trajectory 后达到满分

实现 `recall_readout_mode="memory_attention"` 后，recall 每一步直接 attend over memory write trajectory `[h_t;c_t]`，不再只依赖单一 final memory vector。

Full-data 结果：

| Run | Result |
| --- | --- |
| LSTM write + `memory_attention`, epoch 17 | full/token/length `1.000/1.000/1.000` |

Phase 3 probe：

| State | Current item | Write position |
| --- | ---: | ---: |
| `h_t` | `1.000` | `1.000` |
| `c_t` | `1.000` | `1.000` |
| `[h_t;c_t]` | `1.000` | `1.000` |

Per-length behavior:

| Length | Exact | Token | Duplicate |
| ---: | ---: | ---: | ---: |
| 2-9 | `1.000` | `1.000` | `0.000` |

关键 sanity check：

| Representation | Ordered identity | Length |
| --- | ---: | ---: |
| final compressed `[h;c]` in attention checkpoint | `0.441` | `1.000` |

解释：

- attention checkpoint 行为满分，但 final compressed `[h;c]` 仍不擅长 ordered identity。
- 所以它不是“压缩态 memory 已经完美”，而是“recall 能重读 per-step trajectory”。
- 这验证了 readout bottleneck 假说，也提供了性能上界/control condition。

### 3.7 Explicit binding + auxsplit：压缩态不是纯失败，而是 partial graded

为了判断 compressed/binding regime 是否仍值得作为 cognitive substrate，补测了 explicit-binding + auxsplit full-data checkpoint。

实际 checkpoint：

`stage2_seed0_item_context_binding_auxsplit_dmem64_20260629/best_full_sequence.pt`

selection metadata:

| Metric | Value |
| --- | ---: |
| epoch | `69` |
| validation full | `0.250` |
| validation token | `0.638` |
| validation length | `0.800` |

补测 test behavior：

| Metric | Value |
| --- | ---: |
| full-sequence | `0.325` |
| token | `0.704` |
| predicted length | `0.850` |
| duplicate sequence rate | `0.525` |

Serial-position shape：

| First | Middle | Last | Edge | U-score |
| --- | ---: | ---: | ---: | ---: |
| `0.900` | `0.586` | `0.675` | `0.787` | `0.202` |

Transposition gradient：

| Adjacent | Far | Adjacent fraction | Distance-dependent |
| ---: | ---: | ---: | --- |
| `36` | `21` | `0.632` | `True` |

Probe readout：

| Representation | Current item | Write position | Ordered identity | Length |
| --- | ---: | ---: | ---: | ---: |
| write trajectory `[h_t;c_t]` | `0.859` | `0.914` | n/a | n/a |
| final compressed `[h;c]` | n/a | n/a | `0.868` | `0.600` |

判定：

- 不是单纯失败：token `0.704`，短序列保留，final compressed ordered identity `0.868`。
- 有 graded / human-like 信号：U-score `0.202`，transposition distance-dependent `True`。
- 但还不够干净：U-score 和 adjacent fraction 弱于参考基线，长序列 duplicate 仍高。

结论：explicit-binding 压缩态是 **mixed / partially graded**。它仍是主 substrate 候选，但不能在没有 attention-lesion / blank-delay 对照前直接替代 attention result。

## 4. 结果链条：因为什么改进了什么

| 观察到的结果 / 失败 | 推动的改动 | 改动后的改善 | 新暴露的问题 |
| --- | --- | --- | --- |
| 原始 Stage 2 full = `0.0` | 加 Stage 1 warm-start | token / length 改善 | full recall 仍 `0.0` |
| final memory 不稳定保留 order | 加 memory-order aux / memory-isolates | orderaux full 到 `0.125` | 仍然弱 |
| final memory order probe 低 | 改 slot-compress `D_mem=64` | final order probe 到约 `0.88-0.90`; test full 到 `0.30-0.35` | 长序列 duplicate collapse |
| 长序列重复、token 非零但 exact 低 | 做 current-model audit / duplicate metrics | 明确失败是 recall/readout 层，而非视觉层 | 需要定位 memory/readout |
| LSTM trajectory 有 item/order，final weak | 改 `memory_attention` readout | full/token/length 到 `1.0` | 可能是 re-reading，不一定是 compressed WM |
| attention 满分但无错误结构 | 补测 explicit-binding checkpoint | 发现 partial graded + U-shape + transposition | 需要 attention-lesion / blank-delay 判定主 substrate |

## 5. 当前阶段性解释

当前最稳妥的解释是三层分离：

1. **视觉/segment grounding 已基本不是主瓶颈**：item embeddings 可完美 block-decode，Stage 0 gates 也通过。
2. **memory write 是否保留 ordered identity 很关键**：slot-compress 和 explicit-binding 都能显著提升 final-memory ordered probe。
3. **autonomous recall readout 是当前核心瓶颈**：即便 final memory 有 ordered identity，decoder 仍会在长序列上重复、替换或错位。`memory_attention` 通过直接读取 trajectory 修复 readout，证明 trajectory 里有足够信息。

因此，当前不是“模型还没学会看图”，也不是“单纯容量不够”。更接近：

> 信息已经在中间状态中存在，但不同 readout / compression regime 决定它是成为满分 recall、graded human-like error，还是 duplicate collapse。

## 6. 汇报时建议强调的主线

### 6.1 从工程结果看

- V2 pipeline 已经跑通：data schema、Stage 0、Stage 1/2 train/eval、probe、checkpoint selection 都可复用。
- 最强性能版本已经达到 full recall `1.0`。
- 失败分析已经从“黑箱准确率低”推进到具体机制：final compression vs trajectory readout。

### 6.2 从认知建模看

- `memory_attention` 是性能上界：证明信息足够，但可能像 re-reading trajectory。
- explicit-binding / slot-compress 是更像 compressed working memory 的候选：有部分错误结构、U-shape 和 distance-dependent transposition。
- 下一步 attention-lesion / blank-delay 可以直接检验 cross-attention 是否构成真正 WM maintenance。

### 6.3 从论文叙事看

可形成一条清晰论证：

1. 先建立 embodied Corsi memory recall task。
2. 原模型失败不是视觉失败，而是 memory/readout failure。
3. probing 显示 item/order 信息在 trajectory 中存在。
4. 改 readout 可达满分，说明任务可解且信息可用。
5. 压缩态模型产生 graded error，是 human-like WM substrate 候选。
6. 下一步用 lesion/delay 区分 re-reading 与 true maintenance。

## 7. 仍未完成 / 下一步

| Priority | 实验 | 目的 |
| --- | --- | --- |
| P0 | attention-lesion | mask early states / final-state-only / shuffle memory states，测试 attention 是否依赖 trajectory rereading |
| P0 | output-masking decode 评估 | 比较 duplicate/token/U-shape/final-order probe，判定情况 A 或 B |
| P0 | blank-delay / maintenance-only | 区分 trajectory access 与 compressed maintenance |
| P1 | explicit-binding regime 继续调优 | 若 masking 后仍需要，降低 long-length retrieval error，同时保留 U-shape / transposition structure |
| P1 | D_mem sweep | 只在 compressed regime 中做，用于解释容量曲线，而不是解释 attention 满分 |
| P1 | Stage 1 ablation | 量化 pretraining 是否必要以及多少足够 |
| P1 | K=20 / K=30 V2 ablation | 目前没有当前 V2 证据，不能引用旧 motion-base artifacts |

## 8. 可引用的主要报告

- `reports/v2_current_model_audit_20260629.md`
- `reports/v2_memory_isolates_snapshot_20260626.md`
- `reports/v2_lstm_associative_memory_binding_final_report_20260629.md`
- `reports/v2_lstm_binding_experiment_record_20260629.md`
- `reports/v2_lstm_phase2b_memory_attention_repair_20260629.md`
- `reports/binding_auxsplit_fulldata_phase1_binding_diagnostics.md`

## 9. 一页汇报版摘要

V2 Corsi memory recall 已经从基础可行性推进到机制定位。早期 Stage 2 和 warm-start 只能学到 token/length，full recall 仍为 0。随后通过 memory-isolate 发现，final memory 是否保留 ordered identity 是关键；`slot_compress + D_mem=64` 把 final-memory order probe 提升到约 `0.88-0.90`，并把 test full accuracy 提升到约 `0.30-0.35`，但长序列仍有 duplicate collapse。当前 duplicate 决策是先用 inference-time masking 作为输出合规层，不引入训练时 penalty，以免污染 capacity / U-shape 解释。进一步 LSTM probes 发现，write trajectory 中 current item 和 position 可线性解码，说明信息并未在 write-in 阶段完全丢失，而是在 final-vector compression/readout 阶段退化。基于此加入 `memory_attention` readout 后，full-data recall 达到 `1.000/1.000/1.000`。这说明 trajectory 中的信息足以完成任务，但 attention 更像性能上界或 re-reading control。与此同时，explicit-binding + auxsplit 压缩态 checkpoint 虽 full-sequence 不高，但 token `0.704`、U-score `0.202`、distance-dependent transposition 为真，说明它不是纯失败，而是 partial graded regime。下一步关键实验是 output-masking decode 分型、attention-lesion 和 blank-delay，用来决定主 cognitive substrate 应选 attention readout 还是压缩态 binding memory。
