# V2 LSTM Phase 2B Repair Hooks - 2026-06-29

Phase 1 result: current item and write position are both linearly decodable from LSTM write trajectory, but ordered identity degrades after compression into one final memory vector.

Implemented repair hooks:

- `CorsiMemoryRecallV2Config.memory_write_mode="item_context_binding"`
- explicit additive `item_embedding outer context[position]` write-in, projected to `final_memory`
- learned context vectors are independent of item identity
- `memory_identity_logits` current-item auxiliary head
- `memory_aux_orthogonal_loss`, penalizing shared input directions between order and identity aux heads
- `CorsiMemoryRecallV2Config.recall_readout_mode`
- default `final`, preserving existing behavior
- optional `memory_attention`, where each recall step attends over the memory write trajectory `[h_t;c_t]`
- forward/decode now expose `memory_states` and `memory_state_mask`
- checkpoint loading fills new binding/readout parameters from current initialization for old checkpoints

Configured isolate:

`corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_item_context_binding_dmem64.json`

`corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_item_context_binding_auxsplit_dmem64.json`

`corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_lstm_attention_dmem64.json`

Recommended explicit-binding run:

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_item_context_binding_dmem64.json \
  --stage 2 \
  --seed 0 \
  --output-root corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12 \
  --run-name stage2_seed0_item_context_binding_dmem64_20260629 \
  --max-epochs 160 \
  --allow-full-training
```

Recommended trajectory-attention run:

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_lstm_attention_dmem64.json \
  --stage 2 \
  --seed 0 \
  --output-root corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12 \
  --run-name stage2_seed0_lstm_attention_dmem64_20260629 \
  --max-epochs 160 \
  --allow-full-training
```

Recommended explicit-binding plus aux-split run:

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_item_context_binding_auxsplit_dmem64.json \
  --stage 2 \
  --seed 0 \
  --output-root corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12 \
  --run-name stage2_seed0_item_context_binding_auxsplit_dmem64_20260629 \
  --max-epochs 160 \
  --allow-full-training
```

Validation target:

- rerun Phase 1 probes on the new best-full checkpoint
- compare short-sequence exact, per-length duplicate rate, final-memory probe, and autonomous full-sequence accuracy
- if explicit binding improves final-memory ordered identity, keep it as the write-path repair
- if aux-split reduces the final order-vs-length gap without suppressing current-item probes, keep the orthogonalized auxiliary branch
- if trajectory attention improves long lengths without hurting length 2/3, keep it as the readout repair

Reusable Phase 3 probe command:

```bash
conda run -n robosuite python -B reports/run_v2_lstm_phase1_binding_diagnostics_20260629.py \
  --config <run-config.json> \
  --checkpoint <best-full-checkpoint.pt> \
  --output-json reports/<run-name>_phase1_binding_diagnostics.json \
  --output-md reports/<run-name>_phase1_binding_diagnostics.md \
  --device cpu
```

Current LSTM D_mem=64 baseline behavior sanity, from the reusable Phase 1 CLI:

- serial shape: first `1.000`, middle `0.521`, last `0.750`, edge `0.875`, U-score `0.354`
- transposition-like distance gradient: adjacent `53`, far `19`, adjacent fraction `0.736`, distance-dependent `True`

Smoke validation completed:

Explicit binding:

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_item_context_binding_dmem64.json \
  --stage 2 \
  --seed 0 \
  --output-root /tmp/corsi_memory_recall_v2_smoke \
  --run-name stage2_item_context_binding_smoke \
  --max-epochs 1 \
  --overfit-episodes 2
```

Result: completed on CPU, wrote `/tmp/corsi_memory_recall_v2_smoke/stage2_item_context_binding_smoke/best.pt`, and produced finite Stage 2 losses.

Explicit binding plus aux split:

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_item_context_binding_auxsplit_dmem64.json \
  --stage 2 \
  --seed 0 \
  --output-root /tmp/corsi_memory_recall_v2_smoke \
  --run-name stage2_item_context_binding_auxsplit_smoke \
  --max-epochs 1 \
  --overfit-episodes 2
```

Result: completed on CPU, wrote `/tmp/corsi_memory_recall_v2_smoke/stage2_item_context_binding_auxsplit_smoke/best.pt`, and produced finite `memory_identity_loss` and `memory_aux_orthogonal_loss` components.

Trajectory attention:

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_lstm_attention_dmem64.json \
  --stage 2 \
  --seed 0 \
  --output-root /tmp/corsi_memory_recall_v2_smoke \
  --run-name stage2_lstm_attention_smoke \
  --max-epochs 1 \
  --overfit-episodes 2
```

Result: completed on CPU, wrote `/tmp/corsi_memory_recall_v2_smoke/stage2_lstm_attention_smoke/best.pt`, and produced finite Stage 2 losses.

CUDA/sandbox diagnosis:

- Inside the managed sandbox, `conda run -n robosuite` reported `torch.cuda.is_available() == False` and `device_count == 0`; `/dev/nvidia0`, `/dev/nvidia1`, and `/dev/nvidiactl` were not visible.
- In the escalated/non-sandbox path, the same `robosuite` environment reported `torch.cuda.is_available() == True`, `device_count == 2`, and `device0 == NVIDIA TITAN Xp`; `/dev/nvidia*` device nodes were present.
- Conclusion: CUDA was available in `robosuite`; the earlier CUDA failure was caused by sandbox device-node isolation.

Final validation results:

- Explicit binding plus aux-split short gate, resumed to 80 epochs on CUDA, reached full/token/length accuracy `1.000` on the length-2/3 subset and duplicate sequence rate `0.000`.
- Full-data explicit binding plus aux-split was stopped after epoch 76 as a negative control; best full-sequence accuracy was `0.250`.
- Full-data LSTM write plus `memory_attention` reached best full/token/length accuracy `1.000/1.000/1.000` at epoch 17.
- Best checkpoint: `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_attention_dmem64_20260629/best_full_sequence.pt`.
- Phase 3 report: `reports/v2_lstm_attention_phase3_binding_diagnostics_20260629.md`.
- Follow-up experiment (a) on the explicit-binding plus aux-split epoch-69 `best_full_sequence.pt` shows mixed / partially graded error structure rather than simple failure: test full `0.325`, token `0.704`, length `0.850`, duplicate `0.525`, U-score `0.202`, adjacent transposition fraction `0.632`, distance-dependent `True`. Report: `reports/binding_auxsplit_fulldata_phase1_binding_diagnostics.md`.

Phase 3 readout:

- Write trajectory `[h_t;c_t]` linearly decodes current item `1.000` and write position `1.000` on test.
- Test behavior is solved for lengths 2 through 9: full-sequence accuracy `1.000`, token accuracy `1.000`, length accuracy `1.000`, duplicate sequence rate `0.000`.
- Final compressed `[h;c]` still has weak ordered identity probe accuracy (`0.441`) while final length probe is `1.000`.
- Interpretation: the original failure was not capacity squeeze and not absence of item/order information in the write trajectory. The decisive bottleneck was compressing ordered content into one final memory vector for recall. Memory-trajectory attention fixes the readout by making recall query the per-step write states directly.
- Human-like U-shape/transposition signals are absent in the solved checkpoint because the test error count is zero; those sanity checks are more useful for partially correct checkpoints.
