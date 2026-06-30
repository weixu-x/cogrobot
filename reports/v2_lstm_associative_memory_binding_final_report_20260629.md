# V2 LSTM Associative Memory Binding Failure: Diagnosis and Repair Report

Date: 2026-06-29

Experiment line: `corsi_memory_recall_v2` / `corsi_memory_recall_v2_k12`

Scope: Phase 1 diagnostics, Phase 2B repair hooks, CUDA/sandbox check, full-data validation, Phase 3 reprobe.

## 1. Executive Summary

The original V2 LSTM associative-memory failure was not best explained as a catastrophic write-in failure or a simple `D_mem` capacity squeeze. Low-cost probes showed that the memory write trajectory already contained both current item identity and write position. The failure appeared when ordered content had to be compressed into a single final memory vector and then used for recall.

The successful repair was `recall_readout_mode="memory_attention"`: recall queries the per-step memory write trajectory `[h_t;c_t]` directly instead of relying only on the final compressed memory vector. With the same D_mem=64 setting, the full-data LSTM-write plus memory-attention run reached full/token/length accuracy `1.000/1.000/1.000` at epoch 17.

The strongest mechanistic conclusion is therefore:

> Ordered item information exists in the write trajectory, but the original final-vector readout loses or fails to expose it. Memory-trajectory attention repairs recall by bypassing that final-state compression bottleneck.

## 2. Assessment of the Provided Summary

The supplied summary is largely consistent with the experiments. I would tighten four points.

First, "identity is completely written in" is too strong. The diagnostic proves current item is highly linearly decodable from write states, not that binding is noiseless, lossless, or stored in the exact form required by downstream recall. The safer wording is: item identity and write position are present in the write trajectory and are linearly recoverable.

Second, "not capacity squeeze" is supported for this failure mode, because the same D_mem=64 system succeeds when recall attends to the write trajectory, and length 2/3 are already solved by the baseline. This does not prove capacity is irrelevant for all future regimes; it says the observed collapse was not primarily caused by insufficient memory dimension.

Third, the winning attention model is a readout repair, not evidence that the single final compressed memory vector was repaired. In fact, the Phase 3 final-state probe remains weak for ordered identity.

Fourth, the human-like U-shape/transposition discussion should be treated as interpretation and future-analysis motivation. The solved attention checkpoint has zero test errors, so it cannot exhibit a meaningful transposition gradient. The partially correct compressed-state regimes are the right place to analyze human-like error signatures.

## 3. Experiments Run

| Step | Run / Probe | Purpose | Main result |
| --- | --- | --- | --- |
| CUDA sandbox check | sandbox vs escalated `conda run -n robosuite` probes | Determine whether CUDA failure was environment or sandbox related | CUDA works outside sandbox; sandbox hides `/dev/nvidia*` |
| Phase 1 probe | `reports/run_v2_lstm_phase1_binding_diagnostics_20260629.py` on original LSTM D_mem=64 checkpoint | Decode item/position from write states and order/length from final state | Trajectory contains item/order; final state is weak for order |
| Identity-frozen control | Replace all item embeddings by train-set mean | Test pure position dynamics | Write-position and final-length probes reach `1.000` |
| CPU smoke | explicit binding, auxsplit, attention configs | Check forward/loss/checkpoint integrity | All completed with finite losses |
| CUDA smoke | one-epoch CUDA training | Check training entrypoint on GPU | CUDA training works outside sandbox |
| Short gate | explicit binding + auxsplit, length 2/3 subset, 80 epochs | Test whether explicit binding can learn very short sequences | exact/token/length all `1.000`; duplicate `0.000` |
| Full-data negative control | explicit binding + auxsplit, stopped at epoch 76 | Test whether explicit binding scales directly | best full-sequence `0.250` |
| Full-data winning run | LSTM write + `memory_attention` readout | Test readout repair | best full/token/length `1.000/1.000/1.000` at epoch 17 |
| Phase 3 reprobe | Probe attention best checkpoint | Confirm item/position and per-length behavior | trajectory item/position `1.000/1.000`; length 2-9 solved |
| Regression tests | memory-recall V2 tests | Check code integration | `30 passed`; `git diff --check` passed |

## 4. Phase 1 Diagnostic Findings

Original checkpoint:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_160/best_full_sequence.pt`

Report:

`reports/v2_lstm_phase1_binding_diagnostics_20260629.md`

Key probe results on test:

| Representation | Current item | Write position | Ordered identity | Length |
| --- | ---: | ---: | ---: | ---: |
| write trajectory `[h_t;c_t]` | `0.982` | `0.959` | n/a | n/a |
| final compressed `[h;c]` | n/a | n/a | `0.595` | `0.675` |

Behavior by length in the original checkpoint:

| Length | Exact | Token | Duplicate |
| --- | ---: | ---: | ---: |
| 2 | `1.000` | `1.000` | `0.000` |
| 3 | `1.000` | `1.000` | `0.000` |
| 4 | `0.600` | `0.920` | `0.400` |
| 5 | `0.000` | `0.767` | `1.000` |
| 6 | `0.200` | `0.543` | `0.800` |
| 7 | `0.000` | `0.650` | `1.000` |
| 8 | `0.000` | `0.689` | `1.000` |
| 9 | `0.000` | `0.460` | `1.000` |

Decision:

- Short sequences are not collapsed.
- Write trajectory has item and position information.
- The final vector is substantially weaker for ordered identity.
- The most likely bottleneck is final-state compression/readout, not item write-in absence.

## 5. Repair Hooks Implemented

Implemented in:

- `corsi/experiments/corsi_memory_recall_v2/model.py`
- `corsi/experiments/corsi_memory_recall_v2/losses.py`
- `corsi/experiments/corsi_memory_recall_v2/train.py`
- `corsi/experiments/corsi_memory_recall_v2/analysis.py`

Repair hooks:

- `recall_readout_mode="memory_attention"`: recall attends over the memory write trajectory `[h_t;c_t]`.
- `memory_write_mode="item_context_binding"`: explicit additive `item_embedding outer context[position]` write-in, projected back to final memory.
- `memory_identity_logits`: current-item auxiliary head.
- `memory_aux_orthogonal_loss`: penalizes shared directions between order and identity auxiliary heads.
- `memory_states` and `memory_state_mask`: exposed for decode/readout and interventions.
- Checkpoint compatibility: old checkpoints initialize missing new parameters from current model initialization.

New configs:

- `corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_lstm_attention_dmem64.json`
- `corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_item_context_binding_dmem64.json`
- `corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12_item_context_binding_auxsplit_dmem64.json`

## 6. CUDA / Sandbox Result

CUDA was available in the `robosuite` conda environment. The earlier failure was caused by the managed sandbox hiding device nodes.

| Context | PyTorch CUDA | Device count | `/dev/nvidia*` |
| --- | --- | ---: | --- |
| Managed sandbox | unavailable | `0` | not visible |
| Escalated / non-sandbox path | available | `2` | visible |

Both visible devices were NVIDIA TITAN Xp GPUs. CUDA smoke training completed successfully once run outside the sandbox device-node restriction.

## 7. Final Training Results

| Model / setting | Data | Result |
| --- | --- | --- |
| explicit binding + auxsplit | length 2/3 short gate, 80 epochs | full/token/length `1.000`; duplicate `0.000` |
| explicit binding + auxsplit | full data, stopped at epoch 76 | val best full-sequence `0.250`; follow-up test probe full `0.325`, token `0.704` |
| LSTM write + memory attention | full data | best full/token/length `1.000/1.000/1.000` at epoch 17 |

Best checkpoint:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_attention_dmem64_20260629/best_full_sequence.pt`

The explicit-binding short-gate result shows the new binding path can learn the low-length subset. The full-data negative-control result did not immediately solve full autonomous recall, but a follow-up error-structure probe shows it is not a pure failure: token accuracy remains moderate-high, serial-position structure is U-shaped, and transposition errors are distance-dependent. By contrast, memory attention solved the full task quickly, which matches the Phase 1 diagnosis that the useful information already lives in the trajectory.

## 8. Phase 3 Reprobe of the Winning Checkpoint

Report:

`reports/v2_lstm_attention_phase3_binding_diagnostics_20260629.md`

Checkpoint epoch: 17

Write-step probes:

| State | Current item | Write position |
| --- | ---: | ---: |
| `h_t` | `1.000` | `1.000` |
| `c_t` | `1.000` | `1.000` |
| `[h_t;c_t]` | `1.000` | `1.000` |

Per-length behavior:

| Length | Exact | Token | Duplicate |
| --- | ---: | ---: | ---: |
| 2 | `1.000` | `1.000` | `0.000` |
| 3 | `1.000` | `1.000` | `0.000` |
| 4 | `1.000` | `1.000` | `0.000` |
| 5 | `1.000` | `1.000` | `0.000` |
| 6 | `1.000` | `1.000` | `0.000` |
| 7 | `1.000` | `1.000` | `0.000` |
| 8 | `1.000` | `1.000` | `0.000` |
| 9 | `1.000` | `1.000` | `0.000` |

Final-state probes still show weak ordered identity:

| State | Order token | Known-length exact | Length |
| --- | ---: | ---: | ---: |
| final `h` | `0.377` | `0.125` | `1.000` |
| final `c` | `0.477` | `0.175` | `1.000` |
| final `[h;c]` | `0.441` | `0.100` | `1.000` |

This is the decisive sanity check: behavior is solved even though the final compressed vector remains poor for ordered identity. Therefore the successful model is not solving the task by making final `[h;c]` a good compressed ordered memory. It is solving the task by letting recall directly access the trajectory states.

## 9. Decision Experiment (a): Explicit-Binding Error Structure

Purpose: decide whether the full-data explicit-binding + auxsplit checkpoint behind the val full-sequence `0.250` score is a simple failure or a graded compressed-memory regime with human-like error structure.

Checkpoint used:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_item_context_binding_auxsplit_dmem64_20260629/best_full_sequence.pt`

Selection metadata:

- checkpoint epoch: 69
- validation selection score: full `0.250`, token `0.638`, length `0.800`
- probe report: `reports/binding_auxsplit_fulldata_phase1_binding_diagnostics.md`

Behavior on the test split in the probe report:

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

Decision:

The `0.250` validation full-sequence score is not a simple collapse. The checkpoint has moderate-high token accuracy, clear short-sequence preservation, a nonzero U-shaped serial-position curve, and a distance-dependent transposition gradient. However, the U-score (`0.202`) and adjacent fraction (`0.632`) are weaker than the earlier baseline reference (`0.354` and `0.736`), and duplicate rate is still high on long sequences.

Final classification for experiment (a): **mixed / partially graded**. The explicit-binding compressed regime remains a viable substrate candidate, but it is not clean enough to select as the primary substrate without the complementary attention-lesion / blank-delay experiment. The next decision point should be experiment (b), not immediate structural redesign.

## 10. Interpretation

The original failure has three layers:

1. Per-step item and position information are available in LSTM write states.
2. Compressing all ordered content into a single final vector loses or hides ordered identity.
3. The original recall readout depends too heavily on that weak final vector and therefore collapses into length/counting and duplicate-prone behavior.

Memory attention fixes layer 3 by changing the recall interface. It does not by itself prove that the model learned a stronger compressed associative memory. In cognitive-modeling terms, the attention model is best treated as a high-performance readout/control condition. It establishes that the sensory/write trajectory contains enough information for perfect recall, but it may be less suitable as the primary substrate for capacity-limited human-like error patterns.

The compressed/final-state regimes remain important because they are where U-shaped serial-position effects and transposition-like gradients can appear. The solved attention checkpoint has no errors, so those behavioral signatures are absent by construction.

Experiment (a) confirms that the explicit-binding compressed regime is not merely failed optimization. It carries ordered identity in the final compressed state (`0.868` order-token probe) and produces graded error structure. Its remaining weakness is autonomous retrieval fidelity: length and duplicate control still degrade at longer lengths.

## 11. What Is Verified vs. Still Open

Verified:

- CUDA was available; the earlier failure was sandbox isolation.
- Current item and write position are linearly recoverable from the original write trajectory.
- Length 2/3 are solved before repair, arguing against a catastrophic short-sequence binding failure.
- The final compressed state is weak for ordered identity.
- Memory attention solves the full task at D_mem=64.
- Phase 3 confirms perfect per-length behavior and perfect trajectory probes in the winning checkpoint.
- Experiment (a) shows the explicit-binding + auxsplit full-data checkpoint has partial graded error structure rather than simple failure.

Not yet verified:

- Whether a D_mem sweep would reveal a secondary capacity cliff in compressed-state models.
- Whether lesioning attention access to early trajectory states collapses performance toward the final-state probe level.
- Whether a delayed/blank maintenance interval would separate "trajectory rereading" from "compressed memory maintenance."
- Whether the partially graded explicit-binding regime is strong enough to be the main substrate after attention-lesion / blank-delay controls.

## 12. Recommended Next Experiments

1. Run attention-lesion controls for the winning checkpoint:
   - mask early memory states,
   - keep only final memory state,
   - shuffle memory-state order,
   - compare performance against final-state probe accuracy.

2. Add a blank-delay / maintenance-only condition. If attention still has access to trajectory states, it may behave like replay or rereading. If trajectory is removed and performance drops, that separates readout access from true compressed maintenance.

3. Only after those checks, decide whether a `D_mem` sweep is theoretically useful. The current evidence says D_mem was not the primary cause of the original readout failure, but capacity may still matter for deliberately compressed memory regimes.

## 13. Artifacts

Primary reports:

- `reports/v2_lstm_phase1_binding_diagnostics_20260629.md`
- `reports/v2_lstm_attention_phase3_binding_diagnostics_20260629.md`
- `reports/v2_lstm_phase2b_memory_attention_repair_20260629.md`
- `reports/v2_lstm_binding_experiment_record_20260629.md`
- `reports/binding_auxsplit_fulldata_phase1_binding_diagnostics.md`

Primary diagnostic script:

- `reports/run_v2_lstm_phase1_binding_diagnostics_20260629.py`

Winning run:

- `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_attention_dmem64_20260629/`

Best checkpoint:

- `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_attention_dmem64_20260629/best_full_sequence.pt`

Regression validation:

- `tests/test_corsi_memory_recall_v2_model.py`
- `tests/test_corsi_memory_recall_v2_train_eval.py`
- `tests/test_corsi_memory_recall_v2_integration.py`
- Result: `30 passed`
