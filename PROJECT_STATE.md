# CogRobot Project State

Last updated: 2026-06-30
Updated by: user / Codex
Current branch: `codex/corsi-motion`
Remote branch: `origin/codex/corsi-motion`

Current working state:
- V2 LSTM associative-memory binding diagnosis and readout repair completed on 2026-06-29.
- Canonical final report: `reports/v2_lstm_associative_memory_binding_final_report_20260629.md`.
- Binding experiment record: `reports/v2_lstm_binding_experiment_record_20260629.md`.
- Winning full-data repair is LSTM write + `recall_readout_mode="memory_attention"` at `D_mem=64`, best full/token/length `1.000/1.000/1.000` at epoch 17.
- Main interpretation: ordered item information is present in the write trajectory, but the original final-vector readout loses or fails to expose it; memory-trajectory attention repairs recall by bypassing final-state compression.
- Experiment (a) on explicit-binding + auxsplit full-data checkpoint is mixed / partially graded, not simple failure: test token `0.704`, U-score `0.202`, distance-dependent transposition `True`, duplicate sequence rate `0.525`.
- Duplicate handling decision recorded on 2026-06-30: use inference-time output masking as baseline compliance layer; do not use training-time anti-repeat penalty / inhibition at this stage.
- Generated outputs are under ignored `corsi_artifacts/`.
- Local working tree and true remote state must be checked at the start of any commit / ops task.

Primary objective:
- Treat `memory_attention` as the current solved readout/control condition for V2 memory recall.
- Next model-design/eval target is output-masking decode A/B classification, then attention-lesion / blank-delay; not upstream RGB / Visual / Motor encoding.
- Commit reviewed V2 memory-recall changes only after scope and validation are confirmed.

---

## 0. How To Use This File

- This file is the canonical project ledger and operating handoff for `/home/wei2025/Developer/cogrobot`.
- Read this file at the start of every new Codex thread or agent run.
- Keep this file short. Link to detailed reports instead of pasting long logs.
- If a fact is uncertain, write `UNKNOWN - needs audit`.
- Do not treat old data, old code, checkpoints, reports, or artifacts as valid unless listed as `ACTIVE`.
- Update only the changed rows at the end of each task.
- Do not rewrite historical results unless a new audit explicitly supersedes them.

---

## 0.1 Repository Operating Rules

### Environment

- Use the conda environment `robosuite`.
- Prefer commands of the form:

```bash
conda run -n robosuite python -B -m ...
```

### Change discipline

- Keep changes incremental.
- Separate raw data generation from training / dataset-loader work.
- Do not run raw generation, full training, cleanup deletion, commit, or push unless explicitly requested.
- Do not build new work on deprecated artifacts unless this file explicitly reactivates them.
- Do not assume generated local artifacts exist in another workspace.
- Do not infer local checkpoint / eval results from GitHub-only files unless the artifact path and metrics are recorded here or in a report.

### Protected files / workflows

Do not blindly overwrite or remove:

- `AUTHORS`
- `requirements.txt`
- `.vscode/settings.json`
- local `corsi_artifacts/`

Preserve existing workflows unless an explicit task says otherwise:

- `freecam_index`
- heatmap workflows
- `freecam_ee_xy` workflows

### Generated artifacts

Treat `corsi_artifacts/` as local generated data. It is ignored by Git.

Do not commit generated artifacts, caches, videos, checkpoints, large logs, or temporary outputs unless explicitly approved.

Common do-not-commit paths / patterns:

- `corsi_artifacts/`
- `__pycache__/`
- `.pytest_cache/`
- `.DS_Store`
- `robosuite.egg-info/`
- `robosuite/savevideo/`
- temporary PDFs / logs / scratch outputs
- checkpoints and large evaluation dumps unless explicitly requested

### Reporting requirements

At the end of each task, report:

- files inspected
- files changed
- commands run
- tests passed / failed / not run
- artifacts produced
- unresolved unknowns
- whether this file needs updating

---

## 1. Current Active Lanes

| Lane | Status | Purpose | Current task | Stop condition |
|---|---|---|---|---|
| data | ACTIVE | Dataset/schema loading and validation | V2 data/schema lane implemented and Gate 1/2 validation passed | Dataset status is clear and validator/smoke result is recorded |
| replay | UNKNOWN - needs eval | qpos / controller replay validation | FK posthoc pass and renderer-threshold failure are report-recorded; needs dedicated eval audit | Replay metrics are machine-readable and decision is recorded |
| model | ACTIVE | Model/training work | V2 LSTM binding failure diagnosed and memory-attention readout repair validated. Attention solves full-data recall at `D_mem=64`; compressed final-state ordered identity remains weak, so maintenance/lesion analysis is still open | Binding report and follow-up decisions are recorded |
| cleanup | PARKED | Remove obsolete files/artifacts safely | Deprecated scratch candidates identified; do not delete without approval | Deletion candidates are reviewed before removal |
| commit | PARKED | Review, commit, push | Only after implementation/eval is complete | Working tree scope and tests are confirmed |

---

## 2. Branch / Git State

| Item | Value |
|---|---|
| Current branch | `codex/corsi-motion` |
| Remote tracking branch | `origin/codex/corsi-motion` |
| Last recorded pushed commit | Local upstream ref was recorded as matching HEAD `6f7419c848c278c462ffca6768d105c303f417bb` (`Add Corsi 7-joint motion baseline`); remote was not fetched in that audit |
| Current git truth | UNKNOWN - needs audit before any commit / push |
| Dirty tracked files, last recorded | `corsi/experiments/corsi_memory_recall_v2/train.py`; `evaluate.py`; `run_suite.py`; `model.py`; `losses.py`; V2 train/eval/model/integration tests; `PROJECT_STATE.md` |
| Important untracked files, last recorded | `corsi/experiments/corsi_memory_recall_v2/stage0.py`; `tests/test_corsi_memory_recall_v2_stage0.py`; `reports/v2_memory_isolates_snapshot_20260626.md`; generated outputs ignored under `corsi_artifacts/` |
| Ignored/generated artifacts | `.pytest_cache/`, `corsi_artifacts/`, `robosuite.egg-info/`, `__pycache__/`, `robosuite/savevideo/` |
| Do not commit | `__pycache__`, `.DS_Store`, large generated artifacts, temporary outputs, accidental PDF/log/cache files |

Required preflight for any ops / commit task:

```bash
git status --short --branch
git rev-parse HEAD
git log -1 --oneline
```

Fetch remote only if the task explicitly allows network / remote operations.

---

## 3. Data Registry

| Dataset / Artifact | Status | Path | Schema / Format | Purpose | Notes |
|---|---|---|---|---|---|
| `corsi_motion_raw_len2_9_n50` | ACTIVE | `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50` | `scala_corsi_motion_raw_v1`; 400 episodes; lengths 2-9 x 50; camera `freecam`; arrays include `joint [T,7]`, `action [T,12]`, `qpos/qvel [T,19]` | Raw input for current 7-joint motion baseline and V2 canonicalization | Present and manifest/sample verified in prior audit |
| `corsi_motion_7joint_k12` | ACTIVE | `corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12` | `scala_corsi_motion_canonical_7joint_v1`; `joint_dim=7`; `K=12`; splits train/val/test 320/40/40 | Canonical dataset for current 7-joint motion baseline | Raw manifest hash matches source manifest |
| `corsi_memory_recall_v2_k12` | ACTIVE | `corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12` | `scala_corsi_memory_recall_v2_canonical_v1`; segmented RGB `[L,K,3,128,128]`; block+EOS targets; splits train/val/test 320/40/40 | Canonical dataset for Model V2 memory-recall lane | Fingerprint `782962d47e55a1e4631919f4042bc1eb31ff0b66c7c27adff27aaf7b994b4485`; Gate 1/2 validation, full Stage 1/2 seed-0 training/eval, warm-start rerun, final-memory slot-compress diagnostics, and Stage 0 gates completed |
| motion baseline runs/posthoc/visualization outputs | PARKED | `corsi_artifacts/motion_baseline/` | generated run, posthoc, and visualization outputs | Referenced by reports/code; model lane parked | Preserve; not scratch |
| visual-base `freecam_index`, heatmap, and `freecam_ee_xy` workflows | PARKED | `corsi_artifacts/` and visual-base paths | legacy visual/coordinate artifacts | Preserve per project rules | Not current motion-baseline input |
| `freecam_motion_segment_uniform_len3_k15` | DEPRECATED | `corsi_artifacts/motion_base/datasets/freecam_motion_segment_uniform_len3_k15` | `freecam_motion_v1_segment_uniform` | old qpos knot dataset, K=15 | Dataset root missing; only historical summaries found |
| `freecam_motion_segment_uniform_len3_k20` | DEPRECATED | `corsi_artifacts/motion_base/datasets/freecam_motion_segment_uniform_len3_k20` | `freecam_motion_v1_segment_uniform` | old qpos knot dataset, K=20 | Dataset root missing; only historical summaries found |
| `freecam_motion_segment_uniform_len3_k30` | DEPRECATED | `corsi_artifacts/motion_base/datasets/freecam_motion_segment_uniform_len3_k30` | `freecam_motion_v1_segment_uniform` | old qpos knot dataset, K=30 | Dataset root missing; only historical summaries found |
| motion_base comparison/demo result dirs | DEPRECATED | `corsi_artifacts/motion_base/results/*` | small historical JSON/plot outputs, about 1.9M total | Do not build new work on this | Review before any deletion |

Status vocabulary:

- `ACTIVE`: current valid source for ongoing work.
- `PARKED`: potentially useful later, not part of the current lane.
- `DEPRECATED`: do not build new work on it.
- `DO-NOT-DELETE`: preserve unless explicitly approved.
- `UNKNOWN - needs audit`: do not assume; verify first.

---

## 3.1 Raw Motion Dataset Notes

For `corsi_motion_raw_len2_9_n50`:

| Item | Value |
|---|---|
| Root | `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50` |
| Manifest | `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50/manifest.json` |
| Approximate size | about `1.1G` |
| Episodes | `400` |
| Lengths | `2` through `9` |
| Count per length | `50` |
| Sequence rule | ordered `block_order`, no repeated block within one episode |
| Failed/skipped episodes at validation time | none, unless later audit supersedes this |

Each episode directory contains:

- `arrays.npz`
  - synchronized per-frame arrays:
    - `rgb`
    - `joint`
    - `joint_velocity`
    - `ee_pose`
    - `ee_xy`
    - `action`
    - `qpos`
    - `qvel`
    - `timestamp`
    - `rank`
    - `block_id`
- `metadata.json`
  - episode metadata
  - block positions
  - normalization
  - control settings
  - segment copy
- `segments.json`
  - one segment per visited block
  - fields:
    - `segment_id`
    - `rank`
    - `block_id`
    - `start_frame`
    - `end_frame`

Raw generation code:

- `configs/corsi_motion_raw_len2_9_n50.yaml`
- `corsi/data/motion_raw.py`
- `corsi/data/generate_raw.py`
- `corsi/envs/robosuite_corsi.py`
  - seed support
  - `use_camera_obs` support
- `collect_block_positions` helper in:
  - `corsi/experiments/visual_base/scripts/export_robosuite_ee_xy_dataset.py`

Manual generation / resume command:

```bash
conda run -n robosuite python -m corsi.data.generate_raw \
  --config configs/corsi_motion_raw_len2_9_n50.yaml
```

Do not run raw generation without explicit approval.

The raw-only cleanup intentionally does not keep separate `generate_plan`, `validate_raw`, or raw test files. They were useful during generation but were removed to keep the submitted code focused on generation.

---

## 4. Code Map

| Area | Status | Main files / dirs | Purpose | Notes |
|---|---|---|---|---|
| project state / operating handoff | ACTIVE | `PROJECT_STATE.md` | Single canonical project ledger and operating handoff | Read first in every agent run |
| raw data generation | ACTIVE | `configs/corsi_motion_raw_len2_9_n50.yaml`; `corsi/data/motion_raw.py`; `corsi/data/generate_raw.py`; `corsi/envs/robosuite_corsi.py` | Raw SCALA Corsi motion dataset generation | Do not run generation without explicit approval |
| dataset loading | ACTIVE | `corsi/experiments/corsi_motion_baseline/dataset.py`; `corsi/data/motion_raw.py`; `corsi/data/generate_raw.py` | Canonical 7-joint loader and raw generator | Keep separate from V2 dataset lane |
| collate / batching | ACTIVE | `corsi/experiments/corsi_motion_baseline/dataset.py`; `corsi/data/collate.py`; `corsi/data/collate_visual.py` | Motion collator plus older coord/visual collators | Current baseline uses `collate_motion_prediction_batch` |
| replay / controller | UNKNOWN - needs eval | `corsi/envs/robosuite_corsi.py`; `corsi/scripts/run_robosuite_corsi.py`; visual export scripts | Robosuite Corsi env, qpos collection, rollout/export helpers | FK posthoc passed in reports; renderer Tier-B threshold failed |
| motion baseline model / training | PARKED | `corsi/experiments/corsi_motion_baseline/model.py`; `corsi/experiments/corsi_motion_baseline/train.py`; `corsi/experiments/corsi_motion_baseline/run_suite.py` | 7-joint motion predictor and training suite | Do not train unless explicitly requested |
| motion baseline evaluation / metrics | ACTIVE | `corsi/experiments/corsi_motion_baseline/evaluate.py`; `posthoc_suite.py`; `visualize_predictions.py`; `corsi/analysis/*.py`; `tests/test_corsi_motion_*.py` | Metrics, posthoc summaries, visualization, tests | Not the primary V2 lane |
| V2 memory recall | ACTIVE | `corsi/experiments/corsi_memory_recall_v2/`; `tests/test_corsi_memory_recall_v2_*.py`; `reports/model_v2_plan.md`; `reports/v2_memory_isolates_snapshot_20260626.md`; `reports/v2_lstm_associative_memory_binding_final_report_20260629.md` | V2 segmented RGB memory-recall model, training/eval, Stage 0 gates, memory-isolate diagnostics, and LSTM binding/readout repair | Current solved readout/control checkpoint is LSTM write + `memory_attention` `D_mem=64`; compressed-memory maintenance remains an open analysis target |
| reports / handoff | ACTIVE | `reports/model_v1_snapshot.md`; `reports/corsi_motion_current_handoff.md`; baseline audit/final/convergence/visualization reports | Detailed phase records | `model_v1_snapshot.md` is the frozen V1 reference |

---

## 5. Known Decisions

| Date | Decision | Reason | Affected files/data |
|---|---|---|---|
| 2026-06-23 | Treat this file as a ledger, not a report | New agent threads need fast orientation without replaying long history | `PROJECT_STATE.md` |
| 2026-06-23 | Do not mix OSC action semantics with qpos target semantics | OSC `arm_action_t` and joint `qpos_t` have different meanings and dimensions | dataset/replay/model lanes |
| 2026-06-23 | Do not start model/training work until data and replay semantics are explicit | Prevent training on ambiguous targets or obsolete data paths | model lane |
| 2026-06-23 | Current motion line is the 7-joint Panda arm baseline | Raw audit and canonical manifest verify `joint_dim=7`; old 6-joint / OSC-action line is not current | `corsi_motion_raw_len2_9_n50`, `corsi_motion_7joint_k12` |
| 2026-06-23 | Final checkpoint selection is validation-RMSE only | Test metrics are read afterward in reports | motion-baseline reports/runs |
| 2026-06-23 | Reported motion-baseline metrics are motion-prediction metrics | They are not closed-loop robot success or Corsi working-memory evidence | motion-baseline reports |
| 2026-06-24 | Model V1 snapshot recorded | Next model version will change structure, so V1 is frozen for comparison | `reports/model_v1_snapshot.md` |
| 2026-06-24 | Model V2 plan recorded | Defines V2 as RGB-segment memory recall with explicit V1 comparison, lane ownership, validation gates, and no full training before smoke passes | `reports/model_v2_plan.md` |
| 2026-06-24 | V2 data/schema lane implemented | Provides separate V2 canonical schema, loader/collate, validators, and generated canonical artifact without modifying V1 artifacts | `corsi/experiments/corsi_memory_recall_v2/`; `tests/test_corsi_memory_recall_v2_dataset.py`; `corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12` |
| 2026-06-24 | V2 model/loss and train/eval lanes integrated | Lane C/D outputs were merged through a Lane F compatibility pass; tests cover batch -> model -> loss/eval; tiny Stage 1/2 real-data smoke passed without full training | `corsi/experiments/corsi_memory_recall_v2/model.py`; `losses.py`; `train.py`; `evaluate.py`; `run_suite.py`; `analysis.py`; `extract_states.py`; V2 tests |
| 2026-06-24 | V2 full Stage 1/2 seed-0 training completed | Stage 2 needed a local eval padding fix for batch-local target lengths; best-checkpoint selection remained val full-sequence accuracy, so `best.pt` stayed epoch 0 because all epochs had full-sequence accuracy 0.0 | `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage1_seed0_full_20260624`; `.../stage2_seed0_full_20260624`; `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/` |
| 2026-06-24 | V2 Stage 2 warm-start full rerun completed | Stage 2 loaded 30 Stage 1 grounding keys from epoch-27 `best.pt`; warm-start improved val token accuracy to 0.3077 but full-sequence accuracy remained 0.0, so recall still fails exact sequence recovery | `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_warmstart_full_20260624`; `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_warmstart_full_20260624_*` |
| 2026-06-24 | V2 final-memory ordered-identity localization completed | Segment/item states are block-decodable, but final memory mostly keeps length rather than ordered identity. `D_mem=64` improved short exact only slightly and did not solve order probe. Ordered auxiliary prefix loss helped short sequences; balanced serial-position weighting was the best isolate with val full 0.125, test full 0.075, val length-2 exact 0.8, length-3 exact 0.2, but final_memory order probe stayed only val/test 0.35/0.33 and duplicate rate stayed 0.8. A direct final-memory ordered head/loss worsened behavior and is disabled by default (`memory_order_final=0.0`). | `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_*_minimal_localization.json`; `..._orderaux_balanced_full_20260624_*_eval.json`; `..._orderaux_finalmem_full_20260624_*_eval.json`; `corsi/experiments/corsi_memory_recall_v2/losses.py`; `model.py` |
| 2026-06-24 | Slot-compress `D_mem=64` restores ordered identity in final memory | `stage2_seed0_slot_compress_dmem64_full_20260624` shows the failure was mechanism plus capacity, not upstream encoding. Best full checkpoint: val/test full 0.325/0.300, token 0.627/0.631, EOS 0.875/0.900, length 0.875/0.850, per-length exact val length2/3/4 = 1.0/0.8/0.4 and test length2/3/4 = 1.0/1.0/0.4. Frozen final-memory order probe is val/test 0.882/0.877 for best-full and 0.886/0.895 for best-val-loss; known-length exact probe is 0.675. Duplicate rate drops from about 0.93-0.97 to about 0.50-0.58 but is still too high for long sequences. | `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624`; `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_*`; `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_slot_compress_dmem64.json`; `model.py`; `losses.py`; `train.py` |
| 2026-06-26 | V2 memory-isolates closeout snapshot recorded | Captures current branch/worktree state, Stage 0 integration decision, slot-compress `D_mem=64` result, duplicate-collapse risk, and validation commands | `reports/v2_memory_isolates_snapshot_20260626.md`; `PROJECT_STATE.md` |
| 2026-06-26 | Stage 0 gates integrated into main | Stage 0-specific code/test from a separate worktree were integrated, while that worktree's older project-state edit was not copied. Main Stage 0 run passed all four gates and wrote ignored artifacts. | `corsi/experiments/corsi_memory_recall_v2/stage0.py`; `tests/test_corsi_memory_recall_v2_stage0.py`; `corsi_artifacts/memory_recall_v2/stage0/summary.json` |
| 2026-06-29 | Make this file the single canonical operating handoff and project ledger | Future prompts will explicitly load this file; stable operating rules, protected-file rules, generated-artifact rules, raw dataset notes, and thread prompts are consolidated here to avoid split handoff state | `PROJECT_STATE.md` |
| 2026-06-29 | V2 current-model audit completed in main checkout | Main-checkout artifacts were used to fill complete length 2-9 exact/token/duplicate metrics for the slot-compress `D_mem=64` best-full checkpoint. Lightweight evaluator/loss instrumentation now exposes loss components, per-length token metrics, duplicate metrics, and set-overlap metrics for future eval outputs. | `reports/v2_current_model_audit_20260629.md`; `reports/v2_failure_analysis_20260629.md`; `reports/v2_duplicate_collapse_diagnostic_20260629.md`; `reports/v2_k_ablation_plan_20260629.md`; `reports/v2_loss_output_ablation_plan_20260629.md`; `corsi/experiments/corsi_memory_recall_v2/analysis.py`; `evaluate.py`; `losses.py`; `train.py`; V2 tests |
| 2026-06-29 | LSTM onset no-training diagnostics completed | Frozen-model probes show LSTM `D_mem=64` final memory is still weak on held-out ordered identity: epoch-139 best-full test final-memory order probe is 0.555 and known-length exact is 0.200. No-repeat decoding gives only a modest test full gain 0.350 to 0.400; oracle length alone gives no full-accuracy gain. Slot-compress remains the stronger memory-write mechanism for ordered identity, while LSTM duplicate/content readout also remains incomplete. | `reports/run_v2_lstm_no_training_diagnostics_20260629.py`; `reports/v2_lstm_no_training_diagnostics_20260629.json`; `reports/v2_lstm_no_training_diagnostics_20260629.md`; `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629` |
| 2026-06-29 | LSTM storage-localization probes completed | Item embeddings are perfectly block-decodable on test (1.000), so presentation/segment grounding is not the bottleneck. For epoch-139 best-full, memory trajectory `[h_t;c_t]` predicts current item 0.982 and past set 0.995 exact, but ordered prefix is only 0.743 token / 0.591 exact. Final `c` is stronger than final `h` (0.573 vs 0.545 order token) and final `[h;c]` is 0.582; flattened all-state probes are stronger (all-`h` 0.764, all-`[h;c]` 0.723). This supports attention/readout over memory trajectory and/or slot-compress/hybrid storage rather than revisiting visual grounding. | `reports/run_v2_lstm_storage_localization_20260629.py`; `reports/v2_lstm_storage_localization_20260629.json`; `reports/v2_lstm_storage_localization_20260629.md`; `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_memory_dmem64_onset_20260629` |
| 2026-06-29 | V2 LSTM associative-memory binding final report recorded | Phase 1/3 probes show item and position are linearly recoverable from the LSTM write trajectory, while ordered identity remains weak in the final compressed state. The observed failure is therefore a final-state/readout compression bottleneck rather than catastrophic write failure or primary `D_mem` squeeze. Full-data LSTM write + `memory_attention` reaches full/token/length `1.000/1.000/1.000` at epoch 17; final compressed `[h;c]` order probe remains weak at 0.441, so this is a readout repair/control condition, not proof that compressed maintenance is solved. CUDA is available in `robosuite`; earlier CUDA failures were sandbox device-node isolation. | `reports/v2_lstm_associative_memory_binding_final_report_20260629.md`; `reports/v2_lstm_attention_phase3_binding_diagnostics_20260629.md`; `reports/v2_lstm_phase2b_memory_attention_repair_20260629.md`; `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_lstm_attention_dmem64_20260629/best_full_sequence.pt` |
| 2026-06-29 | Binding experiment (a) completed for explicit-binding + auxsplit full-data checkpoint | Used epoch-69 `best_full_sequence.pt` from `stage2_seed0_item_context_binding_auxsplit_dmem64_20260629`. Probe test metrics: full `0.325`, token `0.704`, length `0.850`, duplicate `0.525`; serial shape first/middle/last `0.900/0.586/0.675`, U-score `0.202`; transposition adjacent/far `36/21`, adjacent fraction `0.632`, distance-dependent `True`; final `[h;c]` ordered identity probe `0.868`. Conclusion: mixed / partially graded, not simple failure. Keep compressed binding regime as a substrate candidate, but run attention-lesion / blank-delay before selecting the primary regime. | `reports/binding_auxsplit_fulldata_phase1_binding_diagnostics.md`; `reports/binding_auxsplit_fulldata_phase1_binding_diagnostics.json`; `reports/v2_lstm_binding_experiment_record_20260629.md`; `reports/v2_lstm_associative_memory_binding_final_report_20260629.md` |
| 2026-06-30 | Duplicate output compliance decision recorded | Adopt inference-time output masking as the baseline compliance layer: maintain the set of already emitted block IDs during autonomous decode and set their logits to `-inf` before softmax. Do not add training-time anti-repeat penalty / inhibition now, because it would change learned representations and confound whether U-shape / transposition comes from capacity limits or engineered loss shaping. Next eval must compare pre/post masking duplicate rate, token accuracy, U-score, and final-order probe. If duplicate drops but other metrics are stable, treat duplicate as an output-legality issue and return baseline calibration to `D_mem` / breakpoint. If duplicate does not drop, treat it as a masking implementation/eval-path bug. | `reports/v2_stage_results_summary_20260630.md`; future output-masking eval report TBD |

---

## 6. Current Validation Commands

### Fast git/status preflight

```bash
git status --short --branch
git rev-parse HEAD
git log -1 --oneline
```

### Dataset/schema check

```bash
conda run -n robosuite python -B -m corsi.data.generate_raw --help

conda run -n robosuite python -B -m corsi.experiments.corsi_motion_baseline.canonicalize --help

conda run -n robosuite python -B -c "from corsi.experiments.corsi_motion_baseline.dataset import CorsiMotionCanonicalDataset, collate_motion_prediction_batch; from corsi.experiments.corsi_motion_baseline.model import build_model; print('ok')"
```

### V2 data/schema Gate 1/2

```bash
conda run -n robosuite python -B -m pytest tests/test_corsi_memory_recall_v2_dataset.py -q

conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.canonicalize \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json \
  --overwrite

conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.validate \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json \
  --check all
```

### V2 model/train/eval Gate 3/4

```bash
conda run -n robosuite python -B -m pytest \
  tests/test_corsi_memory_recall_v2_dataset.py \
  tests/test_corsi_memory_recall_v2_model.py \
  tests/test_corsi_memory_recall_v2_train_eval.py \
  tests/test_corsi_memory_recall_v2_integration.py \
  tests/test_corsi_memory_recall_v2_stage0.py \
  -q

conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train --help

conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.evaluate --help

conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.run_suite --help

conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.extract_states --help

conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.stage0 --help
```

### Tiny smoke runs only

Use only when the task explicitly allows smoke training.

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json \
  --stage 2 \
  --device cpu \
  --output-root /tmp/corsi_v2_smoke_runs \
  --run-name stage2_overfit1_smoke \
  --max-epochs 1 \
  --overfit-episodes 1

conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json \
  --stage 1 \
  --device cpu \
  --output-root /tmp/corsi_v2_smoke_runs \
  --run-name stage1_overfit1_smoke \
  --max-epochs 1 \
  --overfit-episodes 1
```

### Raw generation command

Manual command only. Do not run without explicit approval.

```bash
conda run -n robosuite python -m corsi.data.generate_raw \
  --config configs/corsi_motion_raw_len2_9_n50.yaml
```

### Replay check

```bash
# No current replay command is marked safe/passing.
# Run replay only in an explicit eval lane.
```

### Core motion-baseline smoke / help checks

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_motion_baseline.evaluate --help

conda run -n robosuite python -B -m corsi.experiments.corsi_motion_baseline.posthoc_suite --help

conda run -n robosuite python -B -m corsi.experiments.corsi_motion_baseline.visualize_predictions --help
```

### Historical small checks

These were used after raw-generation cleanup. Do not treat them as sufficient current validation for V2.

```bash
conda run -n robosuite python -m pytest \
  tests/test_corsi_motion_baseline.py \
  tests/test_corsi_motion_posthoc.py \
  -q

conda run -n robosuite python -m corsi.data.generate_raw --help

conda run -n robosuite python -m pytest \
  tests/test_corsi_heatmaps.py \
  tests/test_corsi_attention.py \
  tests/test_corsi_motion_baseline.py
```

### Last known passing result

```text
Date: 2026-06-29
Command: V2 LSTM binding/readout repair validation
Result: passed/completed
Notes:
  - 30 memory-recall V2 model/train-eval/integration tests passed
  - py_compile passed for touched V2 model/loss/train/analysis/probe files
  - git diff --check passed
  - Phase 3 probe passed on memory-attention best checkpoint
  - CUDA was verified usable in robosuite outside managed sandbox device-node isolation
```

---

## 7. Open Risks / Unknowns

| Risk / Unknown | Impact | How to resolve |
|---|---|---|
| True remote state was not fetched | Local upstream ref may be stale relative to GitHub | Fetch only in a commit/ops task when remote network access is intended |
| Snapshot docs may be local-only if not committed/tagged | V1 comparison point can be lost during later edits | Commit `PROJECT_STATE.md` and `reports/model_v1_snapshot.md`; optionally tag the commit |
| Current replay/controller status is unresolved | Model target semantics may still be wrong for replay/closed-loop use | Run a dedicated eval audit separating FK pass from renderer failure |
| Visualization renderer Tier-B validation failed threshold in reports | Visual replay confidence is limited | Inspect `reports/corsi_prediction_visualization_report.md` in an eval lane |
| `joint_only_seed0` long continuation hit a non-finite gradient after the best checkpoint | Long-continuation stability is uncertain | Investigate only if continuation stability matters |
| Deprecated motion-base result dirs remain under `corsi_artifacts` | Cleanup and implementation may conflict | Classify with path/size/purpose/reference/risk before any deletion |
| V2 compressed final-state recall still has weak maintenance/readout evidence | `memory_attention` solves behavior by querying the write trajectory, but final compressed `[h;c]` remains weak for ordered identity in Phase 3 probes | Run attention-lesion, final-state-only, shuffled-memory-state, and blank-delay controls |
| Explicit binding + auxsplit full-data checkpoint is partially graded but still ambiguous as primary substrate | Experiment (a) found token `0.704`, U-score `0.202`, distance-dependent transpositions, and high duplicate `0.525`; this is not simple failure but not a clean substrate decision | Run attention-lesion / blank-delay before choosing main substrate |
| Duplicate masking has not yet been evaluated | Decision is recorded, but no pre/post masking metrics exist yet; duplicate conclusions must not be updated until masking is proven active | Run output-masking autonomous decode and compare duplicate rate / token accuracy / U-score / final-order probe before and after |
| Complete per-length 2-9 metrics are now summarized in the current-model audit, but generated eval JSONs still need enhanced re-export for new fields | Supervisor feedback now has length-by-length interpretation; old eval JSONs predate the new explicit duplicate/set-overlap schema | Use `reports/v2_current_model_audit_20260629.md` for current interpretation; re-run evaluator only when machine-readable enhanced rows are needed |
| Loss component curves may not be logged in enough detail | Hard to explain which loss term drives behavior | Audit `losses.py`, `train.py`, and evaluator outputs; add component logging if missing |
| Stage 1 pretraining value is not fully ablated | Cannot yet say how much pretraining is enough | Run no-pretrain and multiple Stage1-epoch warm-start ablations |
| K=20 / K=30 V2 memory-recall results do not appear to be current ACTIVE artifacts | Cannot compare sampling density yet | Create clean V2 K ablation configs and canonical artifacts only after approval |

---

## 8. Next Task Queue

| Priority | Task | Thread type | Notes |
|---|---|---|---|
| P0 | Run output-masking autonomous decode evaluation | `[cogrobot/eval]` | Compare pre/post duplicate rate, token accuracy, U-score, and final-order probe; classify situation A or B |
| P0 | Run attention-lesion controls on the memory-attention checkpoint | `[cogrobot/eval]` | Mask early memory states, keep only final state, shuffle memory-state order, and compare to final-state probe level |
| P0 | Design blank-delay / maintenance-only condition | `[cogrobot/design/eval]` | Separate trajectory rereading from compressed memory maintenance |
| P0 | Keep current V2 audit reports as the canonical answer to supervisor feedback | `[cogrobot/audit]` | Main summaries are `reports/v2_current_model_audit_20260629.md` and `reports/v2_lstm_associative_memory_binding_final_report_20260629.md` |
| P1 | Commit V2 memory-isolates + Stage 0 closeout | `[cogrobot/commit]` | Include `reports/v2_memory_isolates_snapshot_20260626.md`; do not include generated `corsi_artifacts/` |
| P1 | Choose next V2 recall readout mechanism after slot-compress fix | `[cogrobot/design/impl]` | Final memory now linearly exposes ordered identity at about 0.88-0.90, so the next isolate should address duplicate suppression and long-sequence autonomous decoding |
| P1 | Plan K=12/20/30 V2 ablation | `[cogrobot/design]` | Do not reuse deprecated `freecam_motion_segment_uniform` artifacts as current V2 results |
| P1 | Plan Stage1 pretraining ablation | `[cogrobot/design]` | Compare 0/5/10/20/30/60 Stage1 epochs by downstream Stage2 recall |
| P1 | Plan loss/output ablation | `[cogrobot/design]` | Minimum: seq only, seq+coord, seq+memory_order, seq+aux, current, final-memory order on, length head on |
| P2 | Decide archive/keep status for Stage 0 worktree 629d after commit | `[cogrobot/cleanup]` | Worktree code is integrated; keep until commit review is complete |
| P2 | Cleanup deprecated artifacts | `[cogrobot/cleanup]` | List candidates first; do not delete without approval |

---

## 9. Standard Thread Prompts

### Audit

```text
Read PROJECT_STATE.md first.
Do not modify files.
Use subagents if helpful.

Audit only:
- current git state
- relevant files
- data/artifact status
- validation commands
- risks and unknowns

Output a short decision memo and stop.
```

### Implementation

```text
Read PROJECT_STATE.md first.

Implement only this lane:
Scope:
Allowed files:
Forbidden files:
Validation:
Stop when:

Do not commit.
Update PROJECT_STATE.md only if the project status changes.
```

### Evaluation

```text
Read PROJECT_STATE.md first.

Run only the requested validation/evaluation.
Do not train unless explicitly requested.
Save machine-readable results.
Summarize metrics and update PROJECT_STATE.md if results change project status.
```

### Cleanup

```text
Read PROJECT_STATE.md first.

Do not delete anything yet.
List cleanup candidates with:
- path
- size
- status ACTIVE/PARKED/DEPRECATED/UNKNOWN
- evidence that it is unused
- deletion risk

Wait for approval before deleting.
```

### Commit

```text
Read PROJECT_STATE.md first.

Review:
- git status
- diff
- ignored files
- test results
- generated artifact risk

Do not stage unrelated files.
Report exact commit scope before committing.
Push only after confirming branch and remote.
Update PROJECT_STATE.md with the pushed commit if successful.
```

### V2 current-model audit

```text
Read PROJECT_STATE.md first.

Goal:
Audit the current Corsi Memory Recall V2 model before changing architecture.

Answer:
1. What results do we currently have?
2. What exactly does the model output?
3. During movement / intermediate steps, what is being predicted?
4. What is the loss?
5. How is the loss computed?
6. What is accuracy from length 2 to 9?
7. What limits the model's performance?
8. Is the image input 64x64 or 128x128?
9. What K is currently used?
10. Do we have K=20 / K=30 results?
11. What is Stage 1?
12. What is Stage 2?
13. Why do we need pretraining?
14. How do we know pretraining is enough?
15. Why output block index + XY instead of joints?
16. What should we do next?

Do not guess.
If unavailable, write:
UNKNOWN - needs audit

Suggested deliverable:
reports/v2_current_model_audit_<YYYYMMDD>.md
```

---

## 10. Thread Index

| Thread | Type | Purpose | Outcome |
|---|---|---|---|
| 2026-06-23 local ledger audit | `[cogrobot/audit]` | Current branch/data/code state | Git, datasets, code map, validation commands, risks, and queue audited |
| 2026-06-24 model v1 snapshot | `[cogrobot/audit]` | Freeze current model/training/result state before V2 | `reports/model_v1_snapshot.md` created; V2 design is next |
| 2026-06-24 model v2 plan | `[cogrobot/design]` | Turn V2 requirements into an implementation-ready plan | `reports/model_v2_plan.md` created; smoke-gated implementation lanes are next |
| 2026-06-24 v2 data/schema lane | `[cogrobot/impl]` | Implement V2 canonical data contract and validators | Lane B files created; real V2 canonical artifact generated; Gate 1/2 validation passed |
| 2026-06-24 v2 model/loss lane | `[cogrobot/impl]` | Implement V2 model and losses | Lane C files integrated into main workspace; model tests passed |
| 2026-06-24 v2 train/eval lane | `[cogrobot/impl]` | Implement V2 train/eval/run-suite/state extraction | Lane D files integrated into main workspace; train/eval tests passed |
| 2026-06-24 v2 integration lane | `[cogrobot/impl]` | Merge Lane C/D with Lane B contracts | Lane F compatibility fixes and integration test added; Gate 3/4 and tiny Stage 1/2 smoke passed |
| 2026-06-26 v2 memory-isolates closeout | `[cogrobot/commit]` | Snapshot memory-isolate code/results and integrate Stage 0 gates | `reports/v2_memory_isolates_snapshot_20260626.md` created; Stage 0 worktree 629d code/test integrated; Stage 0 runner passed in main |
| 2026-06-29 project-state consolidation | `[cogrobot/handoff]` | Consolidate operating rules and raw dataset handoff into this file | This file becomes the single canonical operating handoff and project ledger |
| 2026-06-29 v2 LSTM associative-memory binding final report | `[cogrobot/eval/impl]` | Diagnose LSTM binding/readout failure, implement Phase 2B hooks, verify memory-attention repair | `reports/v2_lstm_associative_memory_binding_final_report_20260629.md` recorded; memory-attention checkpoint reaches full/token/length 1.000 at epoch 17; compressed final-state maintenance remains open |
| 2026-06-29 binding experiment (a) explicit-binding error structure | `[cogrobot/eval]` | Probe epoch-69 explicit-binding + auxsplit full-data checkpoint behind val full 0.250 | Mixed / partially graded: test full 0.325, token 0.704, U-score 0.202, distance-dependent transposition True, duplicate 0.525; proceed to attention-lesion / blank-delay |
| 2026-06-30 duplicate output compliance decision | `[cogrobot/design/eval]` | Decide how to handle duplicate outputs before baseline calibration | Use inference-time masking as compliance layer; avoid training-time penalty/inhibition until baseline is locked; next thread should run pre/post masking A/B classification |
| UNKNOWN - needs audit | `[cogrobot/eval]` | qpos replay validation | To be filled |
| UNKNOWN - needs audit | `[cogrobot/cleanup]` | Deprecated data/code cleanup | To be filled |
