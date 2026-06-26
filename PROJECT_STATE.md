# CogRobot Project State

Last updated: 2026-06-26
Updated by: user / Codex
Current branch: codex/corsi-motion
Remote branch: origin/codex/corsi-motion
Working tree status: V2 memory-isolates, slot-compress `D_mem=64`, Stage 0 runner/test, and snapshot report are modified locally; generated outputs are under ignored `corsi_artifacts/`
Primary objective: Commit the reviewed V2 memory-isolates + Stage 0 closeout, then choose the next recall readout / duplicate-collapse isolate.

## 0. How To Use This File

- This file is the current project ledger for `/home/wei2025/Developer/cogrobot`.
- Read this after `AGENTS.md` at the start of every new Codex thread.
- Keep this file short. Link to detailed reports instead of pasting long logs.
- If a fact is uncertain, write `UNKNOWN - needs audit`.
- Do not treat old data/code as valid unless it is listed as `ACTIVE`.
- Update only the changed rows at the end of each task.

## 1. Current Active Lanes

| Lane | Status | Purpose | Current task | Stop condition |
|---|---|---|---|---|
| data | ACTIVE | Dataset/schema loading and validation | V2 data/schema lane implemented and Gate 1/2 validation passed | Dataset status is clear and validator/smoke result is recorded |
| replay | UNKNOWN - needs eval | qpos / controller replay validation | FK posthoc pass and renderer-threshold failure are report-recorded; needs dedicated eval audit | Replay metrics are machine-readable and decision is recorded |
| model | ACTIVE | Model/training work | V2 memory-isolates snapshot created; Stage 0 gates integrated and passed; slot-compress `D_mem=64` restores final-memory ordered identity, but long-sequence recall and duplicate suppression remain weak | Memory-isolates + Stage 0 closeout is committed |
| cleanup | PARKED | Remove obsolete files/artifacts safely | Deprecated scratch candidates identified; do not delete without approval | Deletion candidates are reviewed before removal |
| commit | PARKED | Review, commit, push | Only after implementation/eval is complete | Working tree scope and tests are confirmed |

## 2. Branch / Git State

| Item | Value |
|---|---|
| Current branch | `codex/corsi-motion` |
| Remote tracking branch | `origin/codex/corsi-motion` |
| Last pushed commit | Local upstream ref matches HEAD `6f7419c848c278c462ffca6768d105c303f417bb` (`Add Corsi 7-joint motion baseline`); remote was not fetched |
| Dirty tracked files | `corsi/experiments/corsi_memory_recall_v2/train.py`; `evaluate.py`; `run_suite.py`; `model.py`; `losses.py`; V2 train/eval/model/integration tests; `PROJECT_STATE.md` |
| Important untracked files | `corsi/experiments/corsi_memory_recall_v2/stage0.py`; `tests/test_corsi_memory_recall_v2_stage0.py`; `reports/v2_memory_isolates_snapshot_20260626.md`; generated outputs are ignored under `corsi_artifacts/` |
| Ignored/generated artifacts | `.pytest_cache/`, `corsi_artifacts/`, `robosuite.egg-info/`, `__pycache__/`, `robosuite/savevideo/` |
| Do not commit | `__pycache__`, `.DS_Store`, large generated artifacts, temporary outputs, accidental PDF/log/cache files |

## 3. Data Registry

| Dataset / Artifact | Status | Path | Schema / Format | Purpose | Notes |
|---|---|---|---|---|---|
| corsi_motion_raw_len2_9_n50 | ACTIVE | `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50` | `scala_corsi_motion_raw_v1`; 400 episodes; lengths 2-9 x 50; camera `freecam`; arrays include `joint [T,7]`, `action [T,12]`, `qpos/qvel [T,19]` | Raw input for current 7-joint motion baseline | Present and manifest/sample verified |
| corsi_motion_7joint_k12 | ACTIVE | `corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12` | `scala_corsi_motion_canonical_7joint_v1`; `joint_dim=7`; `K=12`; splits train/val/test 320/40/40 | Canonical dataset for current baseline | Raw manifest hash matches source manifest |
| corsi_memory_recall_v2_k12 | ACTIVE | `corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12` | `scala_corsi_memory_recall_v2_canonical_v1`; segmented RGB `[L,K,3,128,128]`; block+EOS targets; splits train/val/test 320/40/40 | Canonical dataset for Model V2 memory-recall lane | Fingerprint `782962d47e55a1e4631919f4042bc1eb31ff0b66c7c27adff27aaf7b994b4485`; Gate 1/2 validation, full Stage 1/2 seed-0 training/eval, warm-start rerun, final-memory slot-compress diagnostics, and Stage 0 gates completed |
| motion baseline runs/posthoc/visualization outputs | PARKED | `corsi_artifacts/motion_baseline/` | generated run, posthoc, and visualization outputs | Referenced by reports/code; model lane parked | Preserve; not scratch |
| visual-base `freecam_index`, heatmap, and `freecam_ee_xy` workflows | PARKED | `corsi_artifacts/` and visual-base paths | legacy visual/coordinate artifacts | Preserve per project rules | Not current motion-baseline input |
| freecam_motion_segment_uniform_len3_k15 | DEPRECATED | `corsi_artifacts/motion_base/datasets/freecam_motion_segment_uniform_len3_k15` | `freecam_motion_v1_segment_uniform` | old qpos knot dataset, K=15 | Dataset root missing; only historical summaries found |
| freecam_motion_segment_uniform_len3_k20 | DEPRECATED | `corsi_artifacts/motion_base/datasets/freecam_motion_segment_uniform_len3_k20` | `freecam_motion_v1_segment_uniform` | old qpos knot dataset, K=20 | Dataset root missing; only historical summaries found |
| freecam_motion_segment_uniform_len3_k30 | DEPRECATED | `corsi_artifacts/motion_base/datasets/freecam_motion_segment_uniform_len3_k30` | `freecam_motion_v1_segment_uniform` | old qpos knot dataset, K=30 | Dataset root missing; only historical summaries found |
| motion_base comparison/demo result dirs | DEPRECATED | `corsi_artifacts/motion_base/results/*` | small historical JSON/plot outputs, about 1.9M total | Do not build new work on this | Review before any deletion |

Status vocabulary:

- `ACTIVE`: current valid source for ongoing work.
- `PARKED`: potentially useful later, not part of the current lane.
- `DEPRECATED`: do not build new work on it.
- `DO-NOT-DELETE`: preserve unless explicitly approved.
- `UNKNOWN - needs audit`: do not assume; verify first.

## 4. Code Map

| Area | Status | Main files / dirs | Purpose | Notes |
|---|---|---|---|---|
| project rules | ACTIVE | `AGENTS.md` | Repository handoff and operating rules | Read before this file |
| project state | ACTIVE | `PROJECT_STATE.md` | Current status ledger | Keep short |
| dataset loading | ACTIVE | `corsi/experiments/corsi_motion_baseline/dataset.py`; `corsi/data/motion_raw.py`; `corsi/data/generate_raw.py` | Canonical 7-joint loader and raw generator | Do not run generation without explicit permission |
| collate / batching | ACTIVE | `corsi/experiments/corsi_motion_baseline/dataset.py`; `corsi/data/collate.py`; `corsi/data/collate_visual.py` | Motion collator plus older coord/visual collators | Current baseline uses `collate_motion_prediction_batch` |
| replay / controller | UNKNOWN - needs eval | `corsi/envs/robosuite_corsi.py`; `corsi/scripts/run_robosuite_corsi.py`; visual export scripts | Robosuite Corsi env, qpos collection, rollout/export helpers | FK posthoc passed in reports; renderer Tier-B threshold failed |
| model / training | PARKED | `corsi/experiments/corsi_motion_baseline/model.py`; `corsi/experiments/corsi_motion_baseline/train.py`; `corsi/experiments/corsi_motion_baseline/run_suite.py` | Motion predictor and training suite | Do not train unless explicitly requested |
| evaluation / metrics | ACTIVE | `corsi/experiments/corsi_motion_baseline/evaluate.py`; `posthoc_suite.py`; `visualize_predictions.py`; `corsi/analysis/*.py`; `tests/test_corsi_motion_*.py` | Metrics, posthoc summaries, visualization, tests | Pytest not run in this audit |
| V2 memory recall | ACTIVE | `corsi/experiments/corsi_memory_recall_v2/`; `tests/test_corsi_memory_recall_v2_*.py`; `reports/model_v2_plan.md`; `reports/v2_memory_isolates_snapshot_20260626.md` | V2 segmented RGB memory-recall model, training/eval, Stage 0 gates, and memory-isolate diagnostics | Current best production checkpoint is slot-compress `D_mem=64`; next issue is duplicate-collapse / long-sequence recall |
| reports / handoff | ACTIVE | `reports/model_v1_snapshot.md`; `reports/corsi_motion_current_handoff.md`; baseline audit/final/convergence/visualization reports | Detailed phase records | `model_v1_snapshot.md` is the frozen V1 reference |

## 5. Known Decisions

| Date | Decision | Reason | Affected files/data |
|---|---|---|---|
| 2026-06-23 | Treat this file as a ledger, not a report | New Codex threads need fast orientation without replaying long history | `PROJECT_STATE.md` |
| 2026-06-23 | Keep `AGENTS.md`, `PROJECT_STATE.md`, and handoff reports separate | `AGENTS.md` is stable rules; this file is current status; handoff reports are detailed phase records | project root, `reports/` |
| 2026-06-23 | Do not mix OSC action semantics with qpos target semantics | OSC `arm_action_t` and joint `qpos_t` have different meanings and dimensions | dataset/replay/model lanes |
| 2026-06-23 | Do not start model/training work until data and replay semantics are explicit | Prevent training on ambiguous targets or obsolete data paths | model lane |
| 2026-06-23 | Current motion line is the 7-joint Panda arm baseline | Raw audit and canonical manifest verify `joint_dim=7`; old 6-joint / OSC-action line is not current | `corsi_motion_raw_len2_9_n50`, `corsi_motion_7joint_k12` |
| 2026-06-23 | Final checkpoint selection is validation-RMSE only | Test metrics are read afterward in reports | motion-baseline reports/runs |
| 2026-06-23 | Reported metrics are motion-prediction metrics | They are not closed-loop robot success or Corsi working-memory evidence | motion-baseline reports |
| 2026-06-24 | Model V1 snapshot recorded | Next model version will change structure, so V1 is frozen for comparison | `reports/model_v1_snapshot.md` |
| 2026-06-24 | Model V2 plan recorded | Defines V2 as RGB-segment memory recall with explicit V1 comparison, lane ownership, validation gates, and no full training before smoke passes | `reports/model_v2_plan.md` |
| 2026-06-24 | V2 data/schema lane implemented | Provides separate V2 canonical schema, loader/collate, validators, and generated canonical artifact without modifying V1 artifacts | `corsi/experiments/corsi_memory_recall_v2/`; `tests/test_corsi_memory_recall_v2_dataset.py`; `corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12` |
| 2026-06-24 | V2 model/loss and train/eval lanes integrated | Lane C/D outputs were merged through a Lane F compatibility pass; tests cover batch -> model -> loss/eval; tiny Stage 1/2 real-data smoke passed without full training | `corsi/experiments/corsi_memory_recall_v2/model.py`; `losses.py`; `train.py`; `evaluate.py`; `run_suite.py`; `analysis.py`; `extract_states.py`; V2 tests |
| 2026-06-24 | V2 full Stage 1/2 seed-0 training completed | Stage 2 needed a local eval padding fix for batch-local target lengths; best-checkpoint selection remained val full-sequence accuracy, so `best.pt` stayed epoch 0 because all epochs had full-sequence accuracy 0.0 | `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage1_seed0_full_20260624`; `.../stage2_seed0_full_20260624`; `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/` |
| 2026-06-24 | V2 Stage 2 warm-start full rerun completed | Stage 2 loaded 30 Stage 1 grounding keys from epoch-27 `best.pt`; warm-start improved val token accuracy to 0.3077 but full-sequence accuracy remained 0.0, so recall still fails exact sequence recovery | `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_warmstart_full_20260624`; `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_warmstart_full_20260624_*` |
| 2026-06-24 | V2 final-memory ordered-identity localization completed | Segment/item states are block-decodable, but final memory mostly keeps length rather than ordered identity. D_mem=64 improved short exact only slightly and did not solve order probe. Ordered auxiliary prefix loss helped short sequences; balanced serial-position weighting was the best isolate with val full 0.125, test full 0.075, val length-2 exact 0.8, length-3 exact 0.2, but final_memory order probe stayed only val/test 0.35/0.33 and duplicate rate stayed 0.8. A direct final-memory ordered head/loss worsened behavior and is disabled by default (`memory_order_final=0.0`). | `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_*_minimal_localization.json`; `..._orderaux_balanced_full_20260624_*_eval.json`; `..._orderaux_finalmem_full_20260624_*_eval.json`; `corsi/experiments/corsi_memory_recall_v2/losses.py`; `model.py` |
| 2026-06-24 | Slot-compress `D_mem=64` restores ordered identity in final memory | `stage2_seed0_slot_compress_dmem64_full_20260624` shows the failure was mechanism plus capacity, not upstream encoding. Best full checkpoint: val/test full 0.325/0.300, token 0.627/0.631, EOS 0.875/0.900, length 0.875/0.850, per-length exact val length2/3/4 = 1.0/0.8/0.4 and test length2/3/4 = 1.0/1.0/0.4. Frozen final-memory order probe is val/test 0.882/0.877 for best-full and 0.886/0.895 for best-val-loss; known-length exact probe is 0.675. Duplicate rate drops from about 0.93-0.97 to about 0.50-0.58 but is still too high for long sequences. | `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624`; `corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_*`; `corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_slot_compress_dmem64.json`; `model.py`; `losses.py`; `train.py` |
| 2026-06-26 | V2 memory-isolates closeout snapshot recorded | Captures current branch/worktree state, Stage 0 integration decision, slot-compress `D_mem=64` result, duplicate-collapse risk, and validation commands | `reports/v2_memory_isolates_snapshot_20260626.md`; `PROJECT_STATE.md` |
| 2026-06-26 | Stage 0 gates integrated into main | `/home/wei2025/.codex/worktrees/629d/cogrobot` contained Stage 0-specific code/test; `stage0.py` and its test were integrated, while that worktree's older `PROJECT_STATE.md` edit was not copied. Main Stage 0 run passed all four gates and wrote ignored artifacts. | `corsi/experiments/corsi_memory_recall_v2/stage0.py`; `tests/test_corsi_memory_recall_v2_stage0.py`; `corsi_artifacts/memory_recall_v2/stage0/summary.json` |

## 6. Current Validation Commands

Fast git/status preflight:

```bash
git status --short --branch
```

Dataset/schema check:

```bash
conda run -n robosuite python -B -m corsi.data.generate_raw --help
conda run -n robosuite python -B -m corsi.experiments.corsi_motion_baseline.canonicalize --help
conda run -n robosuite python -B -c "from corsi.experiments.corsi_motion_baseline.dataset import CorsiMotionCanonicalDataset, collate_motion_prediction_batch; from corsi.experiments.corsi_motion_baseline.model import build_model; print('ok')"
```

V2 data/schema Gate 1/2:

```bash
conda run -n robosuite python -B -m pytest tests/test_corsi_memory_recall_v2_dataset.py -q
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.canonicalize --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json --overwrite
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.validate --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json --check all
```

V2 model/train/eval Gate 3/4:

```bash
conda run -n robosuite python -B -m pytest \
  tests/test_corsi_memory_recall_v2_dataset.py \
  tests/test_corsi_memory_recall_v2_model.py \
  tests/test_corsi_memory_recall_v2_train_eval.py \
  tests/test_corsi_memory_recall_v2_integration.py -q

conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train --help
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.evaluate --help
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.run_suite --help
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.extract_states --help

conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json \
  --stage 2 --device cpu --output-root /tmp/corsi_v2_smoke_runs \
  --run-name stage2_overfit1_smoke --max-epochs 1 --overfit-episodes 1

conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json \
  --stage 1 --device cpu --output-root /tmp/corsi_v2_smoke_runs \
  --run-name stage1_overfit1_smoke --max-epochs 1 --overfit-episodes 1
```

Replay check:

```bash
# No current replay command is marked safe/passing. Run only in an explicit eval lane.
```

Core pytest / smoke:

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_motion_baseline.evaluate --help
conda run -n robosuite python -B -m corsi.experiments.corsi_motion_baseline.posthoc_suite --help
conda run -n robosuite python -B -m corsi.experiments.corsi_motion_baseline.visualize_predictions --help
# Pytest was not run in the 2026-06-23 audit.
```

Last known passing result:

```text
Date: 2026-06-26
Command: V2 memory-isolates closeout tests, help checks, and Stage 0 runner
Result: passed/completed
Notes: 25 model/train-eval/integration tests passed; 4 Stage 0 tests passed; train/evaluate/stage0 `--help` passed; main-worktree Stage 0 runner passed all four gates and wrote `corsi_artifacts/memory_recall_v2/stage0/summary.json`.
```

## 7. Open Risks / Unknowns

| Risk / Unknown | Impact | How to resolve |
|---|---|---|
| True remote state was not fetched | Local upstream ref may be stale relative to GitHub | Fetch only in a commit/ops task when remote network access is intended |
| Snapshot docs may be local-only if not committed/tagged | V1 comparison point can be lost during later edits | Commit `PROJECT_STATE.md` and `reports/model_v1_snapshot.md`; optionally tag the commit |
| Current replay/controller status is unresolved | Model target semantics may still be wrong for replay/closed-loop use | Run a dedicated eval audit separating FK pass from renderer failure |
| Visualization renderer Tier-B validation failed threshold in reports | Visual replay confidence is limited | Inspect `reports/corsi_prediction_visualization_report.md` in an eval lane |
| `joint_only_seed0` long continuation hit a non-finite gradient after the best checkpoint | Long-continuation stability is uncertain | Investigate only if continuation stability matters |
| Deprecated motion-base result dirs remain under `corsi_artifacts` | Cleanup and implementation may conflict | Classify with path/size/purpose/reference/risk before any deletion |
| V2 recall still repeats blocks and fails long exact sequences | `slot_compress + D_mem=64` fixes final-memory ordered identity, but autonomous constant-token recall still has duplicate rate about 0.50-0.58 and length/EOS tradeoffs across checkpoints | Next isolate should target recall readout / duplicate suppression, likely lightweight feedback or constrained decoding, rather than more item/memory-write supervision |

## 8. Next Task Queue

| Priority | Task | Thread type | Notes |
|---|---|---|---|
| P0 | Commit V2 memory-isolates + Stage 0 closeout | `[cogrobot/commit]` | Include `reports/v2_memory_isolates_snapshot_20260626.md`; do not include generated `corsi_artifacts/` |
| P1 | Choose next V2 recall readout mechanism after slot-compress fix | `[cogrobot/design/impl]` | Final memory now linearly exposes ordered identity at about 0.88-0.90, so the next isolate should address duplicate suppression and long-sequence autonomous decoding, not upstream CNN/Visual/Motor or the failed final-memory head loss |
| P2 | Decide archive/keep status for Stage 0 worktree 629d after commit | `[cogrobot/cleanup]` | Worktree code is integrated; keep until commit review is complete |

## 9. Standard Thread Prompts

### Audit

```text
Read AGENTS.md and PROJECT_STATE.md first.
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
Read AGENTS.md and PROJECT_STATE.md first.
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
Read AGENTS.md and PROJECT_STATE.md first.
Run only the requested validation/evaluation.
Do not train unless explicitly requested.
Save machine-readable results.
Summarize metrics and update PROJECT_STATE.md if results change project status.
```

### Cleanup

```text
Read AGENTS.md and PROJECT_STATE.md first.
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
Read AGENTS.md and PROJECT_STATE.md first.
Review git status, diff, ignored files, and test results.
Do not stage unrelated files.
Report exact commit scope before committing.
Push only after confirming branch and remote.
Update PROJECT_STATE.md with the pushed commit if successful.
```

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
| UNKNOWN - needs audit | `[cogrobot/eval]` | qpos replay validation | To be filled |
| UNKNOWN - needs audit | `[cogrobot/cleanup]` | Deprecated data/code cleanup | To be filled |
