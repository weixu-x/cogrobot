# CogRobot Project State

Last updated: 2026-06-24
Updated by: user / Codex
Current branch: codex/corsi-motion
Remote branch: origin/codex/corsi-motion
Working tree status: Model V1 snapshot docs recorded; superseded draft `reports/2026-06-24_corsi_7joint_motion_baseline_snapshot.md` remains untracked
Primary objective: Preserve Model V1 as a reproducible snapshot before designing the next model version from existing raw data.

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
| data | ACTIVE | Dataset/schema loading and validation | Current raw and canonical motion datasets verified | Dataset status is clear and validator/smoke result is recorded |
| replay | UNKNOWN - needs eval | qpos / controller replay validation | FK posthoc pass and renderer-threshold failure are report-recorded; needs dedicated eval audit | Replay metrics are machine-readable and decision is recorded |
| model | PARKED | Model/training work | Model V1 snapshot recorded; next step is V2 design only | V2 design is approved before implementation/training |
| cleanup | PARKED | Remove obsolete files/artifacts safely | Deprecated scratch candidates identified; do not delete without approval | Deletion candidates are reviewed before removal |
| commit | PARKED | Review, commit, push | Only after implementation/eval is complete | Working tree scope and tests are confirmed |

## 2. Branch / Git State

| Item | Value |
|---|---|
| Current branch | `codex/corsi-motion` |
| Remote tracking branch | `origin/codex/corsi-motion` |
| Last pushed commit | Local upstream ref matches HEAD `6f7419c848c278c462ffca6768d105c303f417bb` (`Add Corsi 7-joint motion baseline`); remote was not fetched |
| Dirty tracked files | None expected after committing snapshot docs; verify with `git status --short --branch` |
| Important untracked files | `reports/2026-06-24_corsi_7joint_motion_baseline_snapshot.md` is a superseded draft; canonical snapshot is `reports/model_v1_snapshot.md` |
| Ignored/generated artifacts | `.pytest_cache/`, `corsi_artifacts/`, `robosuite.egg-info/`, `__pycache__/`, `robosuite/savevideo/` |
| Do not commit | `__pycache__`, `.DS_Store`, large generated artifacts, temporary outputs, accidental PDF/log/cache files |

## 3. Data Registry

| Dataset / Artifact | Status | Path | Schema / Format | Purpose | Notes |
|---|---|---|---|---|---|
| corsi_motion_raw_len2_9_n50 | ACTIVE | `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50` | `scala_corsi_motion_raw_v1`; 400 episodes; lengths 2-9 x 50; camera `freecam`; arrays include `joint [T,7]`, `action [T,12]`, `qpos/qvel [T,19]` | Raw input for current 7-joint motion baseline | Present and manifest/sample verified |
| corsi_motion_7joint_k12 | ACTIVE | `corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12` | `scala_corsi_motion_canonical_7joint_v1`; `joint_dim=7`; `K=12`; splits train/val/test 320/40/40 | Canonical dataset for current baseline | Raw manifest hash matches source manifest |
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
Date: 2026-06-23
Command: conda `--help` checks for raw generation and motion-baseline modules; import smoke for dataset/model
Result: passed
Notes: pytest, replay, training, and artifact-writing eval were not run in this audit
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

## 8. Next Task Queue

| Priority | Task | Thread type | Notes |
|---|---|---|---|
| P0 | Design next model version using existing raw data | `[cogrobot/design]` | Do not overwrite V1 assumptions |
| P1 | Audit V2 plan against raw data and current code | `[cogrobot/audit]` | Use subagents |
| P2 | Implement V2 dataset/model changes | `[cogrobot/impl]` | After design approval |

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
| UNKNOWN - needs audit | `[cogrobot/eval]` | qpos replay validation | To be filled |
| UNKNOWN - needs audit | `[cogrobot/cleanup]` | Deprecated data/code cleanup | To be filled |
