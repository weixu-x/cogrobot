# Model V2 Plan

Date: 2026-06-24

Branch: `codex/corsi-motion`

Status: approved design plan. No V2 implementation or training has been done in this document step.

Primary source requirements:

- `reports/model_v1_snapshot.md`
- `/home/wei2025/.codex/attachments/02c0c217-c6f4-4425-8e1e-930354028958/corsi_v2_build_plan-2.md`

The attachment is titled as a "v1 Build Plan", but in this repository it is treated as the Model V2 design because Model V1 is already frozen as the 7-joint one-step motion-prediction baseline.

## 1. Executive Summary

Model V2 changes the task from one-step joint motion prediction to an autonomous Corsi sequence-memory task.

V1 predicts:

```text
q_hat_{t+1} = f(I_0..I_t, q_0..q_t)
```

V2 predicts:

```text
[block_1, block_2, ..., block_L, EOS] = recall(memory(RGB presentation segments))
```

The learned cognitive model receives segmented RGB presentation frames. It does not receive block IDs, rank labels, ground-truth joints, target coordinates, future frames, or recall feedback as model inputs. The only state crossing the presentation-to-recall boundary is the final associative memory state `M`.

V2 deliberately separates:

- cognitive model: `RGB segments -> item embeddings -> memory -> block+EOS recall`
- fixed motor shell: `predicted block -> fixed target pose / trajectory / optional video`

This keeps failures attributable to sequence memory and recall rather than continuous control drift.

## 2. Comparison Against Model V1

| Area | Model V1 | Model V2 |
|---|---|---|
| Main purpose | 7-joint causal motion-prediction baseline | Cognitive Corsi sequence-memory baseline |
| Input | RGB and/or normalized 7-joint state per canonical timestep | RGB presentation segments only, plus masks |
| Forbidden as input | Metadata was not model input, but joints were input | `block_id`, `rank`, `block_xy`, `length`, true joints, true EE, future labels, recall frames |
| Target | normalized `q_{t+1}` | ordered block token sequence plus EOS |
| Auxiliary target | none | weak block XY coordinate head; optional pose/EE grounding heads |
| Sequence structure | frame-level recurrent prediction across canonical timesteps | segment-level item encoding, item-level memory updates, autonomous recall |
| Memory claim | LSTM traces are dynamics-analysis features, not working-memory slots | associative memory state is an explicit analysis object, with bottleneck/noise knobs |
| Evaluation metric | normalized RMSE, joint MAE, step accuracy, teacher-forced token posthoc | token accuracy, full-sequence accuracy, length/serial-position curves, EOS accuracy, error taxonomy, causal memory checks |
| Motor behavior | no fixed symbolic motor shell; visualization is prediction rendering | fixed motor shell is separate from cognitive scoring |
| Artifact root | `corsi_artifacts/motion_baseline/...` | new V2 root only; do not overwrite V1 |

What stays the same:

- The immutable raw source dataset remains `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50`.
- The train/val/test split should remain episode-level and length-balanced.
- RGB frames remain `freecam` 128x128 raw observations unless V2 config explicitly downsamples inside the V2 pipeline.
- Existing `freecam_index`, heatmap, and `freecam_ee_xy` workflows remain protected.
- V1 artifacts, checkpoints, reports, canonical data, posthoc outputs, and visualizations are read-only comparison points.

What changes:

- V2 must not reuse the V1 canonical dataset as its training target because V1 canonical data drops raw `ee_pose`, `ee_xy`, `action`, `qpos`, and `qvel`, and it is shaped for one-step joint prediction.
- V2 needs a new canonical schema and loader shaped around `[episode, item, frame]` rather than flat canonical timesteps.
- V2 checkpoint selection should use autonomous validation sequence accuracy, not validation RMSE.
- V2 causal sanity checks are required before any cognitive conclusions.

## 3. Raw Data Fields Used

Raw dataset:

```text
corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50
```

Existing raw fields that support V2:

| Raw field / file | Use in V2 | Model input? |
|---|---|---|
| `rgb [T,128,128,3]` | presentation frames sampled per segment | yes |
| `segments.json` | item boundaries and per-item segment metadata | mask/index only |
| `metadata.json.block_positions[*].xy_norm` | weak coordinate target and motor/eval geometry | no |
| `metadata.json.block_positions[*].xyz_world` | fixed motor-shell and FK/arrival validation | no |
| `metadata.json.xy_normalization` | normalize `ee_xy` if needed | no |
| `joint [T,7]` | Stage 1 motor-imagery grounding target | no |
| `ee_pose [T,7]` | optional EE grounding target and validation | no |
| `ee_xy [T,2]` | optional EE XY target after normalization | no |
| `joint_velocity [T,7]` | optional diagnostics, not default training target | no |
| `action [T,12]` | diagnostics only in V2 plan | no |
| `qpos [T,19]` / `qvel [T,19]` | motor-shell/replay diagnostics only | no |
| `rank [T]` | validation of segment alignment and analysis metadata | no |
| `block_id [T]` | target construction and leakage validation | no |
| `timestamp [T]` | segment sampling/interpolation and validation | no |
| manifest `block_order` | target sequence construction | no |

Important raw-data corrections:

- Current raw arrays do not include `tap_event`, `tap_level`, or per-frame speed fields.
- V2 should use the existing generated `segments.json` as item boundaries, not attempt to recompute tap events.
- The raw metadata does not explicitly persist joint names or qpos names; V2 must persist the 7-joint arm order in its own manifest.
- Raw `ee_xy` is not the same as normalized block XY. Use `metadata.xy_normalization` when deriving normalized EE XY.
- Rank/block coverage should be reported before any rank/location representation claims. The raw plan has no repeated block within an episode, and coverage is not guaranteed to be perfectly rank-balanced.

## 4. Target Definition

Primary target:

```text
target_tokens = [block_order[0], block_order[1], ..., block_order[L-1], EOS]
```

Token vocabulary:

```text
0..8 = block IDs
9 = EOS
padding = ignored by loss
```

Weak coordinate target:

```text
target_xy[r] = metadata.block_positions[str(block_order[r])].xy_norm
```

Grounding targets:

- `joint` rows sampled within each segment for pose grounding.
- optional `ee_pose` or normalized `ee_xy` rows sampled within each segment.

Default V2 loss:

```text
L = 1.0 * L_seq
  + 0.1 * L_pose
  + 0.05 * L_EE
  + 0.05 * L_coord
```

Only `L_seq` defines the behavior. Pose, EE, and XY heads are auxiliary.

## 5. Architecture Contract

Presentation:

```text
for each segment r:
    encode K RGB frames with shared CNN
    reset Visual LSTM at segment start
    reset Motor-imagery LSTM at segment start
    hV_K = visual segment state
    hM_K = visual-induced motor state
    z_r = projection([hV_K, hM_K])
```

Memory:

```text
for r in 1..L:
    M_r = AssociativeMemoryLSTM(z_r, M_{r-1})
M = M_L
```

Recall:

```text
h0, c0 = projections(M)
for step in 0..max_len:
    input = same learned RECALL_TOKEN every step
    logits = block_head(RecallLSTM(input))
    coord = weak_coord_head(hidden)
```

Hard constraints:

- Visual and motor-imagery LSTMs reset at each segment.
- Associative memory updates once per item, not once per frame.
- `M` is the only presentation-to-recall state.
- Recall receives no previous ground-truth token, previous predicted token, rank embedding, step embedding, length embedding, joint state, or rendered image.
- Memory bottleneck/noise must be active enough to avoid a trivial flat near-100% sequence task.
- The associative memory LSTM must expose gate and state traces for later analysis.

Default dimensions:

| Setting | Default |
|---|---:|
| image size | `128` |
| K samples per segment | `12` |
| max Corsi length | `9` |
| token classes | `10` |
| CNN output | `64` |
| visual hidden | `64` |
| motor hidden | `64` |
| item dim | `64` |
| memory hidden `D_mem` | `16` |
| recall hidden | `64` |
| recall token dim | `32` |
| memory noise std | `0.0` initially |

## 6. Training Stages

Do not train full models until smoke gates pass.

Stage 0: data and leakage validation

- Build or load V2 canonical data.
- Verify episode-level split, segment alignment, target construction, and model input channel restrictions.
- Stop if leakage is found.

Stage 1: visual-motor grounding

- Train CNN, Visual LSTM, and Motor-imagery LSTM on independent segments.
- Input: RGB only.
- Targets: sampled `joint`, optional `ee_pose` / normalized `ee_xy`.
- Done gate: held-out pose/EE loss plateaus and tiny-overfit sanity passes.
- Suggested limit: 20 to 30 epochs.

Stage 2: full sequence recall

- Train full model end-to-end.
- Input: complete segmented RGB presentation.
- Target: `[block_1..block_L, EOS]`, weak XY, and auxiliary grounding.
- Checkpoint by autonomous validation sequence accuracy, not pose loss.
- Done gate: cognitive metrics, memory sanity checks, and non-flat length/serial-position diagnostics are available.

No Stage 3 in V2:

- No DAgger.
- No joint-image scheduled sampling.
- No autonomous frame-level controller fine-tuning.
- No learned closed-loop visual servoing.

## 7. Evaluation Plan

Primary cognitive metrics:

- block token accuracy
- full-sequence exact accuracy
- accuracy by sequence length
- accuracy by serial position
- EOS accuracy
- predicted length accuracy
- substitution, omission, insertion, and transposition counts
- memory gate and hidden-state traces

Required causal sanity checks:

| Check | Operation | Pass criterion |
|---|---|---|
| Memory zero | replace `M` with zeros before recall | accuracy drops materially |
| Memory shuffle | give episode A memory to episode B recall | accuracy drops materially |
| Presentation order shuffle | reorder presentation segments | decoded order changes with the shuffled presentation |

Motor-shell metrics, reported separately:

- predicted block
- arrived block
- ground-truth block
- arrival success
- endpoint error
- FK endpoint error
- optional rendered video status

V2 claims must distinguish:

- cognitive recall success
- motor-shell execution success
- optional visual/video rendering success

## 8. Dependency Graph

```text
raw dataset + segments
  -> V2 raw/schema validator
    -> V2 canonicalizer
      -> V2 dataset/collate loader
        -> model forward contract
          -> losses
            -> Stage 1 grounding smoke
            -> Stage 2 recall smoke
              -> cognitive evaluator + causal sanity checks
                -> tiny overfit gate
                  -> full training only after approval

decoded block sequence
  -> fixed motor-shell validator
    -> optional rollout/video reports
```

## 9. Parallel Work Lanes

Lane A: coordinator and documentation

- Owns this plan and future integration notes.
- May update `PROJECT_STATE.md` only when project state actually changes.
- Does not write implementation code.

Lane B: data/schema

- Owns V2 canonicalization, dataset loading, collate, validation, and dataset tests.
- Must use the existing raw dataset and a new V2 artifact root.
- Must not modify V1 canonical data or raw generation.

Lane C: model/losses

- Owns V2 model architecture, instrumented recurrent cells, memory bottleneck/noise, heads, and loss functions.
- Must not edit dataset schema or trainer code except through agreed interfaces.

Lane D: train/eval

- Owns Stage 1 and Stage 2 training harnesses, evaluation, run-suite orchestration, state export, and cognitive metrics.
- Must not alter V1 training scripts.

Lane E: motor shell

- Owns fixed block-to-motion execution, target-pose table, arrival/FK validation, and optional rollout/video generation.
- Must keep cognitive evaluation separate from motor metrics.

Lane F: integration/regression

- Owns merge order, conflict resolution, smoke commands, and regression checks.
- Verifies V1 and protected visual workflows remain intact.

## 10. File Ownership

Use a new package:

```text
corsi/experiments/corsi_memory_recall_v2/
```

Lane A:

- `reports/model_v2_plan.md`
- optional future V2 status rows in `PROJECT_STATE.md`

Lane B:

- `corsi/experiments/corsi_memory_recall_v2/__init__.py`
- `corsi/experiments/corsi_memory_recall_v2/canonicalize.py`
- `corsi/experiments/corsi_memory_recall_v2/dataset.py`
- `corsi/experiments/corsi_memory_recall_v2/validate.py`
- `corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json`
- `tests/test_corsi_memory_recall_v2_dataset.py`

Lane C:

- `corsi/experiments/corsi_memory_recall_v2/model.py`
- `corsi/experiments/corsi_memory_recall_v2/losses.py`
- `tests/test_corsi_memory_recall_v2_model.py`

Lane D:

- `corsi/experiments/corsi_memory_recall_v2/train.py`
- `corsi/experiments/corsi_memory_recall_v2/evaluate.py`
- `corsi/experiments/corsi_memory_recall_v2/run_suite.py`
- `corsi/experiments/corsi_memory_recall_v2/analysis.py`
- `corsi/experiments/corsi_memory_recall_v2/extract_states.py`
- `tests/test_corsi_memory_recall_v2_train_eval.py`

Lane E:

- `corsi/experiments/corsi_memory_recall_v2/motor_shell.py`
- `corsi/experiments/corsi_memory_recall_v2/rollout.py`
- `tests/test_corsi_memory_recall_v2_motor_shell.py`

Shared infrastructure, read-only unless a single owner is assigned:

- `corsi/experiments/corsi_motion_baseline/*`
- `corsi/data/*`
- `corsi/envs/*`
- `corsi/analysis/*`
- `corsi/experiments/visual_base/*`
- `tests/test_corsi_motion_baseline.py`
- `tests/test_corsi_motion_posthoc.py`
- `tests/test_corsi_prediction_visualization.py`
- `tests/test_corsi_heatmaps.py`
- `tests/test_corsi_attention.py`

Forbidden write targets during V2 parallel implementation unless explicitly approved:

- `corsi_artifacts/motion_base/datasets/corsi_motion_raw_len2_9_n50`
- `corsi_artifacts/motion_baseline/canonical/corsi_motion_7joint_k12`
- `corsi_artifacts/motion_baseline/runs/corsi_motion_7joint_k12`
- V1 posthoc, state, and visualization outputs
- `AUTHORS`
- `requirements.txt`
- `.vscode/settings.json`

## 11. Artifact Paths

Proposed V2 canonical root:

```text
corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12
```

Proposed V2 run root:

```text
corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12
```

Proposed V2 evaluation root:

```text
corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12
```

Proposed V2 motor-shell root:

```text
corsi_artifacts/memory_recall_v2/motor_shell/corsi_memory_recall_v2_k12
```

These artifacts are local generated outputs and should remain ignored by Git.

## 12. Validation Gates

Gate 1: raw and segment validation

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.validate \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json \
  --check raw
```

Expected checks:

- all 400 raw episodes present
- failed/skipped episode lists empty
- length counts `2..9 x 50`
- arrays have expected keys, shapes, dtypes, finite values
- timestamps monotonic
- one segment per item
- segment spans valid
- segment `rank` and `block_id` match raw arrays
- no repeated block within one episode

Gate 2: V2 canonical smoke

```bash
conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.canonicalize \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json \
  --overwrite

conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.validate \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json \
  --check canonical
```

Expected checks:

- sample tensors shaped around `[L,K,3,128,128]`
- sampled pose/EE arrays aligned with segment frames
- target tokens include EOS
- padding and ignore masks correct
- train/val/test disjoint by episode
- split length counts fixed
- train-only normalization recorded
- V1 manifest/fingerprint unchanged

Gate 3: leakage and model forward smoke

```bash
conda run -n robosuite python -B -m pytest \
  tests/test_corsi_memory_recall_v2_dataset.py \
  tests/test_corsi_memory_recall_v2_model.py -q
```

Expected checks:

- collate output exposes model inputs separately from targets/metadata
- no target labels in model input dict
- per-segment visual/motor states reset
- memory updates once per item
- recall token is constant
- `M` is the only presentation-to-recall bridge
- EOS and padding masks work
- weak coordinate head is auxiliary

Gate 4: evaluator and tiny training smoke

```bash
conda run -n robosuite python -B -m pytest \
  tests/test_corsi_memory_recall_v2_train_eval.py -q

conda run -n robosuite python -B -m corsi.experiments.corsi_memory_recall_v2.train \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json \
  --stage 2 \
  --device cpu \
  --overfit-episodes 2 \
  --max-epochs 3
```

Expected checks:

- synthetic metrics cover token/full/length/serial/EOS/error taxonomy
- memory zero/shuffle/order-shuffle checks are implemented
- tiny overfit run completes without non-finite losses
- checkpoint contains full resume state

Gate 5: protected regression smoke

```bash
conda run -n robosuite python -m pytest \
  tests/test_corsi_heatmaps.py \
  tests/test_corsi_attention.py \
  tests/test_corsi_motion_baseline.py \
  tests/test_corsi_motion_posthoc.py \
  tests/test_corsi_prediction_visualization.py -q
```

Run full V2 training only after Gates 1 through 5 pass.

## 13. Integration Plan

Integration order:

1. Merge Lane B data/schema first.
2. Run Gate 1 and Gate 2.
3. Merge Lane C model/losses.
4. Run Gate 3.
5. Merge Lane D train/eval.
6. Run Gate 4.
7. Merge Lane E motor shell.
8. Run motor-shell smoke and Gate 5.
9. Update reports and `PROJECT_STATE.md` if project status changes.
10. Only then consider full training.

Parallel-worker rules:

- Each worker owns only its assigned files.
- Workers must not revert unrelated changes.
- Workers must not edit V1 files in place.
- Shared helpers should be wrapped or copied into the V2 package unless a common-helper refactor is explicitly assigned.
- Any schema contract change must be coordinated through Lane B before other lanes depend on it.

## 14. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Attachment title says v1 while repo calls this Model V2 | naming confusion in reports/artifacts | use `Model V2` and `corsi_memory_recall_v2` consistently |
| Raw data has no `tap_event` field | cannot recompute tap events from raw arrays | use existing `segments.json` as item boundary source |
| Raw data used `dwell_steps: 1` generation settings, not the attachment's `dwell >= 3` tap definition | segment semantics differ from described tap detector | document this and validate existing segment alignment |
| Fixed IK/minimum-jerk motor shell is not implemented | motor execution claims are not yet supported | build and validate separately from cognitive evaluation |
| Current replay/controller status is unresolved | closed-loop claims could be invalid | keep V2 claims symbolic/cognitive until motor-shell metrics pass |
| V2 could become a trivial 9-symbol memorizer | flat high accuracy is not cognitive evidence | enforce memory bottleneck/noise and causal memory checks |
| Block/rank sampling may not be balanced enough for representation claims | rank and location factors may be confounded | report coverage before rank/location analysis |
| Weak coordinate head could dominate behavior | classifier could be distorted | keep low weight and report ablations |
| Hidden-state analysis could be overclaimed | memory traces are not human memory by default | tie claims to causal checks and perturbation results |
| Protected visual workflows could break | regressions in existing project lanes | keep shared files read-only or single-owner; run regression tests |

## 15. Definition Of Done For V2 Implementation

V2 implementation is not done until:

- a new V2 canonical dataset can be built from the existing raw data
- no V1 artifacts are overwritten
- leakage tests pass
- V2 model forward tests pass
- tiny overfit smoke passes
- cognitive evaluator reports required metrics and causal checks
- motor-shell metrics are separately reported if rollout is used
- protected V1/visual tests still pass
- full training has been run only after the smoke gates
