# V2 Memory-Isolates Snapshot

Date: 2026-06-26

Repository: `/home/wei2025/Developer/cogrobot`

Branch: `codex/corsi-motion`

Base commit at snapshot time: `0d25e725` (`Add Corsi memory recall V2 scaffold`)

Status: local review snapshot for V2 memory-isolates and Stage 0 gates. No new full training was started during this snapshot task.

## 1. Git And Worktree State

Main worktree:

```text
/home/wei2025/Developer/cogrobot  0d25e725 [codex/corsi-motion]
```

Current tracked dirty scope before this snapshot/report commit:

- `PROJECT_STATE.md`
- `corsi/experiments/corsi_memory_recall_v2/evaluate.py`
- `corsi/experiments/corsi_memory_recall_v2/losses.py`
- `corsi/experiments/corsi_memory_recall_v2/model.py`
- `corsi/experiments/corsi_memory_recall_v2/run_suite.py`
- `corsi/experiments/corsi_memory_recall_v2/train.py`
- `tests/test_corsi_memory_recall_v2_integration.py`
- `tests/test_corsi_memory_recall_v2_model.py`
- `tests/test_corsi_memory_recall_v2_train_eval.py`

Stage 0 worktree checked:

```text
/home/wei2025/.codex/worktrees/629d/cogrobot  0d25e725 (detached HEAD)
```

The Stage 0 worktree contained Stage 0-related untracked files:

- `corsi/experiments/corsi_memory_recall_v2/stage0.py`
- `tests/test_corsi_memory_recall_v2_stage0.py`

It also had a local `PROJECT_STATE.md` edit. Only the Stage 0 code/test files were integrated into main; the worktree state-file edit was not copied.

## 2. Code State

The current V2 memory-isolate code extends the scaffold with:

- variable-length Stage 2 eval padding
- Stage 2 warm-start from a Stage 1 grounding checkpoint
- multi-checkpoint selection: `best_full_sequence.pt`, `best_token.pt`, `best_val_loss.pt`, `latest.pt`
- memory-order prefix auxiliary loss
- optional final-memory order and length heads/losses
- configurable memory write mode, including `slot_compress`
- `D_mem=64` slot-compress isolate support through generated run configs
- Stage 0 diagnostic runner and tests

Stage 0 integration status:

- `integrated`, not superseded.
- Main did not have an equivalent Stage 0 runner/test.
- The imported Gate 3 is explicitly a diagnostic upper-bound: it proves a Stage 0 oracle item-memory `final_memory` can be linearly decoded, not that RGB-trained production memory is sufficient.
- Production memory-isolate artifacts separately show real final-memory order probes for the slot-compress runs.

## 3. Data And Artifacts

Tracked code uses existing generated data; no generated artifacts are intended for Git.

Active V2 canonical root:

```text
corsi_artifacts/memory_recall_v2/canonical/corsi_memory_recall_v2_k12
```

V2 canonical fingerprint:

```text
782962d47e55a1e4631919f4042bc1eb31ff0b66c7c27adff27aaf7b994b4485
```

Run root:

```text
corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12
```

Evaluation root:

```text
corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12
```

Stage 0 root:

```text
corsi_artifacts/memory_recall_v2/stage0
```

## 4. Stage 0 Gates

Main-worktree Stage 0 run:

```bash
conda run -n robosuite python -m corsi.experiments.corsi_memory_recall_v2.stage0 \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json \
  --output-root corsi_artifacts/memory_recall_v2/stage0 \
  --device cpu
```

Summary:

```text
corsi_artifacts/memory_recall_v2/stage0/summary.json
```

Gate results:

| Gate | Status | Threshold | Result |
|---|---:|---|---|
| Motor Shell single-block reachability | pass | 9/9 trials, success rate 1.0, xy <= 0.025m, z <= 0.055m | 9/9, worst xy 0.0109m, worst z 0.0093m |
| Oracle recall length-9 upper bound | pass | exact sequence accuracy 1.0 | exact 1.0, token 1.0, 26 steps |
| Frozen linear probe from `M` | pass | token order >= 0.99, exact >= 0.95 | token 1.0, exact 1.0 |
| CNN/Visual/Motor tiny overfit | pass | `L_pose + 0.5 L_EE <= 0.002` | final loss 0.00147 |

Important limitation:

Gate 3 uses a Stage 0-generated oracle item-memory diagnostic checkpoint. It is useful as an upper-bound gate, but the production checkpoint evidence is the separate final-memory order probe in the memory-isolate results below.

## 5. Memory-Isolate Results

Primary current checkpoint:

```text
corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624/best_full_sequence.pt
```

Primary eval files:

```text
corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_val_eval.json
corsi_artifacts/memory_recall_v2/eval/corsi_memory_recall_v2_k12/stage2_seed0_slot_compress_dmem64_full_20260624_best_full_sequence_test_eval.json
```

Selected comparison:

| Run / checkpoint | Val full | Val token | Val EOS | Val length | Test full | Test token | Test EOS | Test length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original Stage 2 scaffold | 0.000 | 0.204 | 1.000 | 0.250 | n/a | n/a | n/a | n/a |
| Stage 1 warm-start | 0.000 | 0.308 | 1.000 | 0.975 | 0.000 | 0.273 | 1.000 | 1.000 |
| orderaux balanced | 0.125 | 0.442 | 1.000 | 0.875 | 0.075 | 0.462 | 1.000 | 0.950 |
| slot-compress `D_mem=64`, best full | 0.325 | 0.627 | 0.875 | 0.875 | 0.300 | 0.631 | 0.900 | 0.850 |
| slot-compress `D_mem=64`, best val loss | 0.250 | 0.677 | 0.975 | 0.975 | 0.350 | 0.654 | 0.925 | 0.875 |

Production final-memory probe for slot-compress `D_mem=64`:

| Checkpoint | Val block-token order | Test block-token order | Known-length exact |
|---|---:|---:|---:|
| best full | 0.882 | 0.877 | 0.675 |
| best val loss | 0.886 | 0.895 | 0.675 |

Duplicate/collapse status for slot-compress `D_mem=64`:

| Split/checkpoint | Duplicate sequence rate | Mean unique predicted blocks | Mean set overlap |
|---|---:|---:|---:|
| best full val | 0.525 | 4.60 | 0.797 |
| best full test | 0.575 | 4.625 | 0.798 |
| best val-loss val | 0.650 | 4.40 | 0.775 |
| best val-loss test | 0.500 | 4.60 | 0.813 |

Interpretation:

- The slot-compress `D_mem=64` run is the current best production memory isolate.
- `final_memory` now carries ordered block identity strongly enough for a frozen probe.
- Autonomous recall still has a duplicate-collapse / long-sequence decoding problem.
- The next isolate should target recall readout or duplicate suppression, not upstream RGB/Visual/Motor encoding.

## 6. Stage 0 Worktree Decision

`/home/wei2025/.codex/worktrees/629d/cogrobot` is now partially integrated:

- Integrated: `stage0.py`, `test_corsi_memory_recall_v2_stage0.py`
- Not integrated: its `PROJECT_STATE.md` edit, because main has newer memory-isolate state
- Generated worktree artifacts: superseded by the main-worktree Stage 0 run under `corsi_artifacts/memory_recall_v2/stage0/`

Recommended thread/worktree handling:

- Keep the 629d worktree until this commit lands and is reviewed.
- After this commit is accepted, 629d can be archived as integrated.

## 7. Validation In This Snapshot Task

Passed:

```bash
conda run -n robosuite python -m pytest \
  tests/test_corsi_memory_recall_v2_model.py \
  tests/test_corsi_memory_recall_v2_train_eval.py \
  tests/test_corsi_memory_recall_v2_integration.py -q
# 25 passed
```

```bash
conda run -n robosuite python -m pytest tests/test_corsi_memory_recall_v2_stage0.py -q
# 4 passed
```

```bash
conda run -n robosuite python -m corsi.experiments.corsi_memory_recall_v2.train --help
conda run -n robosuite python -m corsi.experiments.corsi_memory_recall_v2.evaluate --help
conda run -n robosuite python -m corsi.experiments.corsi_memory_recall_v2.stage0 --help
```

```bash
conda run -n robosuite python -m corsi.experiments.corsi_memory_recall_v2.stage0 \
  --config corsi/experiments/corsi_memory_recall_v2/configs/memory_recall_v2_k12.json \
  --output-root corsi_artifacts/memory_recall_v2/stage0 \
  --device cpu
# passed
```

## 8. Recommendation

Commit the V2 memory-isolates code, Stage 0 runner/test, this snapshot report, and the updated project ledger together if final whitespace/status checks remain clean.

Do not start new full training in this commit thread.

Next technical lane: recall readout / duplicate-collapse isolate using the slot-compress `D_mem=64` checkpoint family as the baseline.
