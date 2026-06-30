# Corsi V2 Stage 2 Direct-Training Audit - 2026-06-30

## Status

Current action: training has been interrupted at the user's request. No additional seeds, D_mem values, diagnostics, or sweep jobs should be launched until a human decision is made.

Pause snapshot directory:

`corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12_expanded800_20260630/pause_snapshot_stage2_direct_interrupt_20260630`

Active runs at audit time:

| Run | Device | Stage | Resume | Stage 1 warm start | Latest observed epoch | Latest observed best full | Latest observed best token | Latest observed best val loss |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| stage2_seed0_binding_auxsplit_dmem64_expanded800_20260630 | cuda:0 | 2 | Yes, from its own Stage 2 latest.pt after dataloader-worker adjustment | No | 40 | 0.9954545455 | 0.9993333333 | 0.0169906131 |
| stage2_seed0_binding_auxsplit_dmem16_expanded800_20260630 | cuda:1 | 2 | No | No | 31 | 0.7727272727 | 0.9600000000 | 0.3119274649 |

These values are the latest checkpoint observations after manual interruption. The runs did not stop by early stopping.

## Training Command Semantics

The active commands use `--stage 2`.

The trainer only loads a Stage 1 checkpoint when either `--warm-start-stage1-checkpoint` is provided or the config contains `warm_start_stage1_checkpoint`.

Current expanded800 configs do not contain `warm_start_stage1_checkpoint`, and the active commands did not pass `--warm-start-stage1-checkpoint`.

Checkpoint metadata confirms top-level `stage = 2` for both active runs, with no `extra.warm_start` entry.

Conclusion: both active runs are direct Stage 2 training from random initialization, except that the D_mem64 run resumed from its own Stage 2 checkpoint after the dataloader-worker adjustment. Stage 1 did not participate in either active run.

## Model Structure Check

`corsi/experiments/corsi_memory_recall_v2/model.py::build_model` accepts `stage` but deletes it as a no-op before constructing `CorsiMemoryRecallV2Model`.

An actual instantiation comparison using the expanded800 config and canonical manifest showed:

| Check | Result |
| --- | --- |
| Stage 1 state_dict keys only | none |
| Stage 2 state_dict keys only | none |
| Shape differences | none |
| Stage 1 trainable parameter count | 295504 |
| Stage 2 trainable parameter count | 295504 |

Conclusion: Stage 1 and Stage 2 use the same model topology for this code path. They differ by training objective and call behavior, not by parameter structure.

## Training Objective Difference

Stage 1 uses `compute_stage1_loss`, which is an auxiliary presentation-frame reconstruction objective over `joint`, `ee_pose`, and `ee_xy`.

Stage 2 uses `compute_stage2_loss`, which wraps `combined_v2_loss`: sequence recall cross-entropy plus configured memory/order/identity/orthogonal/coordinate/auxiliary terms.

The current config includes:

```json
{
  "memory_write_mode": "item_context_binding",
  "recall_readout_mode": "final",
  "loss_weights": {
    "memory_order": 0.3,
    "memory_identity": 0.1,
    "memory_aux_orthogonal": 0.01
  },
  "early_stopping_patience": 30
}
```

## Risk Assessment

If the intended experimental protocol requires Stage 1 presentation pretraining followed by Stage 2 warm-start training, the active commands are missing the Stage 1 component and should be treated as direct Stage 2 baselines, not staged baselines.

If the intended protocol is exactly the objective-file request for a binding+aux-split Stage 2 baseline with val-loss early stopping, then the active commands match that request, but Stage 1 is not part of the run.

## Next Required Step

Before any further training:

1. Inspect raw data and model code.
2. Decide whether direct Stage 2 results should be kept only as an audit baseline.
3. Decide whether to rerun a staged Stage 1 -> Stage 2 warm-start protocol.
4. Do not resume either active run unless explicitly requested.

## Pause Snapshot Contents

Snapshot checkpoint files:

| Run | File | SHA256 |
| --- | --- | --- |
| dmem64_seed0 | best.pt | f708df5fe3cbb68e6358cb1be81a39b9add97af148e049db5de71a534e3a8d8e |
| dmem64_seed0 | best_full_sequence.pt | 432c27d024e1e68f03f19093e952fd389972c566f767ac321d95b455fb13e8af |
| dmem64_seed0 | best_token.pt | 29308a22fda74659c6574a608c0b19913db592ffa7105814224e0ef0635ef07a |
| dmem64_seed0 | best_val_loss.pt | ede2f0f89ac7443f1e865a57941d685af580a63e5824fb7a35712436fbaf3ea3 |
| dmem64_seed0 | latest.pt | 5de05453c5e60f534a9cc81768b3c3a27affccb7b63a057a6958e8f72ce99b0f |
| dmem16_seed0 | best.pt | 3890b3a90e344166f5ba4736eed10e5fa85a0d98fee4341630968f75002faf71 |
| dmem16_seed0 | best_full_sequence.pt | 42499baf5b383aee35479269269cda21ec6232c8d523664ef7fe3113aee68340 |
| dmem16_seed0 | best_token.pt | a0f5fde4463ad32f9db639bae490bd190e0f2036ba137bb0755baae8191f02da |
| dmem16_seed0 | best_val_loss.pt | bd7cb7b93afac38a065404a234743102c4f6c61cf1e105a5256d9a2430e7a506 |
| dmem16_seed0 | latest.pt | 9bc71fafb6bbdd9fa37f11b6f666b76d6956b44eeee4b45a0caf3ccd173fb77c |
