# Corsi Motion Convergence Audit

Date: 2026-06-22

## Loss Mask Audit

Conclusion: `training_loss_includes_within_segment_and_segment_boundary_transitions`.

The training objective uses `loss_mask == transition_mask`. In the canonical dataset,
`transition_mask` is exactly `within_segment_transition | segment_boundary_transition`,
excluding only the final timestep of each episode. Continuation training must preserve this mask.

Counts across the canonical manifest:

- transitions: 26000
- within-segment transitions: 24200
- segment-boundary transitions: 1800
- samples checked: 400

## Resume Audit

Original checkpoints contain model state, optimizer state, epoch, best metric, config,
normalization, and joint names. They do not contain AMP scaler state or RNG states.
Therefore exact latest-state resume is not available for the original runs. Any continuation
from the original checkpoints must be labeled warm-start from `best.pt`, not exact resume.

## Per-Run Convergence

| Model | Seed | Epochs | Class | Best epoch | Since best | Best val RMSE | Final val RMSE | Extend | Reason |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| visual_joint | 0 | 240 | UNCERTAIN | 190 | 50 | 0.041656 | 0.042407 | False | stopped_before_max_by_recorded_early_stopping |
| visual_joint | 5 | 300 | STILL_IMPROVING | 294 | 6 | 0.043191 | 0.043422 | True | reached_epoch_300_and_classified_STILL_IMPROVING |
| visual_joint | 10 | 188 | CONVERGED | 138 | 50 | 0.044323 | 0.045152 | False | stopped_before_max_by_recorded_early_stopping |
| visual_joint | 15 | 226 | UNCERTAIN | 176 | 50 | 0.044294 | 0.045420 | False | stopped_before_max_by_recorded_early_stopping |
| visual_joint | 20 | 220 | OVERFITTING | 170 | 50 | 0.046827 | 0.052213 | False | stopped_before_max_by_recorded_early_stopping |
| joint_only | 0 | 300 | STILL_IMPROVING | 300 | 0 | 0.034439 | 0.034439 | True | reached_epoch_300_and_classified_STILL_IMPROVING |
| joint_only | 5 | 300 | STILL_IMPROVING | 299 | 1 | 0.034897 | 0.035775 | True | reached_epoch_300_and_classified_STILL_IMPROVING |
| joint_only | 10 | 300 | STILL_IMPROVING | 296 | 4 | 0.034203 | 0.035989 | True | reached_epoch_300_and_classified_STILL_IMPROVING |
| joint_only | 15 | 300 | STILL_IMPROVING | 299 | 1 | 0.033231 | 0.033886 | True | reached_epoch_300_and_classified_STILL_IMPROVING |
| joint_only | 20 | 300 | STILL_IMPROVING | 296 | 4 | 0.032953 | 0.034511 | True | reached_epoch_300_and_classified_STILL_IMPROVING |

## Missing History Fields

Learning rate is recoverable from optimizer param groups in the checkpoint, but it was not
recorded per epoch in `curves.json`. Gradient norm, AMP scaler state, scheduler state,
and RNG state were not recorded in the original artifacts. These values are not fabricated.
