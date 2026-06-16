"""Coordinate-based dataset helpers for the minimal Corsi system."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from corsi.envs.sequence_generator import (
    BlockLayout,
    CorsiTrial,
    build_coordinate_features,
    canonical_board_layout,
    generate_trial_collection,
)


class CoordinateCorsiDataset:
    """Small dataset wrapper over generated or precomputed Corsi trials."""

    def __init__(
        self,
        trials: Optional[Sequence[CorsiTrial]] = None,
        *,
        num_trials: Optional[int] = None,
        seq_len_range: tuple[int, int] = (2, 9),
        layout: Optional[BlockLayout] = None,
        feature_mode: str = "xydxdy",
        seed: Optional[int] = None,
    ) -> None:
        if trials is None and num_trials is None:
            raise ValueError("Provide either trials or num_trials")
        if trials is not None and num_trials is not None:
            raise ValueError("Provide trials or num_trials, but not both")

        self.feature_mode = feature_mode
        self.layout = dict(layout) if layout is not None else canonical_board_layout()
        if trials is not None:
            self.trials = list(trials)
        else:
            self.trials = generate_trial_collection(
                num_trials=num_trials or 0,
                seq_len_range=seq_len_range,
                layout=self.layout,
                seed=seed,
            )

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, index: int) -> Dict[str, object]:
        trial = self.trials[index]
        features = build_coordinate_features(
            coords=trial.coords,
            delta_coords=trial.delta_coords,
            feature_mode=self.feature_mode,
        )
        return {
            "trial_id": trial.trial_id,
            "coords": features,
            "targets": list(trial.sequence),
            "length": trial.seq_len,
            "layout": dict(trial.layout),
            "mode": trial.mode,
        }
