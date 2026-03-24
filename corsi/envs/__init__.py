"""Task and sequence utilities for the Corsi project."""

from .sequence_generator import (
    CorsiTrial,
    build_coordinate_features,
    canonical_board_layout,
    compute_delta_coords,
    generate_block_sequence,
    generate_coordinate_trial,
    generate_trial_collection,
)

__all__ = [
    "CorsiTrial",
    "DEFAULT_OFFLINE_CAMERAS",
    "DEFAULT_ONLINE_RENDER_CAMERA",
    "PandaDexRH",
    "build_coordinate_features",
    "canonical_board_layout",
    "compute_delta_coords",
    "generate_block_sequence",
    "generate_coordinate_trial",
    "generate_trial_collection",
]


def __getattr__(name):
    if name in {
        "DEFAULT_OFFLINE_CAMERAS",
        "DEFAULT_ONLINE_RENDER_CAMERA",
        "PandaDexRH",
        "create_env",
        "default_save_root",
        "rollout_sequence_offline",
        "rollout_sequence_online",
    }:
        from . import robosuite_corsi

        return getattr(robosuite_corsi, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
