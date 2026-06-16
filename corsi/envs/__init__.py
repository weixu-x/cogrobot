"""Task and sequence utilities for the Corsi project."""

from .sequence_generator import (
    CorsiTrial,
    STANDARD_CORSI_BLOCK_SIZE,
    STANDARD_CORSI_BOARD_SIZE,
    STANDARD_CORSI_LAYOUT_IMAGE,
    STANDARD_CORSI_ROBOSUITE_TARGET_WIDTH,
    build_coordinate_features,
    canonical_board_layout,
    compute_delta_coords,
    generate_block_sequence,
    generate_coordinate_trial,
    generate_trial_collection,
    standard_corsi_layout,
    standard_corsi_robosuite_board_size,
    standard_corsi_robosuite_layout,
    transform_layout,
)

__all__ = [
    "CorsiTrial",
    "DEFAULT_OFFLINE_CAMERAS",
    "DEFAULT_ONLINE_RENDER_CAMERA",
    "PandaDexRH",
    "STANDARD_CORSI_BLOCK_SIZE",
    "STANDARD_CORSI_BOARD_SIZE",
    "STANDARD_CORSI_LAYOUT_IMAGE",
    "STANDARD_CORSI_ROBOSUITE_TARGET_WIDTH",
    "build_coordinate_features",
    "canonical_board_layout",
    "compute_delta_coords",
    "generate_block_sequence",
    "generate_coordinate_trial",
    "generate_trial_collection",
    "standard_corsi_layout",
    "standard_corsi_robosuite_board_size",
    "standard_corsi_robosuite_layout",
    "transform_layout",
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
