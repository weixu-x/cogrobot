"""Model definitions for Corsi experiments."""

from .lstm_coord import CoordLSTMConfig, CoordinateSeq2SeqLSTM
from .lstm_visual import VisualLSTMConfig, VisualSeq2SeqLSTM
from .sensorimotor import FreecamMotionSensorimotorModel, SensorimotorConfig

__all__ = [
    "CoordLSTMConfig",
    "CoordinateSeq2SeqLSTM",
    "FreecamMotionSensorimotorModel",
    "SensorimotorConfig",
    "VisualLSTMConfig",
    "VisualSeq2SeqLSTM",
]
