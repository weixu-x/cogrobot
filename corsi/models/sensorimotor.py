"""Minimal sensorimotor baseline for freecam_motion_v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from corsi.models.lstm_visual import FrameCNNEncoder


@dataclass
class SensorimotorConfig:
    input_channels: int = 3
    cnn_feature_dim: int = 128
    proprio_dim: int = 0
    proprio_embedding_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    input_image_size: int = 128
    output_dim: int = 2


class FreecamMotionSensorimotorModel(nn.Module):
    """CNN frame encoder plus LSTM head that predicts per-step motion XY."""

    def __init__(self, config: SensorimotorConfig) -> None:
        super().__init__()
        self.config = config
        self.frame_encoder = FrameCNNEncoder(config.input_channels, config.cnn_feature_dim)
        self.proprio_encoder = (
            nn.Sequential(
                nn.Linear(config.proprio_dim, config.proprio_embedding_dim),
                nn.ReLU(),
            )
            if config.proprio_dim > 0
            else None
        )
        rnn_dropout = config.dropout if config.num_layers > 1 else 0.0
        lstm_input_dim = config.cnn_feature_dim + (
            config.proprio_embedding_dim if config.proprio_dim > 0 else 0
        )
        self.temporal = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.output_head = nn.Linear(config.hidden_dim, config.output_dim)

    def _encode_frames(self, frames: Tensor) -> Tensor:
        batch_size, steps, channels, height, width = frames.shape
        flat_frames = frames.reshape(batch_size * steps, channels, height, width)
        if height != self.config.input_image_size or width != self.config.input_image_size:
            flat_frames = F.interpolate(
                flat_frames,
                size=(self.config.input_image_size, self.config.input_image_size),
                mode="bilinear",
                align_corners=False,
            )
        features = self.frame_encoder(flat_frames)
        return features.reshape(batch_size, steps, -1)

    def forward(
        self,
        frames: Tensor,
        frame_lengths: Tensor,
        *,
        proprio: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        features = self._encode_frames(frames)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("proprio tensor is required when proprio_dim > 0")
            if proprio.shape[:2] != frames.shape[:2] or proprio.shape[-1] != self.config.proprio_dim:
                raise ValueError(
                    "proprio must have shape [B, T, proprio_dim] matching frames and config"
                )
            batch_size, steps, _ = proprio.shape
            proprio_features = self.proprio_encoder(proprio.reshape(batch_size * steps, -1))
            proprio_features = proprio_features.reshape(batch_size, steps, -1)
            features = torch.cat([features, proprio_features], dim=-1)

        packed = pack_padded_sequence(
            features,
            frame_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.temporal(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=frames.size(1))
        return {"pred_motion_xy": self.output_head(outputs)}
