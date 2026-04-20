"""Minimal visual encoder-decoder LSTM for robosuite Corsi sequence recall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class VisualLSTMConfig:
    input_channels: int = 3
    cnn_feature_dim: int = 128
    token_embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    num_blocks: int = 9
    target_pad_value: int = -100
    input_image_size: int = 128

    @property
    def start_token_id(self) -> int:
        return self.num_blocks


class FrameCNNEncoder(nn.Module):
    """Lightweight per-frame CNN that compresses RGB frames into feature vectors."""

    def __init__(self, input_channels: int, feature_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, frames: Tensor) -> Tensor:
        return self.backbone(frames).flatten(start_dim=1)


class VisualSeq2SeqLSTM(nn.Module):
    """CNN encoder + LSTM decoder that predicts the full Corsi block sequence."""

    def __init__(self, config: VisualLSTMConfig) -> None:
        super().__init__()
        self.config = config

        rnn_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.frame_encoder = FrameCNNEncoder(config.input_channels, config.cnn_feature_dim)
        self.encoder = nn.LSTM(
            input_size=config.cnn_feature_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.token_embedding = nn.Embedding(config.num_blocks + 1, config.token_embedding_dim)
        self.decoder = nn.LSTM(
            input_size=config.token_embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.output_head = nn.Linear(config.hidden_dim, config.num_blocks)

    def _encode_frames(self, frames: Tensor) -> Tensor:
        batch_size, seq_len, channels, height, width = frames.shape
        flat_frames = frames.reshape(batch_size * seq_len, channels, height, width)
        if height != self.config.input_image_size or width != self.config.input_image_size:
            flat_frames = F.interpolate(
                flat_frames,
                size=(self.config.input_image_size, self.config.input_image_size),
                mode="bilinear",
                align_corners=False,
            )
        encoded = self.frame_encoder(flat_frames)
        return encoded.reshape(batch_size, seq_len, -1)

    def encode(self, frames: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        embedded = self._encode_frames(frames)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, cell) = self.encoder(packed)
        return hidden, cell

    def _teacher_forcing_inputs(self, targets: Tensor) -> Tensor:
        batch_size, seq_len = targets.shape
        decoder_inputs = torch.full(
            (batch_size, seq_len),
            fill_value=self.config.start_token_id,
            dtype=torch.long,
            device=targets.device,
        )
        if seq_len > 1:
            previous_targets = targets[:, :-1].clone()
            previous_targets = torch.where(
                previous_targets == self.config.target_pad_value,
                torch.full_like(previous_targets, self.config.start_token_id),
                previous_targets,
            )
            decoder_inputs[:, 1:] = previous_targets
        return decoder_inputs

    def forward(self, frames: Tensor, lengths: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        hidden, cell = self.encode(frames, lengths)
        decoder_inputs = self._teacher_forcing_inputs(targets)
        decoder_emb = self.token_embedding(decoder_inputs)
        decoder_outputs, _ = self.decoder(decoder_emb, (hidden, cell))
        logits = self.output_head(decoder_outputs)
        return {"logits": logits}

    @torch.no_grad()
    def greedy_decode(self, frames: Tensor, lengths: Tensor, max_steps: Optional[int] = None) -> Tensor:
        hidden, cell = self.encode(frames, lengths)
        batch_size = frames.size(0)
        decode_steps = max_steps if max_steps is not None else int(lengths.max().item())

        current_tokens = torch.full(
            (batch_size, 1),
            fill_value=self.config.start_token_id,
            dtype=torch.long,
            device=frames.device,
        )
        predictions = []

        for _ in range(decode_steps):
            decoder_emb = self.token_embedding(current_tokens[:, -1:])
            decoder_output, (hidden, cell) = self.decoder(decoder_emb, (hidden, cell))
            step_logits = self.output_head(decoder_output[:, -1, :])
            next_tokens = step_logits.argmax(dim=-1)
            predictions.append(next_tokens)
            current_tokens = torch.cat([current_tokens, next_tokens.unsqueeze(1)], dim=1)

        return torch.stack(predictions, dim=1)
