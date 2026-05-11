"""Minimal visual encoder-decoder LSTM for robosuite Corsi sequence recall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from corsi.heatmaps import decode_heatmap_argmax, nearest_block_decode, standard_block_heatmap_xy


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
    output_type: str = "index"
    use_attention: bool = False
    use_step_embedding: bool = False
    max_decode_steps: int = 6
    step_embedding_dim: int = 16
    heatmap_size: int = 32

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


class HeatmapHead(nn.Module):
    """Projects decoder hidden states into spatial heatmap logits."""

    def __init__(self, hidden_dim: int, heatmap_size: int) -> None:
        super().__init__()
        self.heatmap_size = int(heatmap_size)
        self.base_size = 4
        self.base_channels = 64
        self.proj = nn.Linear(hidden_dim, self.base_channels * self.base_size * self.base_size)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(self.base_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        batch_size, steps, hidden_dim = hidden.shape
        flat_hidden = hidden.reshape(batch_size * steps, hidden_dim)
        features = self.proj(flat_hidden).reshape(
            batch_size * steps,
            self.base_channels,
            self.base_size,
            self.base_size,
        )
        logits = self.upsample(features)
        if logits.size(-1) != self.heatmap_size or logits.size(-2) != self.heatmap_size:
            logits = F.interpolate(
                logits,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits.reshape(batch_size, steps, 1, self.heatmap_size, self.heatmap_size)


class VisualSeq2SeqLSTM(nn.Module):
    """CNN encoder + LSTM decoder that predicts the full Corsi block sequence."""

    def __init__(self, config: VisualLSTMConfig) -> None:
        super().__init__()
        self.config = config
        if config.output_type not in {"index", "heatmap"}:
            raise ValueError("output_type must be 'index' or 'heatmap'")
        if config.use_attention:
            raise ValueError("Heatmap branch does not support encoder-decoder attention")
        if config.heatmap_size < 2:
            raise ValueError("heatmap_size must be >= 2")

        rnn_dropout = config.dropout if config.num_layers > 1 else 0.0
        decoder_input_dim = config.token_embedding_dim + (
            config.step_embedding_dim if config.use_step_embedding else 0
        )
        self.frame_encoder = FrameCNNEncoder(config.input_channels, config.cnn_feature_dim)
        self.encoder = nn.LSTM(
            input_size=config.cnn_feature_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.token_embedding = nn.Embedding(config.num_blocks + 1, config.token_embedding_dim)
        self.step_embedding = (
            nn.Embedding(config.max_decode_steps, config.step_embedding_dim)
            if config.use_step_embedding
            else None
        )
        self.decoder = nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.output_head = nn.Linear(config.hidden_dim, config.num_blocks)
        self.heatmap_head = HeatmapHead(config.hidden_dim, config.heatmap_size)
        self.register_buffer(
            "block_heatmap_xy",
            torch.as_tensor(
                standard_block_heatmap_xy(config.heatmap_size, num_blocks=config.num_blocks),
                dtype=torch.float32,
            ),
            persistent=False,
        )

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

    def _decode_input_embedding(self, input_tokens: Tensor, step_index: int) -> Tensor:
        token_emb = self.token_embedding(input_tokens)
        if not self.config.use_step_embedding or self.step_embedding is None:
            return token_emb
        clamped_step = min(step_index, self.config.max_decode_steps - 1)
        step_ids = torch.full_like(input_tokens, fill_value=clamped_step)
        step_emb = self.step_embedding(step_ids)
        return torch.cat([token_emb, step_emb], dim=-1)

    def _teacher_forcing_embeddings(self, decoder_inputs: Tensor) -> Tensor:
        steps = [
            self._decode_input_embedding(decoder_inputs[:, step_index : step_index + 1], step_index)
            for step_index in range(decoder_inputs.size(1))
        ]
        return torch.cat(steps, dim=1)

    def forward(self, frames: Tensor, lengths: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        hidden, cell = self.encode(frames, lengths)
        decoder_inputs = self._teacher_forcing_inputs(targets)
        decoder_emb = self._teacher_forcing_embeddings(decoder_inputs)
        decoder_outputs, _ = self.decoder(decoder_emb, (hidden, cell))
        if self.config.output_type == "index":
            return {
                "logits": self.output_head(decoder_outputs),
                "heatmap_logits": None,
            }
        return {
            "logits": None,
            "heatmap_logits": self.heatmap_head(decoder_outputs),
        }

    @torch.no_grad()
    def greedy_decode(self, frames: Tensor, lengths: Tensor, max_steps: Optional[int] = None) -> Tensor | Dict[str, Tensor]:
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
        heatmap_logits_steps = []
        heatmap_xy_steps = []
        spatial_error_steps = []

        for step_index in range(decode_steps):
            decoder_emb = self._decode_input_embedding(current_tokens[:, -1:], step_index)
            decoder_output, (hidden, cell) = self.decoder(decoder_emb, (hidden, cell))
            if self.config.output_type == "index":
                step_logits = self.output_head(decoder_output[:, -1, :])
                next_tokens = step_logits.argmax(dim=-1)
            else:
                step_heatmap_logits = self.heatmap_head(decoder_output)
                step_xy = decode_heatmap_argmax(step_heatmap_logits)[:, -1]
                next_tokens, step_distances = nearest_block_decode(
                    step_xy,
                    block_xy=self.block_heatmap_xy,
                )
                heatmap_logits_steps.append(step_heatmap_logits[:, -1])
                heatmap_xy_steps.append(step_xy)
                spatial_error_steps.append(step_distances)
            predictions.append(next_tokens)
            current_tokens = torch.cat([current_tokens, next_tokens.unsqueeze(1)], dim=1)

        stacked_predictions = torch.stack(predictions, dim=1)
        if self.config.output_type == "index":
            return stacked_predictions
        return {
            "predictions": stacked_predictions,
            "heatmap_logits": torch.stack(heatmap_logits_steps, dim=1),
            "heatmap_xy": torch.stack(heatmap_xy_steps, dim=1),
            "nearest_distances": torch.stack(spatial_error_steps, dim=1),
        }
