"""Minimal coordinate-based encoder-decoder LSTM for Corsi sequence recall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class CoordLSTMConfig:
    input_dim: int = 4
    coord_embedding_dim: int = 64
    token_embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    num_blocks: int = 9
    target_pad_value: int = -100

    @property
    def start_token_id(self) -> int:
        return self.num_blocks


class CoordinateSeq2SeqLSTM(nn.Module):
    """Encoder-decoder LSTM that predicts the full Corsi block sequence."""

    def __init__(self, config: CoordLSTMConfig) -> None:
        super().__init__()
        self.config = config

        encoder_dropout = config.dropout if config.num_layers > 1 else 0.0
        decoder_dropout = config.dropout if config.num_layers > 1 else 0.0

        self.coord_embed = nn.Sequential(
            nn.Linear(config.input_dim, config.coord_embedding_dim),
            nn.ReLU(),
            nn.Linear(config.coord_embedding_dim, config.coord_embedding_dim),
            nn.ReLU(),
        )
        self.encoder = nn.LSTM(
            input_size=config.coord_embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=encoder_dropout,
            batch_first=True,
        )
        self.token_embedding = nn.Embedding(config.num_blocks + 1, config.token_embedding_dim)
        self.decoder = nn.LSTM(
            input_size=config.token_embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=decoder_dropout,
            batch_first=True,
        )
        self.output_head = nn.Linear(config.hidden_dim, config.num_blocks)

    def encode(self, coords: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        embedded = self.coord_embed(coords)
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

    def forward(self, coords: Tensor, lengths: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        hidden, cell = self.encode(coords, lengths)
        decoder_inputs = self._teacher_forcing_inputs(targets)
        decoder_emb = self.token_embedding(decoder_inputs)
        decoder_outputs, _ = self.decoder(decoder_emb, (hidden, cell))
        logits = self.output_head(decoder_outputs)
        return {"logits": logits}

    @torch.no_grad()
    def greedy_decode(self, coords: Tensor, lengths: Tensor, max_steps: Optional[int] = None) -> Tensor:
        hidden, cell = self.encode(coords, lengths)
        batch_size = coords.size(0)
        decode_steps = max_steps if max_steps is not None else int(lengths.max().item())

        current_tokens = torch.full(
            (batch_size, 1),
            fill_value=self.config.start_token_id,
            dtype=torch.long,
            device=coords.device,
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
