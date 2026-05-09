"""Visual encoder-decoder LSTM with optional retention delay manipulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from corsi.models.attention import (
    AttentionConfig,
    AttentionModule,
    CapacityGateConfig,
    LocalAttentionConfig,
    MemoryDecayConfig,
    NoisyAttentionConfig,
    ResponseSuppressionConfig,
    build_attention_config,
)


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
    use_attention: bool = False
    attention_dim: int = 128
    attention_type: str = "global"
    attention_temperature: float = 1.0
    local_attention: LocalAttentionConfig = field(default_factory=LocalAttentionConfig)
    noisy_attention: NoisyAttentionConfig = field(default_factory=NoisyAttentionConfig)
    memory_decay: MemoryDecayConfig = field(default_factory=MemoryDecayConfig)
    capacity_gate: CapacityGateConfig = field(default_factory=CapacityGateConfig)
    response_suppression: ResponseSuppressionConfig = field(default_factory=ResponseSuppressionConfig)
    use_step_embedding: bool = False
    max_decode_steps: int = 6
    step_embedding_dim: int = 16
    delay_mode: str = "none"
    delay_steps: int = 0
    blank_feature_mode: str = "zero_feature"
    return_hidden_traces: bool = False
    return_error_analysis: bool = False

    @property
    def start_token_id(self) -> int:
        return self.num_blocks

    def attention_config(self) -> AttentionConfig:
        def payload(value: Any) -> Dict[str, Any]:
            return dict(value) if isinstance(value, dict) else asdict(value)

        return build_attention_config(
            {
                "attention_type": self.attention_type,
                "attention_temperature": self.attention_temperature,
                "local_attention": payload(self.local_attention),
                "noisy_attention": payload(self.noisy_attention),
                "memory_decay": payload(self.memory_decay),
                "capacity_gate": payload(self.capacity_gate),
                "response_suppression": payload(self.response_suppression),
            }
        )


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
    """CNN encoder + encoder-LSTM + decoder-LSTM sequence recall baseline."""

    def __init__(self, config: VisualLSTMConfig) -> None:
        super().__init__()
        self.config = config

        rnn_dropout = config.dropout if config.num_layers > 1 else 0.0
        decoder_input_dim = config.token_embedding_dim + (
            config.step_embedding_dim if config.use_step_embedding else 0
        )
        self.frame_encoder = FrameCNNEncoder(config.input_channels, config.cnn_feature_dim)
        self.encoder_lstm = nn.LSTM(
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
        self.decoder_lstm = nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.attention = (
            AttentionModule(
                config.attention_config(),
                hidden_dim=config.hidden_dim,
                attention_dim=config.attention_dim,
            )
            if config.use_attention
            else None
        )
        self.output_head = nn.Linear(config.hidden_dim, config.num_blocks)

        # Backward-compatible aliases for any external code that still uses the old names.
        self.encoder = self.encoder_lstm
        self.decoder = self.decoder_lstm

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        remapped = dict(state_dict)
        legacy_attention_key_map = {
            "attention_encoder_proj.weight": "attention.encoder_proj.weight",
            "attention_decoder_proj.weight": "attention.decoder_proj.weight",
            "attention_score.weight": "attention.score.weight",
            "attention_fusion.weight": "attention.fusion.weight",
            "attention_fusion.bias": "attention.fusion.bias",
        }
        for old_key, new_key in legacy_attention_key_map.items():
            if old_key in remapped and new_key not in remapped:
                remapped[new_key] = remapped[old_key]
            if old_key in remapped and new_key in remapped:
                remapped.pop(old_key)
        return super().load_state_dict(remapped, strict=strict)

    def _encode_frames_to_features(self, frames: Tensor) -> Tensor:
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

    @staticmethod
    def _top_hidden(state: tuple[Tensor, Tensor]) -> Tensor:
        return state[0][-1]

    @staticmethod
    def _empty_trace(batch_size: int, hidden_dim: int, device: torch.device) -> Tensor:
        return torch.empty((batch_size, 0, hidden_dim), device=device)

    def _resolve_delay_args(
        self,
        delay_mode: Optional[str],
        delay_steps: Optional[int],
        blank_feature_mode: Optional[str],
        return_hidden_traces: Optional[bool],
    ) -> tuple[str, int, str, bool]:
        resolved_mode = self.config.delay_mode if delay_mode is None else delay_mode
        resolved_steps = self.config.delay_steps if delay_steps is None else int(delay_steps)
        resolved_blank_mode = (
            self.config.blank_feature_mode if blank_feature_mode is None else blank_feature_mode
        )
        resolved_return_traces = (
            self.config.return_hidden_traces
            if return_hidden_traces is None
            else bool(return_hidden_traces)
        )
        if resolved_mode not in {"none", "hold_state", "encoder_blanks"}:
            raise ValueError(f"Unsupported delay_mode: {resolved_mode}")
        if resolved_blank_mode not in {"zero_feature", "zero_image"}:
            raise ValueError(f"Unsupported blank_feature_mode: {resolved_blank_mode}")
        if resolved_steps < 0:
            raise ValueError("delay_steps must be >= 0")
        return resolved_mode, resolved_steps, resolved_blank_mode, resolved_return_traces

    def encode_frames(
        self,
        frames: Tensor,
        frame_lengths: Tensor,
        return_traces: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Dict[str, Tensor]]:
        """
        Args:
            frames: `[B, T, C, H, W]`
            frame_lengths: `[B]`
        Returns:
            encoder_outputs: `[B, T, H]` padded hidden states from the top encoder layer
            encoder_final_state: `(h, c)`
            traces: lightweight encoder trace payload
        """

        embedded = self._encode_frames_to_features(frames)
        packed = pack_padded_sequence(
            embedded,
            frame_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, final_state = self.encoder_lstm(packed)
        encoder_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        traces: Dict[str, Tensor] = {
            "encoder_last_hidden": self._top_hidden(final_state),
        }
        if return_traces:
            traces["encoder_hidden_trace"] = encoder_outputs
        return encoder_outputs, final_state, traces

    def _blank_features(
        self,
        batch_size: int,
        device: torch.device,
        blank_feature_mode: str,
    ) -> tuple[Tensor, Dict[str, float]]:
        if blank_feature_mode == "zero_feature":
            blank_feat = torch.zeros(batch_size, self.config.cnn_feature_dim, device=device)
        elif blank_feature_mode == "zero_image":
            blank_images = torch.zeros(
                (
                    batch_size,
                    self.config.input_channels,
                    self.config.input_image_size,
                    self.config.input_image_size,
                ),
                device=device,
            )
            blank_feat = self.frame_encoder(blank_images)
        else:
            raise ValueError(f"Unsupported blank_feature_mode: {blank_feature_mode}")

        blank_stats = {
            "blank_feature_l2_norm": float(blank_feat.norm(p=2, dim=1).mean().item()),
            "blank_feature_mean_abs": float(blank_feat.abs().mean().item()),
        }
        return blank_feat, blank_stats

    def apply_delay(
        self,
        encoder_state: tuple[Tensor, Tensor],
        batch_size: int,
        device: torch.device,
        delay_steps: int,
        delay_mode: str,
        *,
        blank_feature_mode: str = "zero_feature",
        return_traces: bool = False,
    ) -> tuple[tuple[Tensor, Tensor], Dict[str, Any]]:
        """
        Args:
            encoder_state: encoder final `(h, c)` after observation frames
        Returns:
            delayed_encoder_state and optional trace metadata
        """

        trace_payload: Dict[str, Any] = {
            "delay_mode": delay_mode,
            "delay_steps": int(delay_steps),
            "blank_feature_mode": blank_feature_mode,
        }

        if delay_mode in {"none", "hold_state"} or delay_steps == 0:
            if return_traces:
                trace_payload["delay_hidden_trace"] = self._empty_trace(
                    batch_size=batch_size,
                    hidden_dim=self.config.hidden_dim,
                    device=device,
                )
            trace_payload["recall_start_hidden"] = self._top_hidden(encoder_state)
            trace_payload["blank_feature_l2_norm"] = 0.0
            trace_payload["blank_feature_mean_abs"] = 0.0
            return encoder_state, trace_payload

        blank_feat, blank_stats = self._blank_features(batch_size, device, blank_feature_mode)
        state = encoder_state
        delay_hidden_trace = []

        for _ in range(delay_steps):
            _, state = self.encoder_lstm(blank_feat.unsqueeze(1), state)
            if return_traces:
                delay_hidden_trace.append(self._top_hidden(state))

        trace_payload.update(blank_stats)
        trace_payload["recall_start_hidden"] = self._top_hidden(state)
        if return_traces:
            trace_payload["delay_hidden_trace"] = (
                torch.stack(delay_hidden_trace, dim=1)
                if delay_hidden_trace
                else self._empty_trace(batch_size, self.config.hidden_dim, device)
            )
        return state, trace_payload

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

    def _encoder_mask(self, encoder_outputs: Tensor, frame_lengths: Tensor) -> Tensor:
        max_time = encoder_outputs.size(1)
        positions = torch.arange(max_time, device=frame_lengths.device).unsqueeze(0)
        return positions < frame_lengths.unsqueeze(1)

    def _apply_attention(
        self,
        step_hidden: Tensor,
        encoder_outputs: Tensor,
        encoder_mask: Tensor,
        step_index: int,
    ) -> tuple[Tensor, Tensor, Dict[str, Tensor]]:
        if not self.config.use_attention or self.attention is None:
            batch_size, time_steps, _ = encoder_outputs.shape
            empty_weights = torch.zeros(batch_size, time_steps, device=step_hidden.device)
            empty_diag = {
                "attention_entropy": torch.zeros(batch_size, device=step_hidden.device),
                "attention_peak_index": torch.zeros(batch_size, dtype=torch.long, device=step_hidden.device),
                "attention_peak_displacement": torch.zeros(batch_size, device=step_hidden.device),
            }
            return step_hidden, empty_weights, empty_diag

        return self.attention(
            step_hidden,
            encoder_outputs,
            encoder_mask,
            step_index=step_index,
            training=self.training,
        )

    def _apply_response_suppression(
        self,
        logits: Tensor,
        previous_outputs_mask: Tensor,
    ) -> Tensor:
        suppression = self.config.response_suppression
        if not suppression.enabled or suppression.beta <= 0:
            return logits
        mask = previous_outputs_mask.bool()
        if suppression.hard_mask:
            return logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        return logits - float(suppression.beta) * previous_outputs_mask.float()

    def _decode_step(
        self,
        *,
        input_tokens: Tensor,
        step_index: int,
        state: tuple[Tensor, Tensor],
        encoder_outputs: Tensor,
        encoder_mask: Tensor,
        previous_outputs_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor], Tensor, Dict[str, Tensor]]:
        decoder_inputs = self._decode_input_embedding(input_tokens.unsqueeze(1), step_index)
        decoder_output, next_state = self.decoder_lstm(decoder_inputs, state)
        step_hidden = decoder_output[:, -1, :]
        fused_hidden, attention_weights, attention_diagnostics = self._apply_attention(
            step_hidden,
            encoder_outputs,
            encoder_mask,
            step_index,
        )
        step_logits = self.output_head(fused_hidden)
        if previous_outputs_mask is not None:
            step_logits = self._apply_response_suppression(step_logits, previous_outputs_mask)
        return step_logits, fused_hidden, next_state, attention_weights, attention_diagnostics

    def decode_autoregressive(
        self,
        encoder_outputs: Tensor,
        frame_lengths: Tensor,
        encoder_state: tuple[Tensor, Tensor],
        targets: Optional[Tensor] = None,
        target_lengths: Optional[Tensor] = None,
        *,
        max_steps: Optional[int] = None,
        return_traces: bool = False,
        teacher_forcing_ratio: float = 1.0,
    ) -> Dict[str, Tensor]:
        """
        Keeps the current encoder-decoder LSTM recall logic.
        Teacher forcing is used when `targets` are provided; otherwise greedy decoding is used.
        """

        if targets is not None and (
            not self.config.use_attention
            and not self.config.use_step_embedding
            and teacher_forcing_ratio >= 1.0
        ):
            decoder_inputs = self._teacher_forcing_inputs(targets)
            decoder_emb = self.token_embedding(decoder_inputs)
            decoder_outputs, final_state = self.decoder_lstm(decoder_emb, encoder_state)
            payload: Dict[str, Tensor] = {
                "logits": self.output_head(decoder_outputs),
                "decoder_final_hidden": self._top_hidden(final_state),
            }
            if return_traces:
                payload["decoder_hidden_trace"] = decoder_outputs
            return payload

        if max_steps is None:
            if target_lengths is not None:
                max_steps = int(target_lengths.max().item())
            else:
                raise ValueError("decode_autoregressive requires target_lengths or max_steps for greedy decoding")

        hidden, cell = encoder_state
        batch_size = hidden.size(1)
        encoder_mask = self._encoder_mask(encoder_outputs, frame_lengths)
        current_tokens = torch.full(
            (batch_size,),
            fill_value=self.config.start_token_id,
            dtype=torch.long,
            device=hidden.device,
        )
        predictions = []
        logits_steps = []
        decoder_hidden_trace = []
        attention_trace = []
        attention_entropy_trace = []
        attention_peak_displacement_trace = []
        previous_outputs_mask = torch.zeros(
            batch_size,
            self.config.num_blocks,
            dtype=torch.bool,
            device=hidden.device,
        )

        if teacher_forcing_ratio < 0.0 or teacher_forcing_ratio > 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")

        previous_targets = None
        if targets is not None and targets.size(1) > 1:
            previous_targets = targets[:, :-1].clone()
            previous_targets = torch.where(
                previous_targets == self.config.target_pad_value,
                torch.full_like(previous_targets, self.config.start_token_id),
                previous_targets,
            )

        for step_index in range(max_steps):
            step_logits, step_hidden, (hidden, cell), attention_weights, attention_diagnostics = self._decode_step(
                input_tokens=current_tokens,
                step_index=step_index,
                state=(hidden, cell),
                encoder_outputs=encoder_outputs,
                encoder_mask=encoder_mask,
                previous_outputs_mask=previous_outputs_mask,
            )
            next_tokens = step_logits.argmax(dim=-1)
            predictions.append(next_tokens)
            logits_steps.append(step_logits)
            emitted_tokens = next_tokens
            if targets is not None and teacher_forcing_ratio >= 1.0 and step_index < targets.size(1):
                emitted_tokens = targets[:, step_index]
            valid_emitted = (emitted_tokens >= 0) & (emitted_tokens < self.config.num_blocks)
            previous_outputs_mask.scatter_(
                1,
                emitted_tokens.clamp(0, self.config.num_blocks - 1).unsqueeze(1),
                valid_emitted.unsqueeze(1),
            )
            if return_traces:
                decoder_hidden_trace.append(step_hidden)
                attention_trace.append(attention_weights)
                attention_entropy_trace.append(attention_diagnostics["attention_entropy"])
                attention_peak_displacement_trace.append(
                    attention_diagnostics["attention_peak_displacement"]
                )
            if targets is not None and step_index + 1 < max_steps:
                if teacher_forcing_ratio >= 1.0:
                    current_tokens = previous_targets[:, step_index]
                elif teacher_forcing_ratio <= 0.0:
                    current_tokens = next_tokens
                else:
                    teacher_mask = (
                        torch.rand(batch_size, device=hidden.device) < teacher_forcing_ratio
                    )
                    gold_tokens = previous_targets[:, step_index]
                    current_tokens = torch.where(teacher_mask, gold_tokens, next_tokens)
            else:
                current_tokens = next_tokens

        payload = {
            "predictions": torch.stack(predictions, dim=1),
            "logits": torch.stack(logits_steps, dim=1),
            "decoder_final_hidden": hidden[-1],
            "attention_weights": torch.stack(attention_trace, dim=1) if attention_trace else None,
        }
        if return_traces:
            payload["decoder_hidden_trace"] = torch.stack(decoder_hidden_trace, dim=1)
            payload["attention_trace"] = torch.stack(attention_trace, dim=1)
            payload["attention_entropy"] = torch.stack(attention_entropy_trace, dim=1)
            payload["attention_peak_displacement"] = torch.stack(attention_peak_displacement_trace, dim=1)
        return payload

    def _compose_hidden_traces(
        self,
        *,
        encoder_traces: Dict[str, Tensor],
        delay_traces: Dict[str, Any],
        decoder_outputs: Dict[str, Tensor],
        frame_lengths: Tensor,
        target_lengths: Optional[Tensor],
    ) -> Dict[str, Any]:
        metadata = {
            "delay_mode": delay_traces["delay_mode"],
            "delay_steps": delay_traces["delay_steps"],
            "blank_feature_mode": delay_traces["blank_feature_mode"],
            "frame_lengths": frame_lengths.detach().clone(),
            "target_lengths": None if target_lengths is None else target_lengths.detach().clone(),
            "encoder_time_lengths": frame_lengths.detach().clone()
            + (
                delay_traces["delay_steps"]
                if delay_traces["delay_mode"] == "encoder_blanks"
                else 0
            ),
            "blank_feature_l2_norm": delay_traces["blank_feature_l2_norm"],
            "blank_feature_mean_abs": delay_traces["blank_feature_mean_abs"],
        }
        hidden_traces: Dict[str, Any] = {
            "encoder_last_hidden": encoder_traces["encoder_last_hidden"],
            "delay_hidden_trace": delay_traces.get("delay_hidden_trace"),
            "decoder_hidden_trace": decoder_outputs.get("decoder_hidden_trace"),
            "attention_trace": decoder_outputs.get("attention_trace"),
            "attention_entropy": decoder_outputs.get("attention_entropy"),
            "attention_peak_displacement": decoder_outputs.get("attention_peak_displacement"),
            "recall_start_hidden": delay_traces["recall_start_hidden"],
            "metadata": metadata,
        }
        if "encoder_hidden_trace" in encoder_traces:
            hidden_traces["encoder_hidden_trace"] = encoder_traces["encoder_hidden_trace"]
        return hidden_traces

    def forward(
        self,
        frames: Tensor,
        frame_lengths: Tensor,
        targets: Optional[Tensor] = None,
        target_lengths: Optional[Tensor] = None,
        delay_mode: Optional[str] = "none",
        delay_steps: Optional[int] = 0,
        return_hidden_traces: bool = False,
        blank_feature_mode: Optional[str] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> Dict[str, Any]:
        resolved_mode, resolved_steps, resolved_blank_mode, resolved_return_traces = self._resolve_delay_args(
            delay_mode=delay_mode,
            delay_steps=delay_steps,
            blank_feature_mode=blank_feature_mode,
            return_hidden_traces=return_hidden_traces,
        )
        encoder_outputs, encoder_state, encoder_traces = self.encode_frames(
            frames,
            frame_lengths,
            return_traces=resolved_return_traces,
        )
        delayed_state, delay_traces = self.apply_delay(
            encoder_state=encoder_state,
            batch_size=frames.size(0),
            device=frames.device,
            delay_steps=resolved_steps,
            delay_mode=resolved_mode,
            blank_feature_mode=resolved_blank_mode,
            return_traces=resolved_return_traces,
        )
        decoder_outputs = self.decode_autoregressive(
            encoder_outputs,
            frame_lengths,
            delayed_state,
            targets=targets,
            target_lengths=target_lengths,
            return_traces=resolved_return_traces,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        payload: Dict[str, Any] = {"logits": decoder_outputs["logits"]}
        if decoder_outputs.get("attention_weights") is not None:
            payload["attention_weights"] = decoder_outputs["attention_weights"]
        if resolved_return_traces:
            payload["hidden_traces"] = self._compose_hidden_traces(
                encoder_traces=encoder_traces,
                delay_traces=delay_traces,
                decoder_outputs=decoder_outputs,
                frame_lengths=frame_lengths,
                target_lengths=target_lengths,
            )
        return payload

    @torch.no_grad()
    def greedy_decode(
        self,
        frames: Tensor,
        frame_lengths: Tensor,
        max_steps: Optional[int] = None,
        *,
        target_lengths: Optional[Tensor] = None,
        delay_mode: Optional[str] = "none",
        delay_steps: Optional[int] = 0,
        return_hidden_traces: bool = False,
        blank_feature_mode: Optional[str] = None,
    ) -> Tensor | Dict[str, Any]:
        resolved_mode, resolved_steps, resolved_blank_mode, resolved_return_traces = self._resolve_delay_args(
            delay_mode=delay_mode,
            delay_steps=delay_steps,
            blank_feature_mode=blank_feature_mode,
            return_hidden_traces=return_hidden_traces,
        )
        if max_steps is None:
            if target_lengths is not None:
                max_steps = int(target_lengths.max().item())
            else:
                max_steps = int(frame_lengths.max().item())

        encoder_outputs, encoder_state, encoder_traces = self.encode_frames(
            frames,
            frame_lengths,
            return_traces=resolved_return_traces,
        )
        delayed_state, delay_traces = self.apply_delay(
            encoder_state=encoder_state,
            batch_size=frames.size(0),
            device=frames.device,
            delay_steps=resolved_steps,
            delay_mode=resolved_mode,
            blank_feature_mode=resolved_blank_mode,
            return_traces=resolved_return_traces,
        )
        decoder_outputs = self.decode_autoregressive(
            encoder_outputs,
            frame_lengths,
            delayed_state,
            targets=None,
            target_lengths=target_lengths,
            max_steps=max_steps,
            return_traces=resolved_return_traces,
        )
        predictions = decoder_outputs["predictions"]
        if not resolved_return_traces:
            return predictions
        return {
            "predictions": predictions,
            "hidden_traces": self._compose_hidden_traces(
                encoder_traces=encoder_traces,
                delay_traces=delay_traces,
                decoder_outputs=decoder_outputs,
                frame_lengths=frame_lengths,
                target_lengths=target_lengths,
            ),
        }
