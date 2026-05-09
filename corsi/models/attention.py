"""Cognitively constrained attention modules for visual Corsi recall."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn


ATTENTION_TYPES = {
    "global",
    "local_distance",
    "local_gaussian",
    "local_window",
    "noisy_global",
    "decay_global",
    "capacity_global",
    "local_noisy",
    "local_decay",
    "cognitive_full",
}


@dataclass
class LocalAttentionConfig:
    enabled: bool = False
    lambda_distance: float = 0.0
    gaussian_bias: bool = False
    gaussian_sigma: float = 1.5
    window_size: Optional[int] = None


@dataclass
class NoisyAttentionConfig:
    enabled: bool = False
    noise_std: float = 0.0
    train_noise_std: Optional[float] = None
    eval_noise_std: Optional[float] = None
    apply_during_training: bool = True
    apply_during_eval: bool = False


@dataclass
class MemoryDecayConfig:
    enabled: bool = False
    decay_type: str = "none"
    alpha: float = 0.0


@dataclass
class CapacityGateConfig:
    enabled: bool = False
    max_memory_slots: Optional[int] = None
    strategy: str = "recent"


@dataclass
class ResponseSuppressionConfig:
    enabled: bool = False
    beta: float = 0.0
    hard_mask: bool = False


@dataclass
class AttentionConfig:
    attention_type: str = "global"
    attention_temperature: float = 1.0
    local_attention: LocalAttentionConfig = field(default_factory=LocalAttentionConfig)
    noisy_attention: NoisyAttentionConfig = field(default_factory=NoisyAttentionConfig)
    memory_decay: MemoryDecayConfig = field(default_factory=MemoryDecayConfig)
    capacity_gate: CapacityGateConfig = field(default_factory=CapacityGateConfig)
    response_suppression: ResponseSuppressionConfig = field(default_factory=ResponseSuppressionConfig)


def _nested_dataclass(cls, value: Any):
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls(**value)
    return cls()


def build_attention_config(value: Any = None, **overrides: Any) -> AttentionConfig:
    """Build an AttentionConfig from nested dictionaries while tolerating missing keys."""

    payload: Dict[str, Any] = {}
    if isinstance(value, AttentionConfig):
        payload.update(value.__dict__)
    elif isinstance(value, dict):
        payload.update(value)
    payload.update({key: item for key, item in overrides.items() if item is not None})

    config = AttentionConfig(
        attention_type=str(payload.get("attention_type", "global")),
        attention_temperature=float(payload.get("attention_temperature", 1.0)),
        local_attention=_nested_dataclass(
            LocalAttentionConfig, payload.get("local_attention")
        ),
        noisy_attention=_nested_dataclass(
            NoisyAttentionConfig, payload.get("noisy_attention")
        ),
        memory_decay=_nested_dataclass(MemoryDecayConfig, payload.get("memory_decay")),
        capacity_gate=_nested_dataclass(CapacityGateConfig, payload.get("capacity_gate")),
        response_suppression=_nested_dataclass(
            ResponseSuppressionConfig, payload.get("response_suppression")
        ),
    )
    if config.attention_type not in ATTENTION_TYPES:
        raise ValueError(f"Unsupported attention_type: {config.attention_type}")
    if config.attention_temperature <= 0:
        raise ValueError("attention_temperature must be > 0")
    if config.local_attention.gaussian_sigma <= 0:
        raise ValueError("local_attention.gaussian_sigma must be > 0")
    if config.capacity_gate.strategy != "recent":
        raise ValueError("Only capacity_gate.strategy='recent' is currently supported")
    return config


class AttentionModule(nn.Module):
    """Bahdanau-style attention with optional cognitive retrieval constraints."""

    def __init__(
        self,
        config: AttentionConfig | Dict[str, Any],
        *,
        hidden_dim: int,
        attention_dim: int,
    ) -> None:
        super().__init__()
        self.config = build_attention_config(config)
        self.encoder_proj = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.decoder_proj = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.score = nn.Linear(attention_dim, 1, bias=False)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def _step_tensor(self, step_index: int | Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        if step_index is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if isinstance(step_index, int):
            return torch.full((batch_size,), step_index, dtype=torch.long, device=device)
        return step_index.to(device=device, dtype=torch.long).reshape(batch_size)

    def _mode_flags(self) -> Dict[str, bool]:
        attention_type = self.config.attention_type
        local = self.config.local_attention
        noisy = self.config.noisy_attention
        decay = self.config.memory_decay
        capacity = self.config.capacity_gate
        return {
            "distance": attention_type in {"local_distance", "local_noisy", "local_decay", "cognitive_full"}
            or (local.enabled and local.lambda_distance > 0),
            "gaussian": attention_type == "local_gaussian" or (local.enabled and local.gaussian_bias),
            "window": attention_type == "local_window" or (local.enabled and local.window_size is not None),
            "noise": attention_type in {"noisy_global", "local_noisy", "cognitive_full"} or noisy.enabled,
            "decay": attention_type in {"decay_global", "local_decay", "cognitive_full"} or decay.enabled,
            "capacity": attention_type == "capacity_global" or capacity.enabled,
        }

    def _apply_noise(self, scores: Tensor) -> Tensor:
        noise = self.config.noisy_attention
        train_std = noise.noise_std if noise.train_noise_std is None else noise.train_noise_std
        eval_std = noise.noise_std if noise.eval_noise_std is None else noise.eval_noise_std
        if self.training and noise.apply_during_training and train_std > 0:
            return scores + torch.randn_like(scores) * float(train_std)
        if (not self.training) and noise.apply_during_eval and eval_std > 0:
            return scores + torch.randn_like(scores) * float(eval_std)
        return scores

    def forward(
        self,
        decoder_state: Tensor,
        encoder_outputs: Tensor,
        encoder_mask: Tensor,
        step_index: int | Tensor | None = None,
        training: bool = False,
    ) -> tuple[Tensor, Tensor, Dict[str, Tensor]]:
        del training
        batch_size, time_steps, _ = encoder_outputs.shape
        device = encoder_outputs.device
        step = self._step_tensor(step_index, batch_size, device)
        positions = torch.arange(time_steps, device=device).unsqueeze(0)
        distance = (positions - step.unsqueeze(1)).abs().float()

        encoder_proj = self.encoder_proj(encoder_outputs)
        decoder_proj = self.decoder_proj(decoder_state).unsqueeze(1)
        energy = torch.tanh(encoder_proj + decoder_proj)
        scores = self.score(energy).squeeze(-1)
        scores = scores / float(self.config.attention_temperature)

        flags = self._mode_flags()
        if flags["distance"]:
            scores = scores - float(self.config.local_attention.lambda_distance) * distance
        if flags["gaussian"]:
            sigma = float(self.config.local_attention.gaussian_sigma)
            scores = scores - (distance.pow(2) / (2.0 * sigma * sigma))
        if flags["decay"]:
            decay = self.config.memory_decay
            if decay.decay_type not in {"none", "recency", "distance"}:
                raise ValueError(f"Unsupported memory_decay.decay_type: {decay.decay_type}")
            alpha = float(decay.alpha)
            if alpha > 0 and decay.decay_type == "recency":
                recency_age = (encoder_mask.sum(dim=1, keepdim=True) - 1 - positions).clamp_min(0).float()
                scores = scores - alpha * recency_age
            elif alpha > 0 and decay.decay_type == "distance":
                scores = scores - alpha * distance
        if flags["noise"]:
            scores = self._apply_noise(scores)

        combined_mask = encoder_mask.bool()
        if flags["window"]:
            window_size = self.config.local_attention.window_size
            if window_size is not None:
                combined_mask = combined_mask & (distance <= int(window_size))
        if flags["capacity"]:
            slots = self.config.capacity_gate.max_memory_slots
            if slots is not None:
                lengths = encoder_mask.sum(dim=1, keepdim=True)
                combined_mask = combined_mask & (positions >= (lengths - int(slots)))

        # Hard local constraints should never produce an all-masked row; fall back to the padded mask.
        empty_rows = ~combined_mask.any(dim=1)
        if bool(empty_rows.any().item()):
            combined_mask = torch.where(empty_rows.unsqueeze(1), encoder_mask.bool(), combined_mask)

        scores = scores.masked_fill(~combined_mask, torch.finfo(scores.dtype).min)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = attention_weights.masked_fill(~combined_mask, 0.0)
        normalizer = attention_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(scores.dtype).eps)
        attention_weights = attention_weights / normalizer

        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        fused_hidden = torch.tanh(self.fusion(torch.cat([decoder_state, context], dim=-1)))
        entropy = -(attention_weights * torch.log(attention_weights.clamp_min(1e-8))).sum(dim=-1)
        peak_index = attention_weights.argmax(dim=-1)
        diagnostics = {
            "attention_entropy": entropy,
            "attention_peak_index": peak_index,
            "attention_peak_displacement": peak_index.float() - step.float(),
        }
        return fused_hidden, attention_weights, diagnostics
