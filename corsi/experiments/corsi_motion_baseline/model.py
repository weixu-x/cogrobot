"""Exact 7-joint Corsi motion baseline models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class MotionModelConfig:
    joint_dim: int = 7
    hidden_dim: int = 128
    visual_feature_dim: int = 128
    joint_feature_dim: int = 32
    fused_dim: int = 128
    use_vision: bool = True


class VisualEncoder(nn.Module):
    """Required random-init 128-d visual encoder for RGB 3x128x128 frames."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.net(images).flatten(start_dim=1)


class JointEncoder(nn.Module):
    def __init__(self, joint_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

    def forward(self, joints: Tensor) -> Tensor:
        return self.net(joints)


class InstrumentedLSTMCell(nn.Module):
    """Standard LSTM cell exposing h, c, and gate activations."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, input_t: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        h_prev, c_prev = state
        gates = (
            torch.matmul(input_t, self.weight_ih.t())
            + self.bias_ih
            + torch.matmul(h_prev, self.weight_hh.t())
            + self.bias_hh
        )
        i_raw, f_raw, g_raw, o_raw = gates.chunk(4, dim=-1)
        input_gate = torch.sigmoid(i_raw)
        forget_gate = torch.sigmoid(f_raw)
        candidate = torch.tanh(g_raw)
        output_gate = torch.sigmoid(o_raw)
        c_t = forget_gate * c_prev + input_gate * candidate
        h_t = output_gate * torch.tanh(c_t)
        return h_t, c_t, {
            "input_gate": input_gate,
            "forget_gate": forget_gate,
            "candidate": candidate,
            "output_gate": output_gate,
        }


class CorsiMotionPredictor(nn.Module):
    """Causal teacher-forced q_hat_{t+1}=f(I_0..I_t, q_0..q_t)."""

    def __init__(self, config: MotionModelConfig) -> None:
        super().__init__()
        self.config = config
        self.visual_encoder = VisualEncoder() if config.use_vision else None
        self.joint_encoder = JointEncoder(config.joint_dim)
        if config.use_vision:
            self.fusion = nn.Sequential(
                nn.Linear(config.visual_feature_dim + config.joint_feature_dim, config.fused_dim),
                nn.LayerNorm(config.fused_dim),
                nn.ReLU(),
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(config.joint_feature_dim, config.fused_dim),
                nn.LayerNorm(config.fused_dim),
                nn.ReLU(),
            )
        self.recurrent = InstrumentedLSTMCell(config.fused_dim, config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.joint_dim)

    def _encode_visual(self, images: Tensor, *, zero_vision: bool = False) -> Tensor:
        batch_size, steps, channels, height, width = images.shape
        if self.visual_encoder is None:
            raise RuntimeError("visual encoder is not enabled")
        if zero_vision:
            return torch.zeros(
                batch_size,
                steps,
                self.config.visual_feature_dim,
                dtype=images.dtype,
                device=images.device,
            )
        flat = images.reshape(batch_size * steps, channels, height, width)
        return self.visual_encoder(flat).reshape(batch_size, steps, -1)

    def forward(
        self,
        *,
        images: Tensor | None,
        joints: Tensor,
        valid_mask: Tensor,
        return_traces: bool = False,
        zero_vision: bool = False,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        batch_size, steps, joint_dim = joints.shape
        if joint_dim != self.config.joint_dim:
            raise ValueError(f"expected joint_dim={self.config.joint_dim}, got {joint_dim}")
        joint_features = self.joint_encoder(joints.reshape(batch_size * steps, joint_dim)).reshape(
            batch_size, steps, -1
        )
        if self.config.use_vision:
            if images is None:
                raise ValueError("images are required when use_vision=True")
            visual_features = self._encode_visual(images, zero_vision=zero_vision)
            fused_input = torch.cat([visual_features, joint_features], dim=-1)
        else:
            visual_features = torch.zeros(
                batch_size,
                steps,
                self.config.visual_feature_dim,
                dtype=joints.dtype,
                device=joints.device,
            )
            fused_input = joint_features
        fused = self.fusion(fused_input.reshape(batch_size * steps, -1)).reshape(batch_size, steps, -1)

        h = torch.zeros(batch_size, self.config.hidden_dim, dtype=joints.dtype, device=joints.device)
        c = torch.zeros_like(h)
        outputs = []
        traces: dict[str, list[Tensor]] = {
            "h_t": [],
            "c_t": [],
            "input_gate": [],
            "forget_gate": [],
            "candidate": [],
            "output_gate": [],
        }
        for step in range(steps):
            h_new, c_new, gates = self.recurrent(fused[:, step], (h, c))
            mask = valid_mask[:, step].to(dtype=joints.dtype).unsqueeze(-1)
            h = h_new * mask + h * (1.0 - mask)
            c = c_new * mask + c * (1.0 - mask)
            pred = self.output(h) * mask
            outputs.append(pred)
            if return_traces:
                traces["h_t"].append(h)
                traces["c_t"].append(c)
                for key, value in gates.items():
                    traces[key].append(value * mask)
        result: dict[str, Tensor | dict[str, Tensor]] = {
            "pred_joints_next": torch.stack(outputs, dim=1),
            "visual_feature": visual_features,
            "joint_feature": joint_features,
            "fused_feature": fused,
        }
        if return_traces:
            result["traces"] = {key: torch.stack(value, dim=1) for key, value in traces.items()}
        return result


def build_model(model_type: str, *, joint_dim: int = 7, hidden_dim: int = 128) -> CorsiMotionPredictor:
    if model_type == "visual_joint":
        return CorsiMotionPredictor(MotionModelConfig(joint_dim=joint_dim, hidden_dim=hidden_dim, use_vision=True))
    if model_type == "joint_only":
        return CorsiMotionPredictor(MotionModelConfig(joint_dim=joint_dim, hidden_dim=hidden_dim, use_vision=False))
    raise ValueError("model_type must be 'visual_joint' or 'joint_only'")


def persistence_prediction(joints: Tensor) -> Tensor:
    return joints.clone()


def parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))
