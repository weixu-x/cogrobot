"""Model components for Corsi memory-recall V2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class CorsiMemoryRecallV2Config:
    image_channels: int = 3
    image_size: int = 128
    k_samples_per_segment: int = 12
    max_sequence_length: int = 9
    num_blocks: int = 9
    eos_token_id: int = 9
    ignore_index: int = -100
    cnn_dim: int = 64
    visual_hidden_dim: int = 64
    motor_hidden_dim: int = 64
    item_dim: int = 64
    memory_dim: int = 16
    memory_write_mode: str = "lstm"
    memory_slot_dim: int = 16
    recall_hidden_dim: int = 64
    recall_token_dim: int = 32
    joint_dim: int = 7
    ee_pose_dim: int = 7
    ee_xy_dim: int = 2
    memory_noise_std: float = 0.0

    @property
    def num_tokens(self) -> int:
        return int(max(self.num_blocks + 1, self.eos_token_id + 1))

    @property
    def max_recall_steps(self) -> int:
        return int(self.max_sequence_length + 1)


class VisualEncoder(nn.Module):
    """CNN encoder for RGB frames shaped as 3x128x128."""

    def __init__(self, output_dim: int = 64, image_channels: int = 3) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        output_groups = _group_count(self.output_dim)
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, 24, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(6, 24),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, output_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(output_groups, output_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(f"expected images [N,C,H,W], got shape={tuple(images.shape)}")
        return self.net(images).flatten(start_dim=1)


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(int(max_groups), int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


class InstrumentedLSTMCell(nn.Module):
    """Standard LSTM cell exposing gate and state activations."""

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
            input_t.matmul(self.weight_ih.t())
            + self.bias_ih
            + h_prev.matmul(self.weight_hh.t())
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


def _stack_trace(trace: dict[str, list[Tensor]], dim: int) -> dict[str, Tensor]:
    return {key: torch.stack(values, dim=dim) for key, values in trace.items()}


class CorsiMemoryRecallV2Model(nn.Module):
    """RGB-segment presentation, item memory, and autonomous block+EOS recall."""

    def __init__(self, config: CorsiMemoryRecallV2Config | None = None) -> None:
        super().__init__()
        self.config = config or CorsiMemoryRecallV2Config()
        cfg = self.config
        self.visual_encoder = VisualEncoder(cfg.cnn_dim, cfg.image_channels)
        self.visual_lstm = InstrumentedLSTMCell(cfg.cnn_dim, cfg.visual_hidden_dim)
        self.motor_lstm = InstrumentedLSTMCell(cfg.visual_hidden_dim, cfg.motor_hidden_dim)
        self.item_projection = nn.Sequential(
            nn.Linear(cfg.visual_hidden_dim + cfg.motor_hidden_dim, cfg.item_dim),
            nn.LayerNorm(cfg.item_dim),
            nn.ReLU(),
        )
        self.memory_lstm = InstrumentedLSTMCell(cfg.item_dim, cfg.memory_dim)
        self.memory_slot_item_projection = nn.Linear(cfg.item_dim, cfg.memory_slot_dim)
        self.memory_slot_position = nn.Parameter(torch.empty(cfg.max_sequence_length, cfg.memory_slot_dim))
        self.memory_slot_projection = nn.Sequential(
            nn.Linear(cfg.max_sequence_length * cfg.memory_slot_dim, cfg.memory_dim),
            nn.LayerNorm(cfg.memory_dim),
            nn.Tanh(),
        )
        self.memory_order_head = nn.Linear(
            2 * cfg.memory_dim,
            cfg.max_sequence_length * cfg.num_blocks,
        )
        self.final_memory_order_head = nn.Linear(
            cfg.memory_dim,
            cfg.max_sequence_length * cfg.num_blocks,
        )
        self.final_memory_length_head = nn.Linear(cfg.memory_dim, cfg.max_sequence_length)
        self.memory_to_recall_h = nn.Linear(cfg.memory_dim, cfg.recall_hidden_dim)
        self.memory_to_recall_c = nn.Linear(cfg.memory_dim, cfg.recall_hidden_dim)
        self.recall_token = nn.Parameter(torch.empty(cfg.recall_token_dim))
        self.recall_lstm = InstrumentedLSTMCell(cfg.recall_token_dim, cfg.recall_hidden_dim)
        self.block_head = nn.Linear(cfg.recall_hidden_dim, cfg.num_tokens)
        self.coord_head = nn.Linear(cfg.recall_hidden_dim, 2)
        self.joint_head = nn.Linear(cfg.motor_hidden_dim, cfg.joint_dim)
        self.ee_pose_head = nn.Linear(cfg.motor_hidden_dim, cfg.ee_pose_dim)
        self.ee_xy_head = nn.Linear(cfg.motor_hidden_dim, cfg.ee_xy_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.recall_token, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_slot_position, mean=0.0, std=0.02)

    def get_recall_inputs(self, batch_size: int, steps: int, *, device: torch.device | None = None) -> Tensor:
        token = self.recall_token
        if device is not None:
            token = token.to(device=device)
        return token.view(1, 1, -1).expand(int(batch_size), int(steps), -1)

    def _validate_presentation_inputs(self, images: Tensor, segment_mask: Tensor, frame_mask: Tensor) -> None:
        if images.ndim != 6:
            raise ValueError(f"expected images [B,L,K,C,H,W], got shape={tuple(images.shape)}")
        if segment_mask.shape != images.shape[:2]:
            raise ValueError(
                f"segment_mask shape {tuple(segment_mask.shape)} does not match images [B,L]={tuple(images.shape[:2])}"
            )
        if frame_mask.shape != images.shape[:3]:
            raise ValueError(
                f"frame_mask shape {tuple(frame_mask.shape)} does not match images [B,L,K]={tuple(images.shape[:3])}"
            )
        if images.shape[3] != self.config.image_channels:
            raise ValueError(f"expected {self.config.image_channels} image channels, got {images.shape[3]}")

    def encode_presentation(
        self,
        *,
        images: Tensor,
        segment_mask: Tensor,
        frame_mask: Tensor,
        return_traces: bool = False,
    ) -> dict[str, Any]:
        self._validate_presentation_inputs(images, segment_mask, frame_mask)
        batch_size, segments, frames, channels, height, width = images.shape
        device = images.device
        dtype = images.dtype
        flat_images = images.reshape(batch_size * segments * frames, channels, height, width)
        frame_features = self.visual_encoder(flat_images).reshape(batch_size, segments, frames, -1)

        valid_frame = (frame_mask & segment_mask.unsqueeze(-1)).to(dtype=dtype).unsqueeze(-1)
        visual_h = torch.zeros(
            batch_size, segments, frames, self.config.visual_hidden_dim, dtype=dtype, device=device
        )
        motor_h = torch.zeros(
            batch_size, segments, frames, self.config.motor_hidden_dim, dtype=dtype, device=device
        )
        item_visual = torch.zeros(batch_size, segments, self.config.visual_hidden_dim, dtype=dtype, device=device)
        item_motor = torch.zeros(batch_size, segments, self.config.motor_hidden_dim, dtype=dtype, device=device)

        for segment_index in range(segments):
            h_v = torch.zeros(batch_size, self.config.visual_hidden_dim, dtype=dtype, device=device)
            c_v = torch.zeros_like(h_v)
            h_m = torch.zeros(batch_size, self.config.motor_hidden_dim, dtype=dtype, device=device)
            c_m = torch.zeros_like(h_m)
            for frame_index in range(frames):
                mask = valid_frame[:, segment_index, frame_index]
                h_v_new, c_v_new, _ = self.visual_lstm(
                    frame_features[:, segment_index, frame_index], (h_v, c_v)
                )
                h_v = h_v_new * mask + h_v * (1.0 - mask)
                c_v = c_v_new * mask + c_v * (1.0 - mask)
                h_m_new, c_m_new, _ = self.motor_lstm(h_v, (h_m, c_m))
                h_m = h_m_new * mask + h_m * (1.0 - mask)
                c_m = c_m_new * mask + c_m * (1.0 - mask)
                visual_h[:, segment_index, frame_index] = h_v
                motor_h[:, segment_index, frame_index] = h_m
            segment_valid = segment_mask[:, segment_index].to(dtype=dtype).unsqueeze(-1)
            item_visual[:, segment_index] = h_v * segment_valid
            item_motor[:, segment_index] = h_m * segment_valid

        item_input = torch.cat([item_visual, item_motor], dim=-1)
        item_embeddings = self.item_projection(item_input.reshape(batch_size * segments, -1)).reshape(
            batch_size, segments, -1
        )
        item_embeddings = item_embeddings * segment_mask.to(dtype=dtype).unsqueeze(-1)

        pred_joint = self.joint_head(motor_h) * valid_frame
        pred_ee_pose = self.ee_pose_head(motor_h) * valid_frame
        pred_ee_xy = self.ee_xy_head(motor_h) * valid_frame

        result: dict[str, Any] = {
            "frame_features": frame_features,
            "item_visual": item_visual,
            "item_motor": item_motor,
            "item_embeddings": item_embeddings,
            "pred_joint": pred_joint,
            "pred_ee_pose": pred_ee_pose,
            "pred_ee_xy": pred_ee_xy,
        }
        if return_traces:
            result["traces"] = {
                "visual_h": visual_h,
                "motor_h": motor_h,
            }
        return result

    def run_memory(
        self,
        item_embeddings: Tensor,
        segment_mask: Tensor,
        *,
        return_traces: bool = False,
    ) -> dict[str, Any]:
        if self.config.memory_write_mode == "slot_compress":
            return self._run_slot_memory(
                item_embeddings,
                segment_mask,
                return_traces=return_traces,
            )
        if self.config.memory_write_mode != "lstm":
            raise ValueError(f"unknown memory_write_mode: {self.config.memory_write_mode}")
        batch_size, segments, _ = item_embeddings.shape
        dtype = item_embeddings.dtype
        device = item_embeddings.device
        h = torch.zeros(batch_size, self.config.memory_dim, dtype=dtype, device=device)
        c = torch.zeros_like(h)
        trace: dict[str, list[Tensor]] = {
            "h_t": [],
            "c_t": [],
            "input_gate": [],
            "forget_gate": [],
            "candidate": [],
            "output_gate": [],
        }
        memory_h: list[Tensor] = []
        memory_c: list[Tensor] = []
        for segment_index in range(segments):
            h_new, c_new, gates = self.memory_lstm(item_embeddings[:, segment_index], (h, c))
            mask = segment_mask[:, segment_index].to(dtype=dtype).unsqueeze(-1)
            h = h_new * mask + h * (1.0 - mask)
            c = c_new * mask + c * (1.0 - mask)
            memory_h.append(h)
            memory_c.append(c)
            if return_traces:
                trace["h_t"].append(h)
                trace["c_t"].append(c)
                for key, value in gates.items():
                    trace[key].append(value * mask)

        memory_before_noise = h
        final_memory = memory_before_noise
        if self.training and self.config.memory_noise_std > 0.0:
            final_memory = final_memory + torch.randn_like(final_memory) * float(self.config.memory_noise_std)

        result: dict[str, Any] = {
            "final_memory": final_memory,
            "memory_before_noise": memory_before_noise,
            "final_memory_order_logits": self.final_memory_order_head(memory_before_noise).reshape(
                batch_size,
                self.config.max_sequence_length,
                self.config.num_blocks,
            ),
            "final_memory_length_logits": self.final_memory_length_head(memory_before_noise),
        }
        if memory_h:
            memory_states = torch.cat(
                [torch.stack(memory_h, dim=1), torch.stack(memory_c, dim=1)],
                dim=-1,
            )
            result["memory_order_logits"] = self.memory_order_head(memory_states).reshape(
                batch_size,
                segments,
                self.config.max_sequence_length,
                self.config.num_blocks,
            )
        if return_traces:
            result["traces"] = _stack_trace(trace, dim=1)
        return result

    def _run_slot_memory(
        self,
        item_embeddings: Tensor,
        segment_mask: Tensor,
        *,
        return_traces: bool = False,
    ) -> dict[str, Any]:
        batch_size, segments, _ = item_embeddings.shape
        if int(segments) > int(self.config.max_sequence_length):
            raise ValueError(
                f"slot_compress memory supports at most {self.config.max_sequence_length} segments, got {segments}"
            )
        dtype = item_embeddings.dtype
        device = item_embeddings.device
        slots = [
            torch.zeros(batch_size, self.config.memory_slot_dim, dtype=dtype, device=device)
            for _ in range(self.config.max_sequence_length)
        ]
        zero_c = torch.zeros(batch_size, self.config.memory_dim, dtype=dtype, device=device)
        trace: dict[str, list[Tensor]] = {
            "h_t": [],
            "c_t": [],
            "input_gate": [],
            "forget_gate": [],
            "candidate": [],
            "output_gate": [],
        }
        memory_h: list[Tensor] = []
        memory_c: list[Tensor] = []
        zero_gate = torch.zeros(batch_size, self.config.memory_dim, dtype=dtype, device=device)
        for segment_index in range(segments):
            position = self.memory_slot_position[segment_index].to(device=device, dtype=dtype).unsqueeze(0)
            write = torch.tanh(self.memory_slot_item_projection(item_embeddings[:, segment_index]) + position)
            mask = segment_mask[:, segment_index].to(dtype=dtype).unsqueeze(-1)
            slots[segment_index] = write * mask + slots[segment_index] * (1.0 - mask)
            slot_tensor = torch.stack(slots, dim=1)
            h = self.memory_slot_projection(slot_tensor.reshape(batch_size, -1))
            c = zero_c
            memory_h.append(h)
            memory_c.append(c)
            if return_traces:
                trace["h_t"].append(h)
                trace["c_t"].append(c)
                for key in ("input_gate", "forget_gate", "candidate", "output_gate"):
                    trace[key].append(zero_gate)

        if memory_h:
            memory_before_noise = memory_h[-1]
            memory_states = torch.cat(
                [torch.stack(memory_h, dim=1), torch.stack(memory_c, dim=1)],
                dim=-1,
            )
        else:
            memory_before_noise = torch.zeros(batch_size, self.config.memory_dim, dtype=dtype, device=device)
            memory_states = torch.zeros(batch_size, 0, 2 * self.config.memory_dim, dtype=dtype, device=device)
        final_memory = memory_before_noise
        if self.training and self.config.memory_noise_std > 0.0:
            final_memory = final_memory + torch.randn_like(final_memory) * float(self.config.memory_noise_std)

        result: dict[str, Any] = {
            "final_memory": final_memory,
            "memory_before_noise": memory_before_noise,
            "final_memory_order_logits": self.final_memory_order_head(memory_before_noise).reshape(
                batch_size,
                self.config.max_sequence_length,
                self.config.num_blocks,
            ),
            "final_memory_length_logits": self.final_memory_length_head(memory_before_noise),
            "memory_order_logits": self.memory_order_head(memory_states).reshape(
                batch_size,
                segments,
                self.config.max_sequence_length,
                self.config.num_blocks,
            ),
        }
        if return_traces:
            result["traces"] = _stack_trace(trace, dim=1)
        return result

    def recall_from_memory(
        self,
        memory: Tensor,
        *,
        max_recall_steps: int | None = None,
        return_traces: bool = False,
    ) -> dict[str, Any]:
        steps = int(max_recall_steps or self.config.max_recall_steps)
        batch_size = int(memory.shape[0])
        h = torch.tanh(self.memory_to_recall_h(memory))
        c = torch.tanh(self.memory_to_recall_c(memory))
        recall_inputs = self.get_recall_inputs(batch_size, steps, device=memory.device).to(dtype=memory.dtype)
        logits = []
        coord = []
        trace: dict[str, list[Tensor]] = {
            "h_t": [],
            "c_t": [],
            "input_gate": [],
            "forget_gate": [],
            "candidate": [],
            "output_gate": [],
        }
        for step in range(steps):
            h, c, gates = self.recall_lstm(recall_inputs[:, step], (h, c))
            logits.append(self.block_head(h))
            coord.append(self.coord_head(h))
            if return_traces:
                trace["h_t"].append(h)
                trace["c_t"].append(c)
                for key, value in gates.items():
                    trace[key].append(value)

        result: dict[str, Any] = {
            "logits": torch.stack(logits, dim=1),
            "coord": torch.stack(coord, dim=1),
            "recall_inputs": recall_inputs,
        }
        if return_traces:
            result["traces"] = _stack_trace(trace, dim=1)
        return result

    def forward(
        self,
        *,
        images: Tensor,
        segment_mask: Tensor,
        frame_mask: Tensor,
        max_recall_steps: int | None = None,
        return_traces: bool = False,
    ) -> dict[str, Any]:
        presentation = self.encode_presentation(
            images=images,
            segment_mask=segment_mask,
            frame_mask=frame_mask,
            return_traces=return_traces,
        )
        memory = self.run_memory(
            presentation["item_embeddings"],
            segment_mask,
            return_traces=return_traces,
        )
        recall = self.recall_from_memory(
            memory["final_memory"],
            max_recall_steps=max_recall_steps,
            return_traces=return_traces,
        )
        result: dict[str, Any] = {
            "logits": recall["logits"],
            "coord": recall["coord"],
            "pred_joint": presentation["pred_joint"],
            "pred_ee_pose": presentation["pred_ee_pose"],
            "pred_ee_xy": presentation["pred_ee_xy"],
            "item_embeddings": presentation["item_embeddings"],
            "item_visual": presentation["item_visual"],
            "item_motor": presentation["item_motor"],
            "final_memory": memory["final_memory"],
            "memory_before_noise": memory["memory_before_noise"],
            "memory_order_logits": memory["memory_order_logits"],
            "final_memory_order_logits": memory["final_memory_order_logits"],
            "final_memory_length_logits": memory["final_memory_length_logits"],
            "recall_inputs": recall["recall_inputs"],
        }
        if return_traces:
            result["traces"] = {
                "presentation": presentation["traces"],
                "memory": memory["traces"],
                "recall": recall["traces"],
            }
        return result

    def decode(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        intervention: str | None = None,
        max_recall_steps: int | None = None,
        return_traces: bool = False,
    ) -> dict[str, Any]:
        """Recall hook used by evaluator causal sanity checks."""

        if intervention not in {None, "memory_zero", "memory_shuffle", "presentation_order_shuffle"}:
            raise ValueError(f"unknown recall intervention: {intervention}")
        presentation = self.encode_presentation(
            images=model_inputs["images"],
            segment_mask=model_inputs["segment_mask"],
            frame_mask=model_inputs["frame_mask"],
            return_traces=return_traces,
        )
        memory = self.run_memory(
            presentation["item_embeddings"],
            model_inputs["segment_mask"],
            return_traces=return_traces,
        )
        recall_memory = memory["final_memory"]
        if intervention == "memory_zero":
            recall_memory = torch.zeros_like(recall_memory)
        elif intervention == "memory_shuffle":
            if recall_memory.shape[0] > 1:
                recall_memory = torch.roll(recall_memory, shifts=1, dims=0)
            else:
                recall_memory = torch.zeros_like(recall_memory)
        recall = self.recall_from_memory(
            recall_memory,
            max_recall_steps=max_recall_steps,
            return_traces=return_traces,
        )
        result: dict[str, Any] = {
            "logits": recall["logits"],
            "coord": recall["coord"],
            "pred_joint": presentation["pred_joint"],
            "pred_ee_pose": presentation["pred_ee_pose"],
            "pred_ee_xy": presentation["pred_ee_xy"],
            "item_embeddings": presentation["item_embeddings"],
            "item_visual": presentation["item_visual"],
            "item_motor": presentation["item_motor"],
            "final_memory": memory["final_memory"],
            "memory_before_noise": memory["memory_before_noise"],
            "memory_order_logits": memory["memory_order_logits"],
            "final_memory_order_logits": memory["final_memory_order_logits"],
            "final_memory_length_logits": memory["final_memory_length_logits"],
            "recall_memory": recall_memory,
            "recall_inputs": recall["recall_inputs"],
        }
        if return_traces:
            result["traces"] = {
                "presentation": presentation["traces"],
                "memory": memory["traces"],
                "recall": recall["traces"],
            }
        return result


_CONFIG_ALIASES = {
    "D_mem": "memory_dim",
    "cnn_out": "cnn_dim",
    "visual_hidden": "visual_hidden_dim",
    "motor_hidden": "motor_hidden_dim",
    "recall_hidden": "recall_hidden_dim",
}


def _config_values_from_mapping(source: Mapping[str, Any] | None) -> dict[str, Any]:
    allowed = {field.name for field in fields(CorsiMemoryRecallV2Config)}
    values: dict[str, Any] = {}
    if not source:
        return values
    for key, value in source.items():
        canonical_key = _CONFIG_ALIASES.get(str(key), str(key))
        if canonical_key in allowed:
            values[canonical_key] = value
    if "joint_names" in source and "joint_dim" not in values:
        values["joint_dim"] = len(source["joint_names"])
    return values


def build_model(
    config: Mapping[str, Any] | None = None,
    *,
    stage: int | None = None,
    manifest: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> CorsiMemoryRecallV2Model:
    """Build the Lane C model while accepting Lane D train/eval metadata.

    ``stage`` is accepted as a no-op so the trainer can use a single factory
    signature for staged models. Only dataclass-backed model fields are copied
    from broader experiment configs/manifests.
    """

    del stage
    values = {
        **_config_values_from_mapping(manifest),
        **_config_values_from_mapping(config),
        **_config_values_from_mapping(kwargs),
    }
    return CorsiMemoryRecallV2Model(CorsiMemoryRecallV2Config(**values))


def parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))
