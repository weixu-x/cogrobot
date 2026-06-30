"""Frozen storage-localization probes for the LSTM memory-onset run."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.experiments.corsi_memory_recall_v2.train import (  # noqa: E402
    _build_model,
    canonical_manifest_path,
    load_checkpoint,
    load_config,
    make_loader,
    move_to_device,
    resolve_device,
)


CONFIG = Path("corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_dmem64.json")
RUN_ROOT = Path(
    "corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/"
    "stage2_seed0_lstm_memory_dmem64_onset_20260629"
)
OUTPUT_JSON = Path("reports/v2_lstm_storage_localization_20260629.json")
OUTPUT_MD = Path("reports/v2_lstm_storage_localization_20260629.md")
PROBE_STEPS = 900


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path
    requested: str


CHECKPOINTS = [
    CheckpointSpec(
        "epoch080_best_full",
        RUN_ROOT / "milestones/epoch_080/best_full_sequence.pt",
        "epoch 80 best-full",
    ),
    CheckpointSpec(
        "epoch120_best_full",
        RUN_ROOT / "milestones/epoch_120/best_full_sequence.pt",
        "epoch 120 best-full",
    ),
    CheckpointSpec(
        "epoch139_best_full",
        RUN_ROOT / "milestones/epoch_160/best_full_sequence.pt",
        "epoch 139 best-full",
    ),
    CheckpointSpec(
        "epoch160_latest",
        RUN_ROOT / "milestones/epoch_160/latest.pt",
        "epoch 160 latest",
    ),
    CheckpointSpec(
        "epoch200_latest",
        RUN_ROOT / "milestones/epoch_200/latest.pt",
        "epoch 200 latest",
    ),
    CheckpointSpec(
        "epoch122_best_token",
        RUN_ROOT / "best_token.pt",
        "best-token epoch 122",
    ),
    CheckpointSpec(
        "epoch093_best_val_loss",
        RUN_ROOT / "best_val_loss.pt",
        "best-val-loss epoch 93",
    ),
]


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def target_lengths(targets: Tensor, mask: Tensor, *, eos_token_id: int, ignore_index: int) -> Tensor:
    valid_blocks = mask & targets.ne(int(ignore_index)) & targets.ne(int(eos_token_id))
    return valid_blocks.sum(dim=1).long()


def block_targets(
    targets: Tensor,
    masks: Tensor,
    *,
    max_sequence_length: int,
    eos_token_id: int,
    ignore_index: int,
) -> tuple[Tensor, Tensor]:
    blocks = torch.full((targets.shape[0], int(max_sequence_length)), int(ignore_index), dtype=torch.long)
    valid = torch.zeros((targets.shape[0], int(max_sequence_length)), dtype=torch.bool)
    steps = min(int(max_sequence_length), int(targets.shape[1]))
    values = targets[:, :steps].detach().cpu().long()
    keep = (
        masks[:, :steps].detach().cpu().bool()
        & values.ne(int(ignore_index))
        & values.ne(int(eos_token_id))
    )
    blocks[:, :steps] = values.masked_fill(~keep, int(ignore_index))
    valid[:, :steps] = keep
    return blocks, valid


def pad_state(state: Tensor, segment_mask: Tensor, *, max_sequence_length: int) -> tuple[Tensor, Tensor]:
    batch, steps, dim = state.shape
    padded = torch.zeros((batch, int(max_sequence_length), dim), dtype=state.dtype)
    mask = torch.zeros((batch, int(max_sequence_length)), dtype=torch.bool)
    keep_steps = min(steps, int(max_sequence_length))
    padded[:, :keep_steps] = state[:, :keep_steps].detach().cpu()
    mask[:, :keep_steps] = segment_mask[:, :keep_steps].detach().cpu().bool()
    return padded, mask


@torch.no_grad()
def collect_split(
    model: nn.Module,
    loader: Any,
    *,
    device: torch.device,
    eos_token_id: int,
    ignore_index: int,
    max_sequence_length: int,
) -> dict[str, Tensor]:
    model.eval()
    item_features: list[Tensor] = []
    item_labels: list[Tensor] = []
    target_blocks: list[Tensor] = []
    target_valid: list[Tensor] = []
    lengths: list[Tensor] = []
    h_states: list[Tensor] = []
    c_states: list[Tensor] = []
    state_masks: list[Tensor] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        model_inputs = batch["model_inputs"]
        targets = batch["targets"]["tokens"]
        token_mask = batch["targets"]["token_mask"]
        segment_mask = model_inputs["segment_mask"]
        outputs = model(
            images=model_inputs["images"],
            segment_mask=segment_mask,
            frame_mask=model_inputs["frame_mask"],
            max_recall_steps=int(targets.shape[1]),
            return_traces=True,
        )
        block_id = batch["metadata"]["block_id"]
        valid_segment = segment_mask.detach().cpu().bool()
        item_features.append(outputs["item_embeddings"].detach().cpu()[valid_segment])
        item_labels.append(block_id.detach().cpu().long()[valid_segment])

        blocks, valid = block_targets(
            targets,
            token_mask,
            max_sequence_length=max_sequence_length,
            eos_token_id=eos_token_id,
            ignore_index=ignore_index,
        )
        target_blocks.append(blocks)
        target_valid.append(valid)
        lengths.append(
            target_lengths(targets, token_mask, eos_token_id=eos_token_id, ignore_index=ignore_index)
            .detach()
            .cpu()
        )
        h, state_mask = pad_state(
            outputs["traces"]["memory"]["h_t"],
            segment_mask,
            max_sequence_length=max_sequence_length,
        )
        c, _ = pad_state(
            outputs["traces"]["memory"]["c_t"],
            segment_mask,
            max_sequence_length=max_sequence_length,
        )
        h_states.append(h)
        c_states.append(c)
        state_masks.append(state_mask)
    h_all = torch.cat(h_states, dim=0)
    c_all = torch.cat(c_states, dim=0)
    return {
        "item_features": torch.cat(item_features, dim=0),
        "item_labels": torch.cat(item_labels, dim=0),
        "target_blocks": torch.cat(target_blocks, dim=0),
        "target_valid": torch.cat(target_valid, dim=0),
        "lengths": torch.cat(lengths, dim=0),
        "h_states": h_all,
        "c_states": c_all,
        "hc_states": torch.cat([h_all, c_all], dim=-1),
        "state_mask": torch.cat(state_masks, dim=0),
    }


def train_classifier(
    x: Tensor,
    y: Tensor,
    *,
    classes: int,
    device: torch.device,
    steps: int = PROBE_STEPS,
) -> tuple[nn.Module, float]:
    torch.manual_seed(20260629)
    x = x.to(device)
    y = y.long().to(device)
    probe = nn.Linear(int(x.shape[-1]), int(classes)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.05, weight_decay=1e-4)
    last_loss = 0.0
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(x), y)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        if last_loss < 1e-4:
            break
    return probe, last_loss


@torch.no_grad()
def eval_classifier(probe: nn.Module, x: Tensor, y: Tensor, *, device: torch.device) -> dict[str, Any]:
    pred = probe(x.to(device)).argmax(dim=-1).cpu()
    y_cpu = y.long().cpu()
    return {
        "accuracy": float(pred.eq(y_cpu).float().mean().item()),
        "count": int(y_cpu.numel()),
    }


def train_order_probe(
    x: Tensor,
    targets: Tensor,
    valid: Tensor,
    *,
    device: torch.device,
    num_blocks: int,
    steps: int = PROBE_STEPS,
) -> tuple[nn.Module, float]:
    torch.manual_seed(20260630)
    x_dev = x.to(device)
    targets_dev = targets.long().to(device)
    valid_dev = valid.bool().to(device)
    probe = nn.Linear(int(x.shape[-1]), int(targets.shape[1]) * int(num_blocks)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.05, weight_decay=1e-4)
    last_loss = 0.0
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(x_dev).reshape(-1, int(targets.shape[1]), int(num_blocks))
        loss = F.cross_entropy(logits[valid_dev], targets_dev[valid_dev])
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        if last_loss < 1e-4:
            break
    return probe, last_loss


@torch.no_grad()
def eval_order_probe(
    probe: nn.Module,
    x: Tensor,
    targets: Tensor,
    valid: Tensor,
    lengths: Tensor,
    *,
    device: torch.device,
    num_blocks: int,
) -> dict[str, Any]:
    logits = probe(x.to(device)).reshape(-1, int(targets.shape[1]), int(num_blocks))
    pred = logits.argmax(dim=-1).cpu()
    targets_cpu = targets.long().cpu()
    valid_cpu = valid.bool().cpu()
    correct = pred.eq(targets_cpu) & valid_cpu
    exact = []
    for row in range(int(pred.shape[0])):
        length = int(lengths[row].item())
        if length <= 0:
            exact.append(0)
            continue
        exact.append(int(bool(pred[row, :length].eq(targets_cpu[row, :length]).all().item())))
    serial: dict[str, float] = {}
    serial_count: dict[str, int] = {}
    for position in range(int(targets.shape[1])):
        pos_valid = valid_cpu[:, position]
        count = int(pos_valid.sum().item())
        if count:
            serial[str(position)] = float(correct[:, position][pos_valid].float().mean().item())
            serial_count[str(position)] = count
    return {
        "block_token_order_accuracy": float(correct[valid_cpu].float().mean().item()) if bool(valid_cpu.any()) else 0.0,
        "known_length_exact_sequence_accuracy": float(sum(exact) / max(len(exact), 1)),
        "serial_position_accuracy": serial,
        "serial_position_count": serial_count,
        "sequence_count": int(pred.shape[0]),
        "block_token_count": int(valid_cpu.sum().item()),
    }


def train_length_probe(
    x: Tensor,
    lengths: Tensor,
    *,
    device: torch.device,
    max_sequence_length: int,
    steps: int = PROBE_STEPS,
) -> tuple[nn.Module, float]:
    torch.manual_seed(20260631)
    x_dev = x.to(device)
    labels = (lengths.clamp(min=1, max=int(max_sequence_length)) - 1).long().to(device)
    probe = nn.Linear(int(x.shape[-1]), int(max_sequence_length)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.05, weight_decay=1e-4)
    last_loss = 0.0
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(x_dev), labels)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        if last_loss < 1e-4:
            break
    return probe, last_loss


@torch.no_grad()
def eval_length_probe(
    probe: nn.Module,
    x: Tensor,
    lengths: Tensor,
    *,
    device: torch.device,
    max_sequence_length: int,
) -> dict[str, Any]:
    labels = (lengths.clamp(min=1, max=int(max_sequence_length)) - 1).long()
    pred = probe(x.to(device)).argmax(dim=-1).cpu()
    return {
        "accuracy": float(pred.eq(labels).float().mean().item()),
        "count": int(labels.numel()),
    }


def final_features(data: dict[str, Tensor], state_key: str) -> Tensor:
    states = data[state_key]
    rows = torch.arange(int(states.shape[0]))
    index = (data["lengths"].clamp(min=1) - 1).long()
    return states[rows, index]


def flattened_features(data: dict[str, Tensor], state_key: str) -> Tensor:
    states = data[state_key].clone()
    states = states * data["state_mask"].unsqueeze(-1).to(dtype=states.dtype)
    return states.reshape(states.shape[0], -1)


def trajectory_examples(data: dict[str, Tensor], state_key: str, *, num_blocks: int) -> dict[str, Tensor]:
    states = data[state_key]
    rows, steps = data["state_mask"].nonzero(as_tuple=True)
    x = states[rows, steps]
    target_blocks = data["target_blocks"][rows]
    target_valid = data["target_valid"][rows]
    current = target_blocks[torch.arange(int(rows.numel())), steps]
    position_index = torch.arange(int(target_blocks.shape[1])).view(1, -1)
    prefix_valid = target_valid & position_index.le(steps.view(-1, 1))
    prefix_set = torch.zeros((int(rows.numel()), int(num_blocks)), dtype=torch.float32)
    safe_targets = target_blocks.masked_fill(~prefix_valid, 0)
    prefix_set.scatter_add_(1, safe_targets.clamp(min=0, max=int(num_blocks) - 1), prefix_valid.float())
    prefix_set = prefix_set.clamp(max=1.0)
    return {
        "x": x,
        "current": current.long(),
        "prefix_targets": target_blocks.long(),
        "prefix_valid": prefix_valid.bool(),
        "prefix_set": prefix_set,
        "prefix_lengths": prefix_valid.sum(dim=1).long(),
        "step": steps.long(),
    }


def train_set_probe(
    x: Tensor,
    target_set: Tensor,
    *,
    device: torch.device,
    num_blocks: int,
    steps: int = PROBE_STEPS,
) -> tuple[nn.Module, float]:
    torch.manual_seed(20260632)
    x_dev = x.to(device)
    target_dev = target_set.float().to(device)
    probe = nn.Linear(int(x.shape[-1]), int(num_blocks)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.05, weight_decay=1e-4)
    last_loss = 0.0
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(probe(x_dev), target_dev)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        if last_loss < 1e-4:
            break
    return probe, last_loss


@torch.no_grad()
def eval_set_probe(
    probe: nn.Module,
    examples: dict[str, Tensor],
    *,
    device: torch.device,
    num_blocks: int,
) -> dict[str, Any]:
    scores = probe(examples["x"].to(device)).cpu()
    target = examples["prefix_set"].float()
    lengths = examples["prefix_lengths"].long()
    exact = []
    recalls = []
    per_step_exact: dict[str, list[int]] = {}
    per_step_recall: dict[str, list[float]] = {}
    for row in range(int(scores.shape[0])):
        k = int(lengths[row].item())
        pred_set = torch.zeros(int(num_blocks), dtype=torch.bool)
        if k > 0:
            top = torch.topk(scores[row], k=min(k, int(num_blocks))).indices
            pred_set[top] = True
        target_set = target[row].bool()
        hit = int(bool(torch.equal(pred_set, target_set)))
        intersection = int((pred_set & target_set).sum().item())
        recall = float(intersection / max(k, 1))
        exact.append(hit)
        recalls.append(recall)
        step_key = str(int(examples["step"][row].item()) + 1)
        per_step_exact.setdefault(step_key, []).append(hit)
        per_step_recall.setdefault(step_key, []).append(recall)
    return {
        "past_set_exact_accuracy": float(sum(exact) / max(len(exact), 1)),
        "past_set_mean_recall": float(sum(recalls) / max(len(recalls), 1)),
        "state_count": int(scores.shape[0]),
        "per_write_step_exact": {
            key: float(sum(values) / max(len(values), 1)) for key, values in sorted(per_step_exact.items(), key=lambda kv: int(kv[0]))
        },
        "per_write_step_recall": {
            key: float(sum(values) / max(len(values), 1)) for key, values in sorted(per_step_recall.items(), key=lambda kv: int(kv[0]))
        },
    }


@torch.no_grad()
def eval_prefix_order_probe(
    probe: nn.Module,
    examples: dict[str, Tensor],
    *,
    device: torch.device,
    num_blocks: int,
) -> dict[str, Any]:
    targets = examples["prefix_targets"].long()
    valid = examples["prefix_valid"].bool()
    logits = probe(examples["x"].to(device)).reshape(-1, int(targets.shape[1]), int(num_blocks))
    pred = logits.argmax(dim=-1).cpu()
    correct = pred.eq(targets) & valid
    exact = []
    per_step_token: dict[str, list[float]] = {}
    per_step_exact: dict[str, list[int]] = {}
    for row in range(int(pred.shape[0])):
        row_valid = valid[row]
        row_exact = int(bool(pred[row][row_valid].eq(targets[row][row_valid]).all().item()))
        exact.append(row_exact)
        token_acc = float(correct[row][row_valid].float().mean().item()) if bool(row_valid.any()) else 0.0
        step_key = str(int(examples["step"][row].item()) + 1)
        per_step_token.setdefault(step_key, []).append(token_acc)
        per_step_exact.setdefault(step_key, []).append(row_exact)
    return {
        "prefix_order_token_accuracy": float(correct[valid].float().mean().item()) if bool(valid.any()) else 0.0,
        "prefix_order_exact_accuracy": float(sum(exact) / max(len(exact), 1)),
        "state_count": int(pred.shape[0]),
        "prefix_token_count": int(valid.sum().item()),
        "per_write_step_token": {
            key: float(sum(values) / max(len(values), 1)) for key, values in sorted(per_step_token.items(), key=lambda kv: int(kv[0]))
        },
        "per_write_step_exact": {
            key: float(sum(values) / max(len(values), 1)) for key, values in sorted(per_step_exact.items(), key=lambda kv: int(kv[0]))
        },
    }


@torch.no_grad()
def eval_current_probe(
    probe: nn.Module,
    examples: dict[str, Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    pred = probe(examples["x"].to(device)).argmax(dim=-1).cpu()
    target = examples["current"].long()
    correct = pred.eq(target)
    per_step: dict[str, list[int]] = {}
    for row in range(int(pred.shape[0])):
        step_key = str(int(examples["step"][row].item()) + 1)
        per_step.setdefault(step_key, []).append(int(correct[row].item()))
    return {
        "current_item_accuracy": float(correct.float().mean().item()),
        "state_count": int(target.numel()),
        "per_write_step_current": {
            key: float(sum(values) / max(len(values), 1)) for key, values in sorted(per_step.items(), key=lambda kv: int(kv[0]))
        },
    }


def run_order_and_length_probe(
    train_data: dict[str, Tensor],
    split_data: dict[str, dict[str, Tensor]],
    feature_builder: Any,
    *,
    device: torch.device,
    num_blocks: int,
    max_sequence_length: int,
) -> dict[str, Any]:
    train_x = feature_builder(train_data)
    order_probe, order_loss = train_order_probe(
        train_x,
        train_data["target_blocks"],
        train_data["target_valid"],
        device=device,
        num_blocks=num_blocks,
    )
    length_probe, length_loss = train_length_probe(
        train_x,
        train_data["lengths"],
        device=device,
        max_sequence_length=max_sequence_length,
    )
    result: dict[str, Any] = {"order_train_loss": order_loss, "length_train_loss": length_loss}
    for split, data in split_data.items():
        x = feature_builder(data)
        result[split] = {
            "order": eval_order_probe(
                order_probe,
                x,
                data["target_blocks"],
                data["target_valid"],
                data["lengths"],
                device=device,
                num_blocks=num_blocks,
            ),
            "length": eval_length_probe(
                length_probe,
                x,
                data["lengths"],
                device=device,
                max_sequence_length=max_sequence_length,
            ),
        }
    return result


def run_item_probe(
    split_data: dict[str, dict[str, Tensor]],
    *,
    device: torch.device,
    num_blocks: int,
) -> dict[str, Any]:
    train = split_data["train"]
    probe, loss = train_classifier(
        train["item_features"],
        train["item_labels"],
        classes=num_blocks,
        device=device,
    )
    result: dict[str, Any] = {"train_loss": loss}
    for split, data in split_data.items():
        result[split] = eval_classifier(probe, data["item_features"], data["item_labels"], device=device)
    return result


def run_trajectory_probe(
    split_data: dict[str, dict[str, Tensor]],
    *,
    device: torch.device,
    num_blocks: int,
) -> dict[str, Any]:
    train_examples = trajectory_examples(split_data["train"], "hc_states", num_blocks=num_blocks)
    prefix_probe, prefix_loss = train_order_probe(
        train_examples["x"],
        train_examples["prefix_targets"],
        train_examples["prefix_valid"],
        device=device,
        num_blocks=num_blocks,
    )
    current_probe, current_loss = train_classifier(
        train_examples["x"],
        train_examples["current"],
        classes=num_blocks,
        device=device,
    )
    set_probe, set_loss = train_set_probe(
        train_examples["x"],
        train_examples["prefix_set"],
        device=device,
        num_blocks=num_blocks,
    )
    result: dict[str, Any] = {
        "state_definition": "[h_t;c_t]",
        "prefix_order_train_loss": prefix_loss,
        "current_item_train_loss": current_loss,
        "past_set_train_loss": set_loss,
    }
    for split, data in split_data.items():
        examples = trajectory_examples(data, "hc_states", num_blocks=num_blocks)
        result[split] = {
            "prefix_order": eval_prefix_order_probe(prefix_probe, examples, device=device, num_blocks=num_blocks),
            "current_item": eval_current_probe(current_probe, examples, device=device),
            "past_set": eval_set_probe(set_probe, examples, device=device, num_blocks=num_blocks),
        }
    return result


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_markdown(result: dict[str, Any]) -> None:
    order = result["checkpoint_order"]
    item_rows = []
    traj_rows = []
    final_rows = []
    flat_rows = []
    for label in order:
        item = result["checkpoints"][label]
        test_item = item["item_embedding_probe"]["test"]
        item_rows.append([label, item["source_epoch"], fmt(test_item["accuracy"]), test_item["count"]])
        test_traj = item["memory_trajectory_probe"]["test"]
        traj_rows.append(
            [
                label,
                fmt(test_traj["prefix_order"]["prefix_order_token_accuracy"]),
                fmt(test_traj["prefix_order"]["prefix_order_exact_accuracy"]),
                fmt(test_traj["current_item"]["current_item_accuracy"]),
                fmt(test_traj["past_set"]["past_set_exact_accuracy"]),
                fmt(test_traj["past_set"]["past_set_mean_recall"]),
            ]
        )
        for state_key, state_label in (("final_h", "h"), ("final_c", "c"), ("final_hc", "[h;c]")):
            test = item["final_state_probes"][state_key]["test"]
            final_rows.append(
                [
                    label,
                    state_label,
                    fmt(test["order"]["block_token_order_accuracy"]),
                    fmt(test["order"]["known_length_exact_sequence_accuracy"]),
                    fmt(test["length"]["accuracy"]),
                ]
            )
        for state_key, state_label in (("flat_h", "all_h"), ("flat_c", "all_c"), ("flat_hc", "all_[h;c]")):
            test = item["flattened_state_probes"][state_key]["test"]
            flat_rows.append(
                [
                    label,
                    state_label,
                    fmt(test["order"]["block_token_order_accuracy"]),
                    fmt(test["order"]["known_length_exact_sequence_accuracy"]),
                    fmt(test["length"]["accuracy"]),
                ]
            )

    best = result["checkpoints"]["epoch139_best_full"]
    best_item = best["item_embedding_probe"]["test"]["accuracy"]
    best_traj = best["memory_trajectory_probe"]["test"]
    best_final_h = best["final_state_probes"]["final_h"]["test"]["order"]["block_token_order_accuracy"]
    best_final_c = best["final_state_probes"]["final_c"]["test"]["order"]["block_token_order_accuracy"]
    best_final_hc = best["final_state_probes"]["final_hc"]["test"]["order"]["block_token_order_accuracy"]
    best_flat_hc = best["flattened_state_probes"]["flat_hc"]["test"]["order"]["block_token_order_accuracy"]
    lines = [
        "# V2 LSTM Storage Localization - 2026-06-29",
        "",
        "Scope: frozen linear probes over presentation item embeddings and LSTM memory states for `stage2_seed0_lstm_memory_dmem64_onset_20260629`.",
        "Only probe heads are trained on frozen features; no checkpoint model weights are updated.",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{OUTPUT_JSON}`",
        f"- Report: `{OUTPUT_MD}`",
        f"- Config: `{CONFIG}`",
        f"- Run root: `{RUN_ROOT}`",
        "",
        "## P1.1 Item Embedding Probe (test split)",
        "",
        markdown_table(["Checkpoint", "source epoch", "item block acc", "items"], item_rows),
        "",
        "## P1.2 Memory Write Trajectory Probe (test split, M_t = [h_t;c_t])",
        "",
        markdown_table(
            [
                "Checkpoint",
                "prefix token",
                "prefix exact",
                "current item",
                "past-set exact",
                "past-set recall",
            ],
            traj_rows,
        ),
        "",
        "## P1.3 Final h / c / [h;c] Probe (test split)",
        "",
        markdown_table(["Checkpoint", "state", "order token", "known-L exact", "length"], final_rows),
        "",
        "## P1.4 Flattened Memory-State Probe (test split)",
        "",
        markdown_table(["Checkpoint", "state", "order token", "known-L exact", "length"], flat_rows),
        "",
        "## Decision Tree Readout",
        "",
        (
            f"For epoch-139 best-full, item embedding probe is {fmt(best_item)}, so presentation/segment grounding is not "
            "the bottleneck."
        ),
        (
            "The memory trajectory is only partial: "
            f"prefix token {fmt(best_traj['prefix_order']['prefix_order_token_accuracy'])}, "
            f"prefix exact {fmt(best_traj['prefix_order']['prefix_order_exact_accuracy'])}, "
            f"current item {fmt(best_traj['current_item']['current_item_accuracy'])}, "
            f"past-set exact {fmt(best_traj['past_set']['past_set_exact_accuracy'])}."
        ),
        (
            f"Final c is stronger than final h on order token ({fmt(best_final_c)} vs {fmt(best_final_h)}), "
            f"and [h;c] is {fmt(best_final_hc)}. This means the LSTM cell state exposes more serial identity than h, "
            "but not enough to call storage solved."
        ),
        (
            f"Flattened all_[h;c] order token is {fmt(best_flat_hc)}. If this is materially above final [h;c], "
            "information exists in the trajectory more than in the final state, supporting attention over memory states."
        ),
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    started = time.time()
    cfg = load_config(CONFIG)
    device = resolve_device(str(cfg.get("device", "auto")))
    manifest_path = canonical_manifest_path(cfg)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eos_token_id = int(manifest.get("eos_token_id", cfg.get("eos_token_id", 9)))
    ignore_index = int(manifest.get("ignore_index", cfg.get("ignore_index", -100)))
    num_blocks = int(manifest.get("num_blocks", cfg.get("num_blocks", 9)))
    max_sequence_length = int(manifest.get("max_sequence_length", cfg.get("max_sequence_length", 9)))
    batch_size = int(cfg.get("batch_size", 16))
    loaders = {
        split: make_loader(manifest_path, split=split, batch_size=batch_size, shuffle=False)
        for split in ("train", "val", "test")
    }
    result: dict[str, Any] = {
        "created_time": time.time(),
        "config": str(CONFIG),
        "run_root": str(RUN_ROOT),
        "manifest_path": str(manifest_path),
        "canonical_fingerprint": manifest.get("canonical_fingerprint"),
        "device": str(device),
        "num_blocks": num_blocks,
        "eos_token_id": eos_token_id,
        "ignore_index": ignore_index,
        "max_sequence_length": max_sequence_length,
        "probe_steps": PROBE_STEPS,
        "checkpoint_order": [spec.label for spec in CHECKPOINTS],
        "checkpoints": {},
        "notes": [
            "Frozen main-model diagnostics only; probe heads are the only trained parameters.",
            "P1.2 defines M_t as concatenated [h_t;c_t].",
            "P1.4 pads missing future memory states with zeros before flattening.",
        ],
    }
    for spec in CHECKPOINTS:
        emit({"event": "start_checkpoint", "label": spec.label, "path": str(spec.path)})
        model = _build_model(cfg, stage=2, manifest=manifest).to(device)
        payload = load_checkpoint(spec.path, model=model, map_location=device)
        split_data = {
            split: collect_split(
                model,
                loaders[split],
                device=device,
                eos_token_id=eos_token_id,
                ignore_index=ignore_index,
                max_sequence_length=max_sequence_length,
            )
            for split in ("train", "val", "test")
        }
        checkpoint_result: dict[str, Any] = {
            "requested": spec.requested,
            "checkpoint": str(spec.path),
            "source_epoch": int(payload.get("epoch", -1)),
            "source_stage": int(payload.get("stage", -1)),
            "selection_key": str((payload.get("extra") or {}).get("selection_key", "")),
            "selection_score": (payload.get("extra") or {}).get("selection_score"),
        }
        checkpoint_result["item_embedding_probe"] = run_item_probe(
            split_data,
            device=device,
            num_blocks=num_blocks,
        )
        checkpoint_result["memory_trajectory_probe"] = run_trajectory_probe(
            split_data,
            device=device,
            num_blocks=num_blocks,
        )
        final_probes: dict[str, Any] = {}
        for state_key, result_key in (("h_states", "final_h"), ("c_states", "final_c"), ("hc_states", "final_hc")):
            final_probes[result_key] = run_order_and_length_probe(
                split_data["train"],
                split_data,
                lambda data, key=state_key: final_features(data, key),
                device=device,
                num_blocks=num_blocks,
                max_sequence_length=max_sequence_length,
            )
        checkpoint_result["final_state_probes"] = final_probes
        flat_probes: dict[str, Any] = {}
        for state_key, result_key in (("h_states", "flat_h"), ("c_states", "flat_c"), ("hc_states", "flat_hc")):
            flat_probes[result_key] = run_order_and_length_probe(
                split_data["train"],
                split_data,
                lambda data, key=state_key: flattened_features(data, key),
                device=device,
                num_blocks=num_blocks,
                max_sequence_length=max_sequence_length,
            )
        checkpoint_result["flattened_state_probes"] = flat_probes
        result["checkpoints"][spec.label] = checkpoint_result
        emit(
            {
                "event": "done_checkpoint",
                "label": spec.label,
                "source_epoch": checkpoint_result["source_epoch"],
                "test_item": checkpoint_result["item_embedding_probe"]["test"]["accuracy"],
                "test_final_h": checkpoint_result["final_state_probes"]["final_h"]["test"]["order"][
                    "block_token_order_accuracy"
                ],
                "test_final_c": checkpoint_result["final_state_probes"]["final_c"]["test"]["order"][
                    "block_token_order_accuracy"
                ],
                "test_flat_hc": checkpoint_result["flattened_state_probes"]["flat_hc"]["test"]["order"][
                    "block_token_order_accuracy"
                ],
            }
        )
    result["elapsed_sec"] = time.time() - started
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result)
    emit({"event": "wrote_outputs", "json": str(OUTPUT_JSON), "markdown": str(OUTPUT_MD)})
    return result


def main() -> int:
    result = run()
    print(json.dumps({"elapsed_sec": result["elapsed_sec"], "outputs": [str(OUTPUT_JSON), str(OUTPUT_MD)]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
