"""Phase 1 binding diagnostics for the V2 LSTM memory-onset checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.experiments.corsi_memory_recall_v2.analysis import compute_sequence_metrics  # noqa: E402
from corsi.experiments.corsi_memory_recall_v2.train import (  # noqa: E402
    _build_model,
    canonical_manifest_path,
    load_checkpoint,
    load_config,
    make_loader,
    move_to_device,
    pad_sequence_metric_batches,
    resolve_device,
)


CONFIG = Path("corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_dmem64.json")
CHECKPOINT = Path(
    "corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/"
    "stage2_seed0_lstm_memory_dmem64_onset_20260629/milestones/epoch_160/best_full_sequence.pt"
)
OUTPUT_JSON = Path("reports/v2_lstm_phase1_binding_diagnostics_20260629.json")
OUTPUT_MD = Path("reports/v2_lstm_phase1_binding_diagnostics_20260629.md")
PROBE_STEPS = 700


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


def train_classifier(
    x: Tensor,
    y: Tensor,
    *,
    classes: int,
    device: torch.device,
    seed: int,
    steps: int = PROBE_STEPS,
) -> tuple[nn.Module, float]:
    torch.manual_seed(int(seed))
    x_dev = x.to(device)
    y_dev = y.long().to(device)
    probe = nn.Linear(int(x.shape[-1]), int(classes)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.05, weight_decay=1e-4)
    last_loss = 0.0
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(x_dev), y_dev)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        if last_loss < 1e-4:
            break
    return probe, last_loss


@torch.no_grad()
def eval_classifier(
    probe: nn.Module,
    x: Tensor,
    y: Tensor,
    *,
    device: torch.device,
    label_name: str,
) -> dict[str, Any]:
    pred = probe(x.to(device)).argmax(dim=-1).cpu()
    target = y.long().cpu()
    correct = pred.eq(target)
    per_label: dict[str, dict[str, Any]] = {}
    for label in sorted(int(value) for value in target.unique().tolist()):
        keep = target.eq(label)
        per_label[str(label)] = {
            "accuracy": float(correct[keep].float().mean().item()),
            "count": int(keep.sum().item()),
        }
    return {
        "accuracy": float(correct.float().mean().item()),
        "count": int(target.numel()),
        f"per_{label_name}": per_label,
    }


def train_order_probe(
    x: Tensor,
    targets: Tensor,
    valid: Tensor,
    *,
    device: torch.device,
    num_blocks: int,
    seed: int,
    steps: int = PROBE_STEPS,
) -> tuple[nn.Module, float]:
    torch.manual_seed(int(seed))
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
    lengths_cpu = lengths.long().cpu()
    correct = pred.eq(targets_cpu) & valid_cpu
    exact: list[int] = []
    per_length_exact: dict[str, list[int]] = {}
    per_length_token: dict[str, list[float]] = {}
    for row in range(int(pred.shape[0])):
        length = int(lengths_cpu[row].item())
        row_valid = valid_cpu[row]
        if length <= 0 or not bool(row_valid.any()):
            row_exact = 0
            row_token = 0.0
        else:
            row_exact = int(bool(pred[row, :length].eq(targets_cpu[row, :length]).all().item()))
            row_token = float(correct[row][row_valid].float().mean().item())
        exact.append(row_exact)
        key = str(length)
        per_length_exact.setdefault(key, []).append(row_exact)
        per_length_token.setdefault(key, []).append(row_token)
    return {
        "block_token_order_accuracy": float(correct[valid_cpu].float().mean().item()) if bool(valid_cpu.any()) else 0.0,
        "known_length_exact_sequence_accuracy": float(sum(exact) / max(len(exact), 1)),
        "sequence_count": int(pred.shape[0]),
        "block_token_count": int(valid_cpu.sum().item()),
        "per_length": {
            key: {
                "count": int(len(per_length_exact[key])),
                "known_length_exact_sequence_accuracy": float(
                    sum(per_length_exact[key]) / max(len(per_length_exact[key]), 1)
                ),
                "block_token_order_accuracy": float(
                    sum(per_length_token[key]) / max(len(per_length_token[key]), 1)
                ),
            }
            for key in sorted(per_length_exact, key=int)
        },
    }


def train_length_probe(
    x: Tensor,
    lengths: Tensor,
    *,
    device: torch.device,
    max_sequence_length: int,
    seed: int,
    steps: int = PROBE_STEPS,
) -> tuple[nn.Module, float]:
    torch.manual_seed(int(seed))
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
    labels = (lengths.clamp(min=1, max=int(max_sequence_length)) - 1).long().cpu()
    pred = probe(x.to(device)).argmax(dim=-1).cpu()
    correct = pred.eq(labels)
    per_length: dict[str, dict[str, Any]] = {}
    for label in sorted(int(value) for value in labels.unique().tolist()):
        keep = labels.eq(label)
        per_length[str(label + 1)] = {
            "accuracy": float(correct[keep].float().mean().item()),
            "count": int(keep.sum().item()),
        }
    return {
        "accuracy": float(correct.float().mean().item()),
        "count": int(labels.numel()),
        "per_length": per_length,
    }


@torch.no_grad()
def collect_item_mean(model: nn.Module, loader: Any, *, device: torch.device) -> Tensor:
    model.eval()
    total: Tensor | None = None
    count = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        model_inputs = batch["model_inputs"]
        presentation = model.encode_presentation(
            images=model_inputs["images"],
            segment_mask=model_inputs["segment_mask"],
            frame_mask=model_inputs["frame_mask"],
            return_traces=False,
        )
        valid = model_inputs["segment_mask"].bool()
        items = presentation["item_embeddings"][valid]
        if total is None:
            total = items.detach().sum(dim=0)
        else:
            total = total + items.detach().sum(dim=0)
        count += int(items.shape[0])
    if total is None or count <= 0:
        raise RuntimeError("could not collect any item embeddings for frozen-identity control")
    return (total / float(count)).detach()


@torch.no_grad()
def collect_split(
    model: nn.Module,
    loader: Any,
    *,
    device: torch.device,
    eos_token_id: int,
    ignore_index: int,
    max_sequence_length: int,
    item_mean: Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    trajectory: dict[str, list[Tensor]] = {"h": [], "c": [], "hc": []}
    final: dict[str, list[Tensor]] = {"h": [], "c": [], "hc": []}
    current_labels: list[Tensor] = []
    position_labels: list[Tensor] = []
    target_blocks: list[Tensor] = []
    target_valid: list[Tensor] = []
    length_rows: list[Tensor] = []
    predictions: list[Tensor] = []
    targets_out: list[Tensor] = []
    masks_out: list[Tensor] = []

    for batch in loader:
        batch = move_to_device(batch, device)
        model_inputs = batch["model_inputs"]
        targets = batch["targets"]["tokens"]
        token_mask = batch["targets"]["token_mask"]
        segment_mask = model_inputs["segment_mask"].bool()
        lengths = target_lengths(targets, token_mask, eos_token_id=eos_token_id, ignore_index=ignore_index)

        if item_mean is None:
            outputs = model(
                images=model_inputs["images"],
                segment_mask=segment_mask,
                frame_mask=model_inputs["frame_mask"],
                max_recall_steps=int(targets.shape[1]),
                return_traces=True,
            )
            memory_traces = outputs["traces"]["memory"]
            logits = outputs["logits"]
        else:
            item = item_mean.to(device=device, dtype=targets.dtype if targets.is_floating_point() else torch.float32)
            item_embeddings = item.view(1, 1, -1).expand(
                int(segment_mask.shape[0]),
                int(segment_mask.shape[1]),
                int(item.shape[-1]),
            )
            item_embeddings = item_embeddings.to(dtype=torch.float32) * segment_mask.to(dtype=torch.float32).unsqueeze(-1)
            memory = model.run_memory(item_embeddings, segment_mask, return_traces=True)
            recall = model.recall_from_memory(memory["final_memory"], max_recall_steps=int(targets.shape[1]))
            memory_traces = memory["traces"]
            logits = recall["logits"]

        h = memory_traces["h_t"]
        c = memory_traces["c_t"]
        hc = torch.cat([h, c], dim=-1)
        rows, steps = segment_mask.nonzero(as_tuple=True)
        trajectory["h"].append(h[rows, steps].detach().cpu())
        trajectory["c"].append(c[rows, steps].detach().cpu())
        trajectory["hc"].append(hc[rows, steps].detach().cpu())
        position_labels.append(steps.detach().cpu().long())
        block_id = batch["metadata"]["block_id"]
        current_labels.append(block_id[rows, steps].detach().cpu().long())

        batch_rows = torch.arange(int(h.shape[0]), device=h.device)
        final_index = (lengths.clamp(min=1) - 1).long()
        final["h"].append(h[batch_rows, final_index].detach().cpu())
        final["c"].append(c[batch_rows, final_index].detach().cpu())
        final["hc"].append(hc[batch_rows, final_index].detach().cpu())

        blocks, valid = block_targets(
            targets,
            token_mask,
            max_sequence_length=max_sequence_length,
            eos_token_id=eos_token_id,
            ignore_index=ignore_index,
        )
        target_blocks.append(blocks)
        target_valid.append(valid)
        length_rows.append(lengths.detach().cpu())
        predictions.append(logits.argmax(dim=-1).detach().cpu())
        targets_out.append(targets.detach().cpu())
        masks_out.append(token_mask.detach().cpu())

    return {
        "trajectory": {key: torch.cat(values, dim=0) for key, values in trajectory.items()},
        "final": {key: torch.cat(values, dim=0) for key, values in final.items()},
        "current_labels": torch.cat(current_labels, dim=0),
        "position_labels": torch.cat(position_labels, dim=0),
        "target_blocks": torch.cat(target_blocks, dim=0),
        "target_valid": torch.cat(target_valid, dim=0),
        "lengths": torch.cat(length_rows, dim=0),
        "predictions": predictions,
        "targets": targets_out,
        "masks": masks_out,
    }


def sequence_metrics_from_split(
    data: dict[str, Any],
    *,
    eos_token_id: int,
    ignore_index: int,
) -> dict[str, Any]:
    predictions, targets, masks = pad_sequence_metric_batches(
        data["predictions"],
        data["targets"],
        data["masks"],
        ignore_index=ignore_index,
        eos_token_id=eos_token_id,
    )
    metrics = compute_sequence_metrics(
        predictions,
        targets,
        masks,
        eos_token_id=eos_token_id,
        ignore_index=ignore_index,
    )
    return {
        "full_sequence_accuracy": float(metrics["full_sequence_accuracy"]),
        "token_accuracy": float(metrics["token_accuracy"]),
        "eos_accuracy": float(metrics["eos_accuracy"]),
        "predicted_length_accuracy": float(metrics["predicted_length_accuracy"]),
        "duplicate_sequence_rate": float(metrics["duplicate_sequence_rate"]),
        "sequence_count": int(metrics["sequence_count"]),
        "per_length_metrics": metrics["per_length_metrics"],
        "behavior_sanity": metrics["behavior_sanity"],
    }


def run_trajectory_probes(
    split_data: dict[str, dict[str, Any]],
    *,
    device: torch.device,
    num_blocks: int,
    max_sequence_length: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for state_key, state_label in (("h", "h_t"), ("c", "c_t"), ("hc", "[h_t;c_t]")):
        train_x = split_data["train"]["trajectory"][state_key]
        item_probe, item_loss = train_classifier(
            train_x,
            split_data["train"]["current_labels"],
            classes=num_blocks,
            device=device,
            seed=2026062901,
        )
        position_probe, position_loss = train_classifier(
            train_x,
            split_data["train"]["position_labels"],
            classes=max_sequence_length,
            device=device,
            seed=2026062902,
        )
        state_result: dict[str, Any] = {
            "state": state_label,
            "current_item_train_loss": item_loss,
            "write_position_train_loss": position_loss,
        }
        for split, data in split_data.items():
            state_result[split] = {
                "current_item": eval_classifier(
                    item_probe,
                    data["trajectory"][state_key],
                    data["current_labels"],
                    device=device,
                    label_name="block",
                ),
                "write_position": eval_classifier(
                    position_probe,
                    data["trajectory"][state_key],
                    data["position_labels"],
                    device=device,
                    label_name="write_step",
                ),
            }
        result[state_key] = state_result
    return result


def run_final_probes(
    split_data: dict[str, dict[str, Any]],
    *,
    device: torch.device,
    num_blocks: int,
    max_sequence_length: int,
    include_order: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for state_key, state_label in (("h", "final h"), ("c", "final c"), ("hc", "final [h;c]")):
        train_x = split_data["train"]["final"][state_key]
        length_probe, length_loss = train_length_probe(
            train_x,
            split_data["train"]["lengths"],
            device=device,
            max_sequence_length=max_sequence_length,
            seed=2026062903,
        )
        state_result: dict[str, Any] = {
            "state": state_label,
            "length_train_loss": length_loss,
        }
        if include_order:
            order_probe, order_loss = train_order_probe(
                train_x,
                split_data["train"]["target_blocks"],
                split_data["train"]["target_valid"],
                device=device,
                num_blocks=num_blocks,
                seed=2026062904,
            )
            state_result["order_train_loss"] = order_loss
        for split, data in split_data.items():
            split_result: dict[str, Any] = {
                "length": eval_length_probe(
                    length_probe,
                    data["final"][state_key],
                    data["lengths"],
                    device=device,
                    max_sequence_length=max_sequence_length,
                )
            }
            if include_order:
                split_result["order"] = eval_order_probe(
                    order_probe,
                    data["final"][state_key],
                    data["target_blocks"],
                    data["target_valid"],
                    data["lengths"],
                    device=device,
                    num_blocks=num_blocks,
                )
            state_result[split] = split_result
        result[state_key] = state_result
    return result


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def decision_readout(result: dict[str, Any]) -> list[str]:
    original = result["original"]
    frozen = result["identity_frozen"]
    hc_traj = original["trajectory_probes"]["hc"]["test"]
    hc_final = original["final_probes"]["hc"]["test"]
    frozen_hc = frozen["trajectory_position_probes"]["hc"]["test"]["write_position"]
    behavior = original["behavior"]["test"]["per_length_metrics"]
    len2_exact = float(behavior.get("2", {}).get("full_sequence_accuracy", 0.0))
    len3_exact = float(behavior.get("3", {}).get("full_sequence_accuracy", 0.0))
    current_item = float(hc_traj["current_item"]["accuracy"])
    write_position = float(hc_traj["write_position"]["accuracy"])
    final_order = float(hc_final["order"]["block_token_order_accuracy"])
    final_length = float(hc_final["length"]["accuracy"])
    frozen_position = float(frozen_hc["accuracy"])

    lines = [
        (
            f"Trajectory [h_t;c_t] test probes: current item {fmt(current_item)}, "
            f"write position {fmt(write_position)}."
        ),
        (
            f"Final [h;c] test probes: ordered identity token {fmt(final_order)}, "
            f"length {fmt(final_length)}."
        ),
        f"Short-sequence behavior: test length-2 exact {fmt(len2_exact)}, length-3 exact {fmt(len3_exact)}.",
        f"Identity-frozen control: [h_t;c_t] write-position probe {fmt(frozen_position)}.",
    ]
    if current_item >= 0.9 and write_position >= 0.9 and min(len2_exact, len3_exact) >= 0.8:
        lines.append(
            "Decision: current item and step are present in the write trajectory, and short sequences are solved. "
            "This does not match a catastrophic item-write failure or a pure capacity squeeze."
        )
    elif write_position >= 0.9 and current_item < 0.6:
        lines.append(
            "Decision: position is decodable but current identity is weak in the write trajectory. "
            "This matches a binding/write-in identity failure."
        )
    else:
        lines.append(
            "Decision: probes show mixed evidence. Treat the problem as write/readout binding plus optimization "
            "until a capacity sweep shows a clear length cliff that shifts with D_mem."
        )
    if final_order + 0.15 < current_item:
        lines.append(
            "The main bottleneck is after per-step writing: ordered identity is much stronger in trajectory states "
            "than in the final compressed state."
        )
    if frozen_position >= 0.9:
        lines.append(
            "The constant-identity control can still encode write position, so position dynamics are not broken by themselves."
        )
    return lines


def write_markdown(
    result: dict[str, Any],
    *,
    output_json: Path,
    output_md: Path,
    config: Path,
    checkpoint: Path,
) -> None:
    original = result["original"]
    frozen = result["identity_frozen"]
    traj_rows = []
    for state_key, label in (("h", "h_t"), ("c", "c_t"), ("hc", "[h_t;c_t]")):
        test = original["trajectory_probes"][state_key]["test"]
        traj_rows.append(
            [
                label,
                fmt(test["current_item"]["accuracy"]),
                fmt(test["write_position"]["accuracy"]),
                test["current_item"]["count"],
            ]
        )
    final_rows = []
    for state_key, label in (("h", "final h"), ("c", "final c"), ("hc", "final [h;c]")):
        test = original["final_probes"][state_key]["test"]
        short = test["order"]["per_length"]
        final_rows.append(
            [
                label,
                fmt(test["order"]["block_token_order_accuracy"]),
                fmt(test["order"]["known_length_exact_sequence_accuracy"]),
                fmt(test["length"]["accuracy"]),
                fmt(short["2"]["known_length_exact_sequence_accuracy"]),
                fmt(short["3"]["known_length_exact_sequence_accuracy"]),
            ]
        )
    behavior = original["behavior"]["test"]["per_length_metrics"]
    sanity = original["behavior"]["test"].get("behavior_sanity", {})
    serial_shape = sanity.get("serial_position_shape", {})
    distance_gradient = sanity.get("transposition_distance_gradient", {})
    behavior_rows = [
        [
            length,
            item["count"],
            fmt(item["full_sequence_accuracy"]),
            fmt(item["token_accuracy"]),
            fmt(item["duplicate_sequence_rate"]),
        ]
        for length, item in sorted(behavior.items(), key=lambda kv: int(kv[0]))
    ]
    frozen_rows = []
    for state_key, label in (("h", "h_t"), ("c", "c_t"), ("hc", "[h_t;c_t]")):
        traj_test = frozen["trajectory_position_probes"][state_key]["test"]["write_position"]
        final_test = frozen["final_length_probes"][state_key]["test"]["length"]
        frozen_rows.append([label, fmt(traj_test["accuracy"]), fmt(final_test["accuracy"]), traj_test["count"]])
    checkpoint_epoch = result.get("checkpoint_epoch", "unknown")
    lines = [
        "# V2 LSTM Binding Diagnostics - 2026-06-29",
        "",
        f"Scope: frozen probes for checkpoint epoch {checkpoint_epoch}. Only probe heads are trained.",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{output_json}`",
        f"- Report: `{output_md}`",
        f"- Config: `{config}`",
        f"- Checkpoint: `{checkpoint}`",
        "",
        "## 1.1 Write-Step Linear Probes",
        "",
        markdown_table(["state", "current item acc", "write position acc", "states"], traj_rows),
        "",
        "## 1.2 Short-Sequence Behavior",
        "",
        markdown_table(["length", "count", "exact", "token", "duplicate"], behavior_rows),
        "",
        "## 3 Behavior Sanity Check",
        "",
        markdown_table(
            ["first", "middle", "last", "edge", "U-score"],
            [
                [
                    fmt(serial_shape.get("first_accuracy", 0.0)),
                    fmt(serial_shape.get("middle_accuracy", 0.0)),
                    fmt(serial_shape.get("last_accuracy", 0.0)),
                    fmt(serial_shape.get("edge_accuracy", 0.0)),
                    fmt(serial_shape.get("u_shape_score", 0.0)),
                ]
            ],
        ),
        "",
        markdown_table(
            ["adjacent", "far", "adjacent fraction", "distance-dependent", "distance counts"],
            [
                [
                    distance_gradient.get("adjacent_count", 0),
                    distance_gradient.get("far_count", 0),
                    fmt(distance_gradient.get("adjacent_fraction", 0.0)),
                    str(bool(distance_gradient.get("distance_dependent", False))),
                    json.dumps(distance_gradient.get("distance_counts", {}), sort_keys=True),
                ]
            ],
        ),
        "",
        "## 1.1/1.2 Final-State Probes",
        "",
        markdown_table(
            ["state", "order token", "known-L exact", "length acc", "L2 exact", "L3 exact"],
            final_rows,
        ),
        "",
        "## 1.3 Identity-Frozen Control",
        "",
        "All item embeddings are replaced by the train-set mean item embedding before memory write-in.",
        "",
        markdown_table(["state", "write position acc", "final length acc", "states"], frozen_rows),
        "",
        "## Decision Readout",
        "",
        *decision_readout(result),
        "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    config: Path = CONFIG,
    checkpoint: Path = CHECKPOINT,
    output_json: Path = OUTPUT_JSON,
    output_md: Path = OUTPUT_MD,
    device_name: str | None = None,
) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(config)
    if device_name:
        cfg["device"] = device_name
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

    model = _build_model(cfg, stage=2, manifest=manifest).to(device)
    payload = load_checkpoint(checkpoint, model=model, map_location=device)
    emit({"event": "loaded_checkpoint", "epoch": int(payload.get("epoch", -1)), "device": str(device)})

    item_mean = collect_item_mean(model, loaders["train"], device=device)
    emit({"event": "collected_item_mean", "dim": int(item_mean.numel())})

    original_data = {
        split: collect_split(
            model,
            loaders[split],
            device=device,
            eos_token_id=eos_token_id,
            ignore_index=ignore_index,
            max_sequence_length=max_sequence_length,
            item_mean=None,
        )
        for split in ("train", "val", "test")
    }
    emit({"event": "collected_original"})
    frozen_data = {
        split: collect_split(
            model,
            loaders[split],
            device=device,
            eos_token_id=eos_token_id,
            ignore_index=ignore_index,
            max_sequence_length=max_sequence_length,
            item_mean=item_mean,
        )
        for split in ("train", "val", "test")
    }
    emit({"event": "collected_identity_frozen"})

    original = {
        "behavior": {
            split: sequence_metrics_from_split(
                data,
                eos_token_id=eos_token_id,
                ignore_index=ignore_index,
            )
            for split, data in original_data.items()
        },
        "trajectory_probes": run_trajectory_probes(
            original_data,
            device=device,
            num_blocks=num_blocks,
            max_sequence_length=max_sequence_length,
        ),
        "final_probes": run_final_probes(
            original_data,
            device=device,
            num_blocks=num_blocks,
            max_sequence_length=max_sequence_length,
            include_order=True,
        ),
    }
    emit({"event": "finished_original_probes"})
    identity_frozen = {
        "behavior": {
            split: sequence_metrics_from_split(
                data,
                eos_token_id=eos_token_id,
                ignore_index=ignore_index,
            )
            for split, data in frozen_data.items()
        },
        "trajectory_position_probes": run_trajectory_probes(
            frozen_data,
            device=device,
            num_blocks=num_blocks,
            max_sequence_length=max_sequence_length,
        ),
        "final_length_probes": run_final_probes(
            frozen_data,
            device=device,
            num_blocks=num_blocks,
            max_sequence_length=max_sequence_length,
            include_order=False,
        ),
    }
    emit({"event": "finished_identity_frozen_probes"})

    result: dict[str, Any] = {
        "created_time": time.time(),
        "elapsed_sec": time.time() - started,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "device": str(device),
        "canonical_fingerprint": manifest.get("canonical_fingerprint"),
        "num_blocks": num_blocks,
        "max_sequence_length": max_sequence_length,
        "eos_token_id": eos_token_id,
        "ignore_index": ignore_index,
        "probe_steps": PROBE_STEPS,
        "original": original,
        "identity_frozen": identity_frozen,
        "decision_readout": decision_readout({"original": original, "identity_frozen": identity_frozen}),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(
        result,
        output_json=output_json,
        output_md=output_md,
        config=config,
        checkpoint=checkpoint,
    )
    emit({"event": "wrote_outputs", "json": str(output_json), "markdown": str(output_md)})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run reusable Phase 1 binding diagnostics for a V2 checkpoint.")
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--device", default="")
    args = parser.parse_args(argv)
    result = run(
        config=Path(args.config),
        checkpoint=Path(args.checkpoint),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
        device_name=str(args.device) or None,
    )
    print(
        json.dumps(
            {"elapsed_sec": result["elapsed_sec"], "outputs": [str(args.output_json), str(args.output_md)]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
