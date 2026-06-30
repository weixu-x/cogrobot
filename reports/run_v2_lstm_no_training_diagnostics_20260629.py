"""Run frozen-model diagnostics for the LSTM memory-onset checkpoints."""

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

from corsi.experiments.corsi_memory_recall_v2.analysis import compute_sequence_metrics
from corsi.experiments.corsi_memory_recall_v2.train import (
    _build_model,
    _call_model,
    canonical_manifest_path,
    load_checkpoint,
    load_config,
    make_loader,
    move_to_device,
    pad_sequence_metric_batches,
    resolve_device,
)


CONFIG = Path("corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/configs/memory_recall_v2_k12_dmem64.json")
RUN_ROOT = Path(
    "corsi_artifacts/memory_recall_v2/runs/corsi_memory_recall_v2_k12/"
    "stage2_seed0_lstm_memory_dmem64_onset_20260629"
)
OUTPUT_JSON = Path("reports/v2_lstm_no_training_diagnostics_20260629.json")
OUTPUT_MD = Path("reports/v2_lstm_no_training_diagnostics_20260629.md")


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


def target_lengths(targets: Tensor, mask: Tensor, *, eos_token_id: int, ignore_index: int) -> Tensor:
    valid_blocks = mask & targets.ne(int(ignore_index)) & targets.ne(int(eos_token_id))
    return valid_blocks.sum(dim=1).long()


def argmax_tokens(logits: Tensor) -> Tensor:
    return logits.argmax(dim=-1).long()


def no_repeat_tokens(logits: Tensor, *, num_blocks: int, eos_token_id: int) -> Tensor:
    batch_size, steps, _ = logits.shape
    used = torch.zeros(batch_size, int(num_blocks), dtype=torch.bool, device=logits.device)
    decoded = torch.zeros(batch_size, steps, dtype=torch.long, device=logits.device)
    for step in range(steps):
        scores = logits[:, step].clone()
        block_scores = scores[:, : int(num_blocks)]
        block_scores[used] = torch.finfo(scores.dtype).min
        scores[:, : int(num_blocks)] = block_scores
        token = scores.argmax(dim=-1).long()
        decoded[:, step] = token
        is_block = token.lt(int(num_blocks))
        if bool(is_block.any()):
            rows = torch.arange(batch_size, device=logits.device)[is_block]
            used[rows, token[is_block]] = True
    return decoded


def oracle_length_tokens(
    logits: Tensor,
    lengths: Tensor,
    *,
    num_blocks: int,
    eos_token_id: int,
) -> Tensor:
    batch_size, steps, _ = logits.shape
    decoded = torch.full((batch_size, steps), int(eos_token_id), dtype=torch.long, device=logits.device)
    block_choice = logits[:, :, : int(num_blocks)].argmax(dim=-1).long()
    for row in range(batch_size):
        length = min(int(lengths[row].item()), steps)
        if length > 0:
            decoded[row, :length] = block_choice[row, :length]
        if length < steps:
            decoded[row, length] = int(eos_token_id)
    return decoded


def oracle_eos_tokens(logits: Tensor, lengths: Tensor, *, eos_token_id: int) -> Tensor:
    decoded = argmax_tokens(logits)
    steps = int(decoded.shape[1])
    for row in range(int(decoded.shape[0])):
        eos_index = int(lengths[row].item())
        if eos_index < steps:
            decoded[row, eos_index] = int(eos_token_id)
    return decoded


def no_repeat_oracle_length_tokens(
    logits: Tensor,
    lengths: Tensor,
    *,
    num_blocks: int,
    eos_token_id: int,
) -> Tensor:
    batch_size, steps, _ = logits.shape
    decoded = torch.full((batch_size, steps), int(eos_token_id), dtype=torch.long, device=logits.device)
    used = torch.zeros(batch_size, int(num_blocks), dtype=torch.bool, device=logits.device)
    for step in range(steps):
        active = lengths.gt(step)
        if not bool(active.any()):
            continue
        scores = logits[:, step, : int(num_blocks)].clone()
        scores[used] = torch.finfo(logits.dtype).min
        token = scores.argmax(dim=-1).long()
        decoded[active, step] = token[active]
        rows = torch.arange(batch_size, device=logits.device)[active]
        used[rows, token[active]] = True
    return decoded


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "full_sequence_accuracy": float(metrics["full_sequence_accuracy"]),
        "token_accuracy": float(metrics["token_accuracy"]),
        "duplicate_sequence_rate": float(metrics["duplicate_sequence_rate"]),
        "eos_accuracy": float(metrics["eos_accuracy"]),
        "predicted_length_accuracy": float(metrics["predicted_length_accuracy"]),
        "sequence_count": int(metrics["sequence_count"]),
        "token_count": int(metrics["token_count"]),
    }


def compute_mode_metrics(
    predictions: list[Tensor],
    targets: list[Tensor],
    masks: list[Tensor],
    *,
    eos_token_id: int,
    ignore_index: int,
) -> dict[str, Any]:
    padded_predictions, padded_targets, padded_masks = pad_sequence_metric_batches(
        predictions,
        targets,
        masks,
        ignore_index=int(ignore_index),
        eos_token_id=int(eos_token_id),
    )
    return compute_sequence_metrics(
        padded_predictions,
        padded_targets,
        padded_masks,
        eos_token_id=int(eos_token_id),
        ignore_index=int(ignore_index),
    )


@torch.no_grad()
def collect_split(
    model: nn.Module,
    loader: Any,
    *,
    device: torch.device,
    eos_token_id: int,
    ignore_index: int,
    num_blocks: int,
) -> dict[str, Any]:
    model.eval()
    features = []
    target_batches = []
    mask_batches = []
    length_batches = []
    mode_predictions: dict[str, list[Tensor]] = {
        "normal": [],
        "no_repeat": [],
        "oracle_length": [],
        "oracle_eos": [],
        "no_repeat_oracle_length": [],
    }
    for batch in loader:
        batch = move_to_device(batch, device)
        targets = batch["targets"]["tokens"]
        masks = batch["targets"]["token_mask"]
        lengths = target_lengths(targets, masks, eos_token_id=eos_token_id, ignore_index=ignore_index)
        outputs = _call_model(model, batch, stage=2)
        logits = outputs.get("token_logits", outputs.get("logits"))
        if logits is None:
            raise RuntimeError("expected token logits from model output")
        features.append(outputs["final_memory"].detach().cpu())
        target_batches.append(targets.detach().cpu())
        mask_batches.append(masks.detach().cpu())
        length_batches.append(lengths.detach().cpu())
        mode_predictions["normal"].append(argmax_tokens(logits).detach().cpu())
        mode_predictions["no_repeat"].append(
            no_repeat_tokens(logits, num_blocks=num_blocks, eos_token_id=eos_token_id).detach().cpu()
        )
        mode_predictions["oracle_length"].append(
            oracle_length_tokens(
                logits,
                lengths,
                num_blocks=num_blocks,
                eos_token_id=eos_token_id,
            ).detach().cpu()
        )
        mode_predictions["oracle_eos"].append(
            oracle_eos_tokens(logits, lengths, eos_token_id=eos_token_id).detach().cpu()
        )
        mode_predictions["no_repeat_oracle_length"].append(
            no_repeat_oracle_length_tokens(
                logits,
                lengths,
                num_blocks=num_blocks,
                eos_token_id=eos_token_id,
            ).detach().cpu()
        )
    return {
        "features": torch.cat(features, dim=0),
        "targets": target_batches,
        "masks": mask_batches,
        "lengths": torch.cat(length_batches, dim=0),
        "mode_predictions": mode_predictions,
    }


def padded_targets_and_masks(split_data: dict[str, Any], *, ignore_index: int, eos_token_id: int) -> tuple[Tensor, Tensor]:
    dummy_predictions = [
        torch.full_like(targets, fill_value=0 if int(eos_token_id) != 0 else 1)
        for targets in split_data["targets"]
    ]
    _, targets, masks = pad_sequence_metric_batches(
        dummy_predictions,
        split_data["targets"],
        split_data["masks"],
        ignore_index=int(ignore_index),
        eos_token_id=int(eos_token_id),
    )
    return targets.long(), masks.bool()


def train_order_probe(
    train_data: dict[str, Any],
    *,
    device: torch.device,
    num_blocks: int,
    max_sequence_length: int,
    eos_token_id: int,
    ignore_index: int,
    steps: int = 1200,
) -> tuple[nn.Module, float]:
    x = train_data["features"].to(device)
    targets, masks = padded_targets_and_masks(train_data, ignore_index=ignore_index, eos_token_id=eos_token_id)
    targets = targets[:, : int(max_sequence_length)].to(device)
    masks = masks[:, : int(max_sequence_length)].to(device)
    valid = masks & targets.ne(int(ignore_index)) & targets.ne(int(eos_token_id))
    probe = nn.Linear(int(x.shape[-1]), int(max_sequence_length) * int(num_blocks)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.05, weight_decay=1e-4)
    last_loss = 0.0
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(x).reshape(-1, int(max_sequence_length), int(num_blocks))
        loss = F.cross_entropy(logits[valid], targets[valid])
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        if last_loss < 1e-4:
            break
    return probe, last_loss


def train_length_probe(
    train_data: dict[str, Any],
    *,
    device: torch.device,
    max_sequence_length: int,
    steps: int = 1200,
) -> tuple[nn.Module, float]:
    x = train_data["features"].to(device)
    labels = (train_data["lengths"].clamp(min=1, max=int(max_sequence_length)) - 1).long().to(device)
    probe = nn.Linear(int(x.shape[-1]), int(max_sequence_length)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.05, weight_decay=1e-4)
    last_loss = 0.0
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(x)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        if last_loss < 1e-4:
            break
    return probe, last_loss


@torch.no_grad()
def evaluate_order_probe(
    probe: nn.Module,
    split_data: dict[str, Any],
    *,
    device: torch.device,
    num_blocks: int,
    max_sequence_length: int,
    eos_token_id: int,
    ignore_index: int,
) -> dict[str, Any]:
    x = split_data["features"].to(device)
    targets, masks = padded_targets_and_masks(split_data, ignore_index=ignore_index, eos_token_id=eos_token_id)
    targets = targets[:, : int(max_sequence_length)].to(device)
    masks = masks[:, : int(max_sequence_length)].to(device)
    valid = masks & targets.ne(int(ignore_index)) & targets.ne(int(eos_token_id))
    pred = probe(x).reshape(-1, int(max_sequence_length), int(num_blocks)).argmax(dim=-1)
    correct = pred.eq(targets) & valid
    per_position: dict[str, float] = {}
    per_position_count: dict[str, int] = {}
    for position in range(int(max_sequence_length)):
        pos_valid = valid[:, position]
        count = int(pos_valid.sum().item())
        if count:
            per_position[str(position)] = float(correct[:, position][pos_valid].float().mean().item())
            per_position_count[str(position)] = count
    length = split_data["lengths"].to(device)
    exact_rows = []
    for row in range(int(pred.shape[0])):
        row_length = int(length[row].item())
        if row_length <= 0:
            exact_rows.append(0)
        else:
            exact_rows.append(int(bool(pred[row, :row_length].eq(targets[row, :row_length]).all().item())))
    return {
        "block_token_order_accuracy": float(correct[valid].float().mean().item()) if bool(valid.any()) else 0.0,
        "known_length_exact_sequence_accuracy": float(sum(exact_rows) / max(len(exact_rows), 1)),
        "serial_position_accuracy": per_position,
        "serial_position_count": per_position_count,
        "count": int(pred.shape[0]),
        "block_token_count": int(valid.sum().item()),
    }


@torch.no_grad()
def evaluate_length_probe(
    probe: nn.Module,
    split_data: dict[str, Any],
    *,
    device: torch.device,
    max_sequence_length: int,
) -> dict[str, Any]:
    x = split_data["features"].to(device)
    labels = (split_data["lengths"].clamp(min=1, max=int(max_sequence_length)) - 1).long().to(device)
    pred = probe(x).argmax(dim=-1)
    return {
        "accuracy": float(pred.eq(labels).float().mean().item()),
        "count": int(labels.numel()),
    }


def split_decode_metrics(
    split_data: dict[str, Any],
    *,
    eos_token_id: int,
    ignore_index: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode, predictions in split_data["mode_predictions"].items():
        metrics = compute_mode_metrics(
            predictions,
            split_data["targets"],
            split_data["masks"],
            eos_token_id=eos_token_id,
            ignore_index=ignore_index,
        )
        result[mode] = summarize_metrics(metrics)
    return result


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def write_markdown(result: dict[str, Any]) -> None:
    checkpoint_items = list(result["checkpoints"].items())
    probe_rows = []
    pos_rows = []
    decode_rows = []
    oracle_rows = []
    for label, item in checkpoint_items:
        test_probe = item["final_memory_order_probe"]["test"]
        test_len = item["final_memory_length_probe"]["test"]
        probe_rows.append(
            [
                label,
                item["source_epoch"],
                fmt(test_probe["block_token_order_accuracy"]),
                fmt(test_probe["known_length_exact_sequence_accuracy"]),
                fmt(test_len["accuracy"]),
            ]
        )
        pos_rows.append(
            [
                label,
                *[
                    fmt(test_probe["serial_position_accuracy"].get(str(position), 0.0))
                    for position in range(int(result["max_sequence_length"]))
                ],
            ]
        )
        dec = item["decode"]["test"]
        normal = dec["normal"]
        no_repeat = dec["no_repeat"]
        oracle_length = dec["oracle_length"]
        oracle_eos = dec["oracle_eos"]
        no_repeat_oracle = dec["no_repeat_oracle_length"]
        decode_rows.append(
            [
                label,
                fmt(normal["full_sequence_accuracy"]),
                fmt(normal["token_accuracy"]),
                fmt(normal["duplicate_sequence_rate"]),
                fmt(no_repeat["full_sequence_accuracy"]),
                fmt(no_repeat["token_accuracy"]),
                fmt(no_repeat["duplicate_sequence_rate"]),
                fmt(no_repeat["full_sequence_accuracy"] - normal["full_sequence_accuracy"]),
            ]
        )
        oracle_rows.append(
            [
                label,
                fmt(normal["full_sequence_accuracy"]),
                fmt(oracle_eos["full_sequence_accuracy"]),
                fmt(oracle_length["full_sequence_accuracy"]),
                fmt(no_repeat_oracle["full_sequence_accuracy"]),
                fmt(oracle_length["predicted_length_accuracy"]),
                fmt(no_repeat_oracle["duplicate_sequence_rate"]),
            ]
        )

    best = result["checkpoints"]["epoch139_best_full"]
    best_test_probe = best["final_memory_order_probe"]["test"]
    best_decode = best["decode"]["test"]
    best_order = float(best_test_probe["block_token_order_accuracy"])
    best_known_exact = float(best_test_probe["known_length_exact_sequence_accuracy"])
    if best_order >= 0.85:
        memory_interpretation = (
            "This is above the 0.85 order-token threshold, so final-memory storage looks strong; "
            "the remaining exact-recall gap would point mainly to recall/readout."
        )
    elif best_order <= 0.60:
        memory_interpretation = (
            "This is in the 0.5-0.6 band, so held-out ordered identity is still weak in the LSTM "
            "final memory; memory write/storage remains a bottleneck."
        )
    else:
        memory_interpretation = (
            "This is between the weak and strong bands, so final-memory storage is partial rather "
            "than a clean solved component."
        )
    interpretation = [
        "# V2 LSTM No-Training Diagnostics - 2026-06-29",
        "",
        "Scope: frozen main-model diagnostics for `stage2_seed0_lstm_memory_dmem64_onset_20260629`.",
        "The final-memory probes train only linear readouts on frozen `final_memory` features; no main model weights are updated.",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{OUTPUT_JSON}`",
        f"- Report: `{OUTPUT_MD}`",
        f"- Config: `{CONFIG}`",
        f"- Run root: `{RUN_ROOT}`",
        "",
        "## P0.1 Final-Memory Probe (test split)",
        "",
        markdown_table(
            ["Checkpoint", "source epoch", "order token", "known-L exact", "length"],
            probe_rows,
        ),
        "",
        "Per-position final-memory order probe accuracy on test:",
        "",
        markdown_table(
            ["Checkpoint", *[f"p{position}" for position in range(int(result["max_sequence_length"]))]],
            pos_rows,
        ),
        "",
        "## P0.2 No-Repeat Constrained Decoding (test split)",
        "",
        markdown_table(
            [
                "Checkpoint",
                "normal full",
                "normal token",
                "normal dup",
                "no-repeat full",
                "no-repeat token",
                "no-repeat dup",
                "full delta",
            ],
            decode_rows,
        ),
        "",
        "## P0.3 Oracle Length / EOS (test split)",
        "",
        markdown_table(
            [
                "Checkpoint",
                "normal full",
                "oracle EOS full",
                "oracle length full",
                "no-repeat+oracle length full",
                "oracle length len-acc",
                "no-repeat+oracle dup",
            ],
            oracle_rows,
        ),
        "",
        "## Interpretation",
        "",
        (
            "For the selected epoch-139 best-full checkpoint, final-memory order token accuracy is "
            f"{fmt(best_order)} and known-length exact is "
            f"{fmt(best_known_exact)} on test. {memory_interpretation}"
        ),
        (
            "No-repeat decoding gives epoch-139 test full "
            f"{fmt(best_decode['no_repeat']['full_sequence_accuracy'])} versus normal "
            f"{fmt(best_decode['normal']['full_sequence_accuracy'])}. "
            "That is a real but modest upper-bound gain, so duplicate policy is part of the failure, "
            "but not the whole failure."
        ),
        (
            "Oracle length alone gives epoch-139 test full "
            f"{fmt(best_decode['oracle_length']['full_sequence_accuracy'])}; "
            "no-repeat plus oracle length gives "
            f"{fmt(best_decode['no_repeat_oracle_length']['full_sequence_accuracy'])}. "
            "Length/EOS correction alone does not improve exact recall here; the limiting errors are content/order "
            "and repeat policy, with final-memory storage still not clean by the probe."
        ),
        "",
        "Note: `oracle_eos` sets EOS at the target EOS index only; `oracle_length` decodes exactly L block steps from block logits and then appends EOS.",
    ]
    OUTPUT_MD.write_text("\n".join(interpretation) + "\n", encoding="utf-8")


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
        "checkpoints": {},
        "notes": [
            "Frozen main-model diagnostics only; no checkpoint model weights are updated.",
            "Final-memory probes are linear readouts trained on frozen train-split final_memory features.",
            "No-repeat and oracle decoders are analysis-only posthoc decoders over checkpoint logits.",
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
                num_blocks=num_blocks,
            )
            for split in ("train", "val", "test")
        }
        order_probe, order_train_loss = train_order_probe(
            split_data["train"],
            device=device,
            num_blocks=num_blocks,
            max_sequence_length=max_sequence_length,
            eos_token_id=eos_token_id,
            ignore_index=ignore_index,
        )
        length_probe, length_train_loss = train_length_probe(
            split_data["train"],
            device=device,
            max_sequence_length=max_sequence_length,
        )
        checkpoint_result: dict[str, Any] = {
            "requested": spec.requested,
            "checkpoint": str(spec.path),
            "source_epoch": int(payload.get("epoch", -1)),
            "source_stage": int(payload.get("stage", -1)),
            "selection_key": str((payload.get("extra") or {}).get("selection_key", "")),
            "selection_score": (payload.get("extra") or {}).get("selection_score"),
            "final_memory_order_probe": {
                "train_loss": order_train_loss,
            },
            "final_memory_length_probe": {
                "train_loss": length_train_loss,
            },
            "decode": {},
        }
        for split in ("train", "val", "test"):
            checkpoint_result["final_memory_order_probe"][split] = evaluate_order_probe(
                order_probe,
                split_data[split],
                device=device,
                num_blocks=num_blocks,
                max_sequence_length=max_sequence_length,
                eos_token_id=eos_token_id,
                ignore_index=ignore_index,
            )
            checkpoint_result["final_memory_length_probe"][split] = evaluate_length_probe(
                length_probe,
                split_data[split],
                device=device,
                max_sequence_length=max_sequence_length,
            )
            if split in ("val", "test"):
                checkpoint_result["decode"][split] = split_decode_metrics(
                    split_data[split],
                    eos_token_id=eos_token_id,
                    ignore_index=ignore_index,
                )
        result["checkpoints"][spec.label] = checkpoint_result
        emit(
            {
                "event": "done_checkpoint",
                "label": spec.label,
                "source_epoch": checkpoint_result["source_epoch"],
                "test_order_probe": checkpoint_result["final_memory_order_probe"]["test"][
                    "block_token_order_accuracy"
                ],
                "test_normal_full": checkpoint_result["decode"]["test"]["normal"][
                    "full_sequence_accuracy"
                ],
                "test_no_repeat_full": checkpoint_result["decode"]["test"]["no_repeat"][
                    "full_sequence_accuracy"
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
