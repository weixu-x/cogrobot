"""Inspect a trained coordinate Corsi model and export a readable report.

Example:
    python corsi/analysis/inspect_coord_model.py \
        --checkpoint corsi_artifacts/coordinate_base/coord_baseline_mixed_2_6/best_model.pt \
        --num-samples 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from corsi.data import CoordinateCorsiDataset
from corsi.models.lstm_coord import CoordLSTMConfig, CoordinateSeq2SeqLSTM
from corsi.rendering import render_board_png, render_board_svg
from corsi.training.device import resolve_torch_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "train_config" not in checkpoint:
        raise ValueError("Checkpoint is missing train_config")
    return checkpoint


def model_config_from_train_config(train_config: Dict[str, object]) -> CoordLSTMConfig:
    input_dim = 2 if train_config["feature_mode"] == "xy" else 4
    return CoordLSTMConfig(
        input_dim=input_dim,
        coord_embedding_dim=int(train_config["coord_embedding_dim"]),
        token_embedding_dim=int(train_config["token_embedding_dim"]),
        hidden_dim=int(train_config["hidden_dim"]),
        num_layers=int(train_config["num_layers"]),
        dropout=float(train_config["dropout"]),
    )


def build_inspection_dataset(train_config: Dict[str, object], num_samples: int, seed: int) -> CoordinateCorsiDataset:
    return CoordinateCorsiDataset(
        num_trials=num_samples,
        seq_len_range=(int(train_config["seq_min"]), int(train_config["seq_max"])),
        feature_mode=str(train_config["feature_mode"]),
        seed=seed,
    )


def round_list(values: Sequence[float], digits: int = 4) -> List[float]:
    return [round(float(value), digits) for value in values]


def round_matrix(values: Sequence[Sequence[float]], digits: int = 4) -> List[List[float]]:
    return [round_list(row, digits=digits) for row in values]


def board_rows_from_layout(layout: Dict[int, Sequence[float]]) -> List[List[int]]:
    grouped: Dict[float, List[tuple[float, int]]] = {}
    for block_id, (x, y) in layout.items():
        grouped.setdefault(float(y), []).append((float(x), int(block_id)))

    rows: List[List[int]] = []
    for y in sorted(grouped.keys(), reverse=True):
        rows.append([block_id for _, block_id in sorted(grouped[y], key=lambda item: item[0])])
    return rows


def render_sequence_board(layout: Dict[int, Sequence[float]], sequence: Sequence[int], title: str) -> str:
    order_map: Dict[int, List[int]] = {}
    for step_index, block_id in enumerate(sequence, start=1):
        order_map.setdefault(int(block_id), []).append(step_index)

    rows = board_rows_from_layout(layout)
    rendered = [title]
    for row in rows:
        cells = []
        for block_id in row:
            orders = ",".join(str(index) for index in order_map.get(block_id, [])) or "-"
            cells.append(f"{block_id}:{orders}")
        rendered.append(" | ".join(cells))
    return "\n".join(rendered)


def topk_from_logits(logits: torch.Tensor, top_k: int) -> List[Dict[str, float]]:
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=min(top_k, probs.numel()))
    return [
        {
            "block_id": int(index.item()),
            "prob": round(float(value.item()), 6),
        }
        for value, index in zip(values, indices)
    ]


@torch.no_grad()
def inspect_sample(model, sample: Dict[str, object], device: torch.device, top_k: int) -> Dict[str, object]:
    coords = torch.tensor(sample["coords"], dtype=torch.float32, device=device).unsqueeze(0)
    targets = torch.tensor(sample["targets"], dtype=torch.long, device=device).unsqueeze(0)
    lengths = torch.tensor([sample["length"]], dtype=torch.long, device=device)

    coord_emb = model.coord_embed(coords)
    hidden, cell = model.encode(coords, lengths)
    teacher_inputs = model._teacher_forcing_inputs(targets)

    teacher_emb = model.token_embedding(teacher_inputs)
    decoder_outputs, _ = model.decoder(teacher_emb, (hidden, cell))
    logits = model.output_head(decoder_outputs).squeeze(0)

    teacher_steps = []
    for step_index in range(int(sample["length"])):
        teacher_steps.append(
            {
                "step": step_index + 1,
                "decoder_input_token": int(teacher_inputs[0, step_index].item()),
                "top_predictions": topk_from_logits(logits[step_index], top_k),
                "target_block": int(targets[0, step_index].item()),
            }
        )

    greedy_predictions = []
    greedy_steps = []
    current_token = torch.full(
        (1, 1),
        fill_value=model.config.start_token_id,
        dtype=torch.long,
        device=device,
    )
    decode_hidden = hidden
    decode_cell = cell

    for step_index in range(int(sample["length"])):
        decoder_emb = model.token_embedding(current_token[:, -1:])
        decoder_output, (decode_hidden, decode_cell) = model.decoder(decoder_emb, (decode_hidden, decode_cell))
        step_logits = model.output_head(decoder_output[:, -1, :]).squeeze(0)
        predicted_block = int(step_logits.argmax(dim=-1).item())
        greedy_predictions.append(predicted_block)
        greedy_steps.append(
            {
                "step": step_index + 1,
                "decoder_input_token": int(current_token[0, -1].item()),
                "top_predictions": topk_from_logits(step_logits, top_k),
                "chosen_block": predicted_block,
                "hidden_norm": round(float(decode_hidden.norm().item()), 6),
                "cell_norm": round(float(decode_cell.norm().item()), 6),
            }
        )
        current_token = torch.cat(
            [current_token, torch.tensor([[predicted_block]], device=device, dtype=torch.long)],
            dim=1,
        )

    return {
        "trial_id": sample["trial_id"],
        "mode": sample["mode"],
        "length": int(sample["length"]),
        "target_sequence": list(sample["targets"]),
        "predicted_sequence": greedy_predictions,
        "coords_features": round_matrix(sample["coords"]),
        "layout": {str(key): round_list(value) for key, value in sample["layout"].items()},
        "coord_embedding_preview": round_matrix(coord_emb.squeeze(0).cpu().tolist(), digits=4),
        "encoder_hidden_last_layer_preview": round_list(hidden[-1, 0, :8].cpu().tolist()),
        "encoder_cell_last_layer_preview": round_list(cell[-1, 0, :8].cpu().tolist()),
        "teacher_forcing_inputs": [int(token) for token in teacher_inputs.squeeze(0).cpu().tolist()],
        "teacher_forcing_steps": teacher_steps,
        "greedy_steps": greedy_steps,
        "target_board": render_sequence_board(sample["layout"], sample["targets"], "Target Order"),
        "predicted_board": render_sequence_board(sample["layout"], greedy_predictions, "Predicted Order"),
    }


def attach_visualizations(sample_reports: Sequence[Dict[str, object]], output_dir: Path) -> List[Dict[str, object]]:
    visuals_dir = output_dir / "board_visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    for index, report in enumerate(sample_reports, start=1):
        target_png = visuals_dir / f"sample_{index:02d}_target.png"
        predicted_png = visuals_dir / f"sample_{index:02d}_predicted.png"
        target_svg = visuals_dir / f"sample_{index:02d}_target.svg"
        predicted_svg = visuals_dir / f"sample_{index:02d}_predicted.svg"

        layout = {int(key): tuple(value) for key, value in report["layout"].items()}
        render_board_png(layout, report["target_sequence"], f"Sample {index} Target", target_png)
        render_board_png(layout, report["predicted_sequence"], f"Sample {index} Predicted", predicted_png)
        render_board_svg(layout, report["target_sequence"], f"Sample {index} Target", target_svg)
        render_board_svg(layout, report["predicted_sequence"], f"Sample {index} Predicted", predicted_svg)

        report["visualizations"] = {
            "target_png": str(target_png.resolve()),
            "predicted_png": str(predicted_png.resolve()),
            "target_svg": str(target_svg.resolve()),
            "predicted_svg": str(predicted_svg.resolve()),
        }

    return list(sample_reports)


def format_step_table(steps: Sequence[Dict[str, object]], label: str) -> str:
    lines = [
        f"### {label}",
        "",
        "| Step | Decoder Input | Top-1 | Top-2 | Top-3 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for step in steps:
        top_predictions = list(step["top_predictions"])
        while len(top_predictions) < 3:
            top_predictions.append({"block_id": "-", "prob": "-"})
        top_cells = [
            f"{item['block_id']} ({item['prob']})" if item["block_id"] != "-" else "-"
            for item in top_predictions[:3]
        ]
        lines.append(
            f"| {step['step']} | {step['decoder_input_token']} | {top_cells[0]} | {top_cells[1]} | {top_cells[2]} |"
        )
    return "\n".join(lines)


def build_markdown_report(
    *,
    checkpoint_path: Path,
    device_info: Dict[str, object],
    train_config: Dict[str, object],
    model,
    sample_reports: Sequence[Dict[str, object]],
) -> str:
    lines = [
        "# Coordinate Model Inspection Report",
        "",
        "## Overview",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Requested device: `{device_info['requested_device']}`",
        f"- Resolved device: `{device_info['resolved_device']}`",
        f"- Feature mode: `{train_config['feature_mode']}`",
        f"- Span range: `{train_config['seq_min']}` to `{train_config['seq_max']}`",
        "",
        "## Model Structure",
        "",
        "```text",
        str(model),
        "```",
        "",
        "## How To Read The Report",
        "",
        "- `coords_features` are the actual model inputs at each time step.",
        "- For `xydxdy`, each row is `[x, y, dx, dy]`.",
        "- `teacher_forcing_inputs` are the decoder tokens used during supervised training.",
        "- `teacher_forcing_steps` show what the decoder would rank highest when fed the true previous token.",
        "- `greedy_steps` show actual inference behavior when the decoder feeds back its own previous prediction.",
        "- `Target Order` and `Predicted Order` are 3x3 board visualizations. Each cell uses `block_id:step_positions`.",
        "",
    ]

    for index, report in enumerate(sample_reports, start=1):
        lines.extend(
            [
                f"## Sample {index}: `{report['trial_id']}`",
                "",
                f"- Length: `{report['length']}`",
                f"- Mode: `{report['mode']}`",
                f"- Target sequence: `{report['target_sequence']}`",
                f"- Predicted sequence: `{report['predicted_sequence']}`",
                "",
                "### Input Features",
                "",
                "| Step | x | y | dx | dy |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for step_index, row in enumerate(report["coords_features"], start=1):
            if len(row) == 2:
                row = [row[0], row[1], "-", "-"]
            lines.append(f"| {step_index} | {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

        lines.extend(
            [
                "",
                "### Encoder Output Preview",
                "",
                f"- Final hidden state preview: `{report['encoder_hidden_last_layer_preview']}`",
                f"- Final cell state preview: `{report['encoder_cell_last_layer_preview']}`",
                "",
                "### Board View",
                "",
                "```text",
                report["target_board"],
                "",
                report["predicted_board"],
                "```",
                "",
                "### Board Images",
                "",
                f"![Sample {index} Target PNG]({report['visualizations']['target_png']})",
                "",
                f"![Sample {index} Predicted PNG]({report['visualizations']['predicted_png']})",
                "",
                f"- Target SVG: `{report['visualizations']['target_svg']}`",
                f"- Predicted SVG: `{report['visualizations']['predicted_svg']}`",
                "",
                format_step_table(report["teacher_forcing_steps"], "Teacher Forcing Decoder Steps"),
                "",
                format_step_table(report["greedy_steps"], "Greedy Decoder Steps"),
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = load_checkpoint(checkpoint_path)

    device, device_info = resolve_torch_device(args.device)
    train_config = dict(checkpoint["train_config"])
    model_config = model_config_from_train_config(train_config)
    model = CoordinateSeq2SeqLSTM(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = build_inspection_dataset(train_config, num_samples=args.num_samples, seed=args.seed)
    sample_reports = [
        inspect_sample(model, dataset[index], device=device, top_k=args.top_k)
        for index in range(len(dataset))
    ]

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint_path.parent / "inspection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_reports = attach_visualizations(sample_reports, output_dir)

    report = build_markdown_report(
        checkpoint_path=checkpoint_path,
        device_info=device_info,
        train_config=train_config,
        model=model,
        sample_reports=sample_reports,
    )
    report_path = output_dir / "inspection_report.md"
    details_path = output_dir / "inspection_details.json"
    summary_path = output_dir / "inspection_summary.json"

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    with open(details_path, "w", encoding="utf-8") as handle:
        json.dump(sample_reports, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "device_info": device_info,
                "num_samples": len(sample_reports),
                "report_path": str(report_path),
                "details_path": str(details_path),
                "visual_dir": str((output_dir / "board_visuals").resolve()),
            },
            handle,
            indent=2,
        )

    print(json.dumps({"report_path": str(report_path), "details_path": str(details_path)}, indent=2))


if __name__ == "__main__":
    main()
