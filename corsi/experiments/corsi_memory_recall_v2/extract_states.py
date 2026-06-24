"""Export V2 memory and recurrent traces from Lane C-compatible checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from corsi.experiments.corsi_memory_recall_v2.train import (
    canonical_manifest_path,
    load_checkpoint,
    load_config,
    make_loader,
    move_to_device,
    resolve_device,
    _build_model,
    _call_model,
)


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _collect_trace_arrays(outputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    traces = outputs.get("traces", outputs.get("memory_traces", {}))
    arrays: dict[str, np.ndarray] = {}
    if isinstance(traces, Mapping):
        for key, value in traces.items():
            if torch.is_tensor(value) or isinstance(value, np.ndarray):
                arrays[str(key)] = _as_numpy(value)
    for key in (
        "memory",
        "memory_state",
        "final_memory",
        "memory_before_noise",
        "recall_memory",
        "item_embeddings",
        "recall_hidden",
        "recall_inputs",
    ):
        if key in outputs and (torch.is_tensor(outputs[key]) or isinstance(outputs[key], np.ndarray)):
            arrays[key] = _as_numpy(outputs[key])
    return arrays


@torch.no_grad()
def export_states(
    *,
    config: Mapping[str, Any],
    checkpoint: str | Path,
    split: str,
    output_path: str | Path,
    batch_size: int = 8,
    device_name: str = "auto",
) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("V2 state export requires h5py") from exc

    device = resolve_device(device_name)
    manifest_path = canonical_manifest_path(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loader = make_loader(manifest_path, split=split, batch_size=batch_size, shuffle=False)
    model = _build_model(config, stage=2, manifest=manifest).to(device)
    payload = load_checkpoint(checkpoint, model=model, map_location=device)
    model.eval()

    collected: dict[str, list[np.ndarray]] = {}
    seq_ids: list[bytes] = []
    lengths: list[int] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = dict(_call_model(model, batch, stage=2))
        if hasattr(model, "extract_states"):
            outputs.update(model.extract_states(batch["model_inputs"]))
        for key, array in _collect_trace_arrays(outputs).items():
            collected.setdefault(key, []).append(array)
        seq_ids.extend(str(value).encode("utf-8") for value in batch["metadata"]["seq_id"])
        lengths.extend(int(value) for value in batch["metadata"]["length"].detach().cpu().tolist())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("seq_id", data=np.asarray(seq_ids))
        handle.create_dataset("length", data=np.asarray(lengths, dtype=np.int64))
        for key, arrays in collected.items():
            try:
                handle.create_dataset(key, data=np.concatenate(arrays, axis=0))
            except ValueError:
                group = handle.create_group(key)
                for index, array in enumerate(arrays):
                    group.create_dataset(str(index), data=array)
        handle.attrs["checkpoint"] = str(checkpoint)
        handle.attrs["split"] = split
        handle.attrs["epoch"] = int(payload.get("epoch", -1))
        handle.attrs["canonical_fingerprint"] = str(manifest.get("canonical_fingerprint", ""))
    metadata = {
        "path": str(output_path),
        "format": "hdf5",
        "rows": len(seq_ids),
        "split": split,
        "checkpoint": str(checkpoint),
        "fields": sorted(["seq_id", "length", *collected.keys()]),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Corsi memory-recall V2 state traces.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    result = export_states(
        config=load_config(args.config),
        checkpoint=args.checkpoint,
        split=args.split,
        output_path=args.output,
        batch_size=int(args.batch_size),
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
