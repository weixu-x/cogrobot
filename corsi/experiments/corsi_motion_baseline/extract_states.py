"""Export hidden states and gate traces for best visual-joint checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from corsi.experiments.corsi_motion_baseline.canonicalize import load_config
from corsi.experiments.corsi_motion_baseline.dataset import (
    CorsiMotionCanonicalDataset,
    collate_motion_prediction_batch,
)
from corsi.experiments.corsi_motion_baseline.model import build_model
from corsi.experiments.corsi_motion_baseline.train import _denormalize_joints, resolve_device


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


@torch.no_grad()
def export_states(
    *,
    config: dict,
    checkpoint: str | Path,
    split: str,
    output_path: str | Path,
    batch_size: int = 16,
    device_name: str = "auto",
) -> dict:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("State export requires h5py for HDF5 output") from exc

    manifest_path = Path(str(config["canonical_root"])) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = CorsiMotionCanonicalDataset(manifest_path, split=split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_motion_prediction_batch,
    )
    device = resolve_device(device_name)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    seed = int(payload.get("seed", -1))
    model = build_model("visual_joint", joint_dim=int(manifest["joint_dim"]), hidden_dim=128).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    rows = {
        "seq_id": [],
        "split": [],
        "seed": [],
        "length": [],
        "timestep": [],
        "source_frame_index": [],
        "source_timestamp": [],
        "rank": [],
        "block_id": [],
        "block_xy": [],
        "segment_progress": [],
        "boundary": [],
        "joint_normalized": [],
        "joint_physical": [],
        "prediction_normalized": [],
        "prediction_physical": [],
        "target_normalized": [],
        "target_physical": [],
        "visual_feature": [],
        "joint_feature": [],
        "fused_feature": [],
        "h_t": [],
        "c_t": [],
        "input_gate": [],
        "forget_gate": [],
        "candidate": [],
        "output_gate": [],
    }
    for batch in loader:
        moved = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        outputs = model(
            images=moved["images"],
            joints=moved["joints"],
            valid_mask=moved["valid_mask"],
            return_traces=True,
        )
        traces = outputs["traces"]  # type: ignore[index]
        pred = outputs["pred_joints_next"]  # type: ignore[index]
        pred_phys = _denormalize_joints(pred, manifest)
        target_phys = _denormalize_joints(moved["targets_next"], manifest)
        joint_phys = _denormalize_joints(moved["joints"], manifest)
        for batch_index, seq_id in enumerate(batch["seq_id"]):
            valid_steps = torch.nonzero(moved["valid_mask"][batch_index], as_tuple=False).flatten()
            for step_tensor in valid_steps:
                step = int(step_tensor.item())
                rows["seq_id"].append(str(seq_id).encode("utf-8"))
                rows["split"].append(split.encode("utf-8"))
                rows["seed"].append(seed)
                rows["length"].append(int(moved["length"][batch_index].item()))
                rows["timestep"].append(step)
                rows["source_frame_index"].append(int(moved["source_frame_index"][batch_index, step].item()))
                rows["source_timestamp"].append(float(moved["source_timestamp"][batch_index, step].item()))
                rows["rank"].append(int(moved["rank"][batch_index, step].item()))
                rows["block_id"].append(int(moved["block_id"][batch_index, step].item()))
                rows["block_xy"].append(_as_numpy(moved["block_xy"][batch_index, step]))
                rows["segment_progress"].append(float(moved["segment_progress"][batch_index, step].item()))
                rows["boundary"].append(bool(moved["boundary"][batch_index, step].item()))
                rows["joint_normalized"].append(_as_numpy(moved["joints"][batch_index, step]))
                rows["joint_physical"].append(_as_numpy(joint_phys[batch_index, step]))
                rows["prediction_normalized"].append(_as_numpy(pred[batch_index, step]))
                rows["prediction_physical"].append(_as_numpy(pred_phys[batch_index, step]))
                rows["target_normalized"].append(_as_numpy(moved["targets_next"][batch_index, step]))
                rows["target_physical"].append(_as_numpy(target_phys[batch_index, step]))
                rows["visual_feature"].append(_as_numpy(outputs["visual_feature"][batch_index, step]))  # type: ignore[index]
                rows["joint_feature"].append(_as_numpy(outputs["joint_feature"][batch_index, step]))  # type: ignore[index]
                rows["fused_feature"].append(_as_numpy(outputs["fused_feature"][batch_index, step]))  # type: ignore[index]
                for key in ["h_t", "c_t", "input_gate", "forget_gate", "candidate", "output_gate"]:
                    rows[key].append(_as_numpy(traces[key][batch_index, step]))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        for key, values in rows.items():
            handle.create_dataset(key, data=np.asarray(values))
        handle.attrs["canonical_fingerprint"] = manifest["canonical_fingerprint"]
        handle.attrs["checkpoint"] = str(checkpoint)
        handle.attrs["split"] = split
        handle.attrs["seed"] = seed
    metadata = {
        "path": str(output_path),
        "format": "hdf5",
        "rows": int(len(rows["seq_id"])),
        "split": split,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "canonical_fingerprint": manifest["canonical_fingerprint"],
        "fields": sorted(rows.keys()),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Corsi motion hidden-state traces.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
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
