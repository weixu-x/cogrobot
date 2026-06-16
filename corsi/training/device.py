"""Device selection helpers for PyTorch training scripts."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def detect_torch_devices() -> Dict[str, bool]:
    return {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_built() and torch.backends.mps.is_available(),
        "cpu": True,
    }


def resolve_torch_device(requested: str = "auto") -> Tuple[torch.device, Dict[str, object]]:
    availability = detect_torch_devices()
    requested = requested.lower()

    if requested == "auto":
        if availability["cuda"]:
            resolved = "cuda"
        elif availability["mps"]:
            resolved = "mps"
        else:
            resolved = "cpu"
    else:
        if requested not in {"cpu", "mps", "cuda"}:
            raise ValueError("requested device must be one of: auto, cpu, mps, cuda")
        if requested != "cpu" and not availability[requested]:
            raise RuntimeError(f"Requested device '{requested}' is not available on this machine")
        resolved = requested

    return torch.device(resolved), {
        "requested_device": requested,
        "resolved_device": resolved,
        "availability": availability,
    }
