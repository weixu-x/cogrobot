"""Training and inference entry points for Corsi experiments."""

from .device import detect_torch_devices, resolve_torch_device

__all__ = ["detect_torch_devices", "resolve_torch_device"]
