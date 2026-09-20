"""Generic device detection interface.

Delegates to backend-specific modules (cuda, mps, xpu, rocm...).
Adding a new hardware backend only requires a new module + entries here.
This file is a PURE dispatcher: per-device knowledge (counts, probes,
describe fields) lives only in the backend file — see this directory's
CLAUDE.md hard rule.
"""

import importlib

import torch


def _backend(name: str):
    """Lazily import a backend module (see this directory's CLAUDE.md hard rule).

    Resolved relative to ``__package__`` so the same file works inside the
    ``razordl`` package and in a full-mode export, where it lives at
    ``ops.hardware`` and ``razordl`` is not installed.
    """
    return importlib.import_module(f"{__package__}.{name}")


def get_available_device() -> str:
    """Return the best available device type: 'cuda', 'mps', or 'cpu'."""
    if _backend("cuda").is_available():
        return "cuda"
    if _backend("mps").is_available():
        return "mps"
    return "cpu"


def get_torch_device():
    """Return the torch device namespace (``torch.cuda`` / ``torch.mps`` ...).

    Falls back to ``torch.cuda`` when torch has no namespace for the active
    device type, matching the historical behaviour of the FSDP2 helpers.
    """
    name = get_available_device()
    return getattr(torch, name, torch.cuda)


def get_device_id() -> int:
    """Return the index of the current accelerator (0 for mps / cpu)."""
    if get_available_device() == "cuda":
        return _backend("cuda").current_device()
    return 0


def get_device_count() -> int:
    """Return the number of available accelerators (GPUs, etc.)."""
    device = get_available_device()
    if device == "cuda":
        return _backend("cuda").get_device_count()
    if device == "mps":
        return _backend("mps").get_device_count()
    return 0


def supports_native_bf16() -> bool:
    """Return True if the active accelerator has native (non-emulated) bf16."""
    device = get_available_device()
    if device == "cuda":
        return _backend("cuda").supports_native_bf16()
    if device == "mps":
        return _backend("mps").supports_native_bf16()
    return False  # CPU bf16 is emulated on most x86


def supports_flash_attention_2() -> bool:
    """Return True if the active accelerator can run flash-attn 2 kernels."""
    device = get_available_device()
    if device == "cuda":
        return _backend("cuda").supports_flash_attention_2()
    if device == "mps":
        return _backend("mps").supports_flash_attention_2()
    return False


def describe() -> dict:
    """Return a flat dict of accelerator capabilities for startup logging."""
    device = get_available_device()
    if device == "cuda":
        return _backend("cuda").describe()
    if device == "mps":
        return _backend("mps").describe()
    return {
        "device": "cpu",
        "name": None,
        "count": 0,
        "compute_capability": None,
        "memory_gb": 0.0,
        "native_bf16": False,
        "flash_attention_2": False,
    }


def check_device_compatibility() -> None:
    """Raise RuntimeError with guidance if the accelerator is not usable."""
    device = get_available_device()
    if device == "cuda":
        _backend("cuda").check_compatibility()
        return
    if device == "mps":
        _backend("mps").check_compatibility()
        return
    index_url = _backend("cuda").get_recommended_torch_index()
    raise RuntimeError(
        "No GPU accelerator detected (CUDA or MPS).\n"
        "RazorDL training requires at least one NVIDIA or Apple Silicon GPU.\n\n"
        "If you have an NVIDIA GPU:\n"
        "  1. Check driver:  nvidia-smi\n"
        "  2. Reinstall PyTorch with CUDA:\n"
        f"     pip install torch --index-url {index_url}\n\n"
        "See: https://pytorch.org/get-started/locally/"
    )
