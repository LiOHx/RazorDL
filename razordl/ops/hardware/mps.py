"""Apple Silicon (MPS) backend — capability probes only.

Policy built on these probes lives in ``precision.py`` (backend-independent
by design); nothing here decides *what* to run, only *what the hardware is*.
"""


def is_available() -> bool:
    """MPS is available (Apple Silicon GPU; full op coverage needs macOS 13+)."""
    import torch

    return torch.backends.mps.is_available()


def get_device_count() -> int:
    """Apple Silicon has at most 1 GPU per process."""
    return 1


def supports_native_bf16() -> bool:
    """MPS has no bf16 acceleration; the precision policy uses fp16 instead."""
    return False


def supports_fp16() -> bool:
    """MPS executes fp16 natively (unified memory, GPU fp16 ALU paths)."""
    return is_available()


def supports_flash_attention_2() -> bool:
    """flash-attn is a CUDA-only package."""
    return False


def get_device_memory_gb() -> float:
    """Recommended working set size in GB; 0.0 when the probe is unavailable."""
    import torch

    probe = getattr(torch.mps, "recommended_max_working_set_size", None)
    if probe is None:
        return 0.0
    try:
        return round(probe() / (1024**3), 1)
    except Exception:
        return 0.0


def describe() -> dict:
    """Flat capability dict for the startup hardware log."""
    return {
        "device": "mps",
        "name": "Apple Silicon (MPS)",
        "count": get_device_count(),
        "compute_capability": None,
        "memory_gb": get_device_memory_gb(),
        "native_bf16": False,
        "flash_attention_2": False,
    }


def check_compatibility() -> None:
    """MPS is available; nothing extra to check for now."""
