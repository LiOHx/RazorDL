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


_bf16_supported = None


def supports_native_bf16() -> bool:
    """Whether bf16 executes on this MPS build (probed once, then cached).

    Older PyTorch/macOS had no bf16 on MPS at all; current builds run it on
    the GPU proper.  Measured on an M-series Mac: bf16 matmul throughput
    equals fp16 (both ~12.8 TFLOPs at 2048^3, fp32 measures 3.2), so this is
    hardware bf16, not software emulation.  ``resolve_precision`` runs once
    per training step, hence the cache.
    """
    global _bf16_supported
    if _bf16_supported is None:
        _bf16_supported = _probe_bf16()
    return _bf16_supported


def _probe_bf16() -> bool:
    """Run one tiny bf16 matmul; older builds raise instead of running it."""
    import torch

    try:
        a = torch.ones(8, 8, dtype=torch.bfloat16, device="mps")
        (a @ a).sum().item()
        return True
    except Exception:
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
        "native_bf16": supports_native_bf16(),
        "flash_attention_2": False,
    }


def check_compatibility() -> None:
    """MPS is available; nothing extra to check for now."""
