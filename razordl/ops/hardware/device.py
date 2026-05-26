"""Generic device detection interface.

Delegates to backend-specific modules (cuda, mps, xpu, rocm...).
Adding a new hardware backend only requires a new module + a line here.
"""

import torch

import razordl.ops.hardware.cuda as cuda


def get_available_device() -> str:
    """Return the best available device type: 'cuda', 'mps', or 'cpu'."""
    if cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_count() -> int:
    """Return the number of available accelerators (GPUs, etc.)."""
    device = get_available_device()
    if device == "cuda":
        return cuda.get_device_count()
    if device == "mps":
        return 1  # Apple Silicon has at most 1 GPU per process
    return 0


def check_device_compatibility() -> None:
    """Raise RuntimeError with guidance if the accelerator is not usable.

    Checks CUDA first (most common), then MPS.  Extend here when adding
    new backends (XPU, ROCm, etc.).
    """
    device = get_available_device()
    if device == "cuda":
        cuda.check_compatibility()
        return
    if device == "mps":
        # MPS is available; nothing extra to check for now
        return
    # CPU fallback -- training will fail later with a clearer error,
    # but we can warn here if desired.
    index_url = cuda.get_recommended_torch_index()
    raise RuntimeError(
        "No GPU accelerator detected (CUDA or MPS).\n"
        "RazorDL training requires at least one NVIDIA or Apple Silicon GPU.\n\n"
        "If you have an NVIDIA GPU:\n"
        "  1. Check driver:  nvidia-smi\n"
        "  2. Reinstall PyTorch with CUDA:\n"
        f"     pip install torch --index-url {index_url}\n\n"
        "See: https://pytorch.org/get-started/locally/"
    )
