"""CUDA-specific hardware detection."""

import subprocess
import torch


def is_available() -> bool:
    """Return True if CUDA is available and functional."""
    return torch.cuda.is_available()


def get_torch_cuda_version() -> str | None:
    """Return the CUDA version PyTorch was compiled with, or None for CPU build."""
    return torch.version.cuda


def get_driver_cuda_version() -> str | None:
    """Return the CUDA version supported by the NVIDIA driver."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if "CUDA Version:" in line:
                # Line looks like: "| NVIDIA-SMI 545.23.08    Driver Version: 545.23.08    CUDA Version: 12.3 |"
                parts = line.split("CUDA Version:")
                if len(parts) == 2:
                    return parts[1].strip().rstrip(" |")
        return None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def get_device_count() -> int:
    """Return the number of CUDA GPUs."""
    return torch.cuda.device_count() if is_available() else 0


# --- Runtime capability probes -------------------------------------------------
#
# Compute capability ("SM version") is the stable, exact criterion for what a
# NVIDIA GPU can do natively.  We deliberately do NOT use
# ``torch.cuda.is_bf16_supported()``: it defaults to ``including_emulation=True``
# and returns True on Turing (sm_75), where bf16 is emulated in software -- so it
# cannot answer "is bf16 *native* here?", which is what callers actually mean.
# Whether emulated bf16 is worth running is a policy question, settled with
# measurements in ``precision.py``; this file only reports the capability.

_BF16_MIN_SM = (8, 0)   # Ampere: first arch with native bf16 tensor cores
_FA2_MIN_SM = (8, 0)    # flash-attn 2 requires Ampere or newer


def get_device_capability(index: int = 0) -> tuple[int, int] | None:
    """Return the (major, minor) compute capability of GPU *index*, or None."""
    if not is_available():
        return None
    try:
        return torch.cuda.get_device_capability(index)
    except (RuntimeError, AssertionError):
        return None


def get_min_device_capability() -> tuple[int, int] | None:
    """Return the lowest compute capability across all visible GPUs.

    Heterogeneous multi-GPU nodes must be governed by their weakest device --
    reporting the strongest would let the framework pick a dtype or kernel that
    some ranks cannot run natively.
    """
    count = get_device_count()
    if count == 0:
        return None
    caps = [get_device_capability(i) for i in range(count)]
    caps = [c for c in caps if c is not None]
    return min(caps) if caps else None


def supports_native_bf16() -> bool:
    """Return True if every visible GPU has native (non-emulated) bf16."""
    cap = get_min_device_capability()
    return cap is not None and cap >= _BF16_MIN_SM


def supports_flash_attention_2() -> bool:
    """Return True if every visible GPU can run flash-attn 2 kernels."""
    cap = get_min_device_capability()
    return cap is not None and cap >= _FA2_MIN_SM


def get_device_memory_gb(index: int = 0) -> float:
    """Return total memory of GPU *index* in GiB, or 0.0 if unavailable."""
    if not is_available():
        return 0.0
    try:
        return torch.cuda.get_device_properties(index).total_memory / 1024**3
    except (RuntimeError, AssertionError):
        return 0.0


def describe() -> dict:
    """Return a flat dict of capabilities for startup logging."""
    cap = get_min_device_capability()
    return {
        "device": "cuda",
        "name": torch.cuda.get_device_name(0) if is_available() else None,
        "count": get_device_count(),
        "compute_capability": f"sm_{cap[0]}{cap[1]}" if cap else None,
        "memory_gb": round(get_device_memory_gb(), 1),
        "native_bf16": supports_native_bf16(),
        "flash_attention_2": supports_flash_attention_2(),
    }


def get_recommended_torch_index() -> str:
    """Return the recommended PyTorch index-url based on driver or torch CUDA version."""
    # Best: nvidia-smi driver CUDA version
    driver_cuda = get_driver_cuda_version()
    if driver_cuda:
        return f"https://download.pytorch.org/whl/cu{driver_cuda.split('.')[0]}0"
    # Fallback: PyTorch's own CUDA version from compiled build
    torch_cuda = get_torch_cuda_version()
    if torch_cuda:
        return f"https://download.pytorch.org/whl/cu{torch_cuda.replace('.', '')}"
    # Last resort
    return "https://download.pytorch.org/whl/cu118"


def check_compatibility() -> None:
    """Raise RuntimeError with clear guidance if CUDA is broken."""
    if is_available():
        return

    torch_cuda = get_torch_cuda_version()
    driver_cuda = get_driver_cuda_version()

    if torch_cuda is not None:
        msg = (
            f"PyTorch was built with CUDA {torch_cuda} but no GPU is accessible.\n\n"
        )
        if driver_cuda:
            index_url = get_recommended_torch_index()
            msg += (
                f"Your NVIDIA driver supports CUDA {driver_cuda}.\n"
                f"Reinstall PyTorch with a matching CUDA version:\n\n"
                f"  pip install torch --index-url {index_url}\n\n"
                f"See: https://pytorch.org/get-started/locally/"
            )
        else:
            msg += (
                "nvidia-smi not found. Either:\n"
                "  1. No NVIDIA driver is installed, or\n"
                "  2. nvidia-smi is not in PATH.\n\n"
                "Install the NVIDIA driver first, then reinstall PyTorch:\n"
                "  https://www.nvidia.com/Download/index.aspx"
            )
        raise RuntimeError(msg)

    # CPU-only PyTorch
    index_url = get_recommended_torch_index()
    raise RuntimeError(
        "PyTorch CPU version is installed but GPU training requires CUDA.\n\n"
        "Install PyTorch with CUDA support:\n"
        f"  pip install torch --index-url {index_url}\n\n"
        "If the URL above does not match your environment, check your CUDA version\n"
        "with 'nvidia-smi' and pick the matching index:\n"
        "  https://pytorch.org/get-started/locally/"
    )
