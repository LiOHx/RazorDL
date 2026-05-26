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
