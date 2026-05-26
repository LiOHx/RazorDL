"""Hardware detection and compatibility utilities.

Supports CUDA today; MPS (Apple Silicon), XPU (Intel), and ROCm (AMD)
can be added as new modules without touching engine code.

Usage:
    from razordl.ops.hardware.device import check_device_compatibility
    check_device_compatibility()  # raises RuntimeError with guidance if broken
"""
