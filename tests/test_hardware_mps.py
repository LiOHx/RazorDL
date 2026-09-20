"""MPS backend + dispatcher behaviour, forced by monkeypatching (no Mac needed)."""

import pytest
import torch

from razordl.ops.hardware import device
from razordl.ops.hardware import mps as mps_backend


@pytest.fixture
def force_mps(monkeypatch):
    monkeypatch.setattr(device._backend("cuda"), "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


@pytest.fixture
def force_cpu(monkeypatch):
    monkeypatch.setattr(device._backend("cuda"), "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def test_available_device_mps(force_mps):
    assert device.get_available_device() == "mps"


def test_mps_count_and_id(force_mps):
    assert mps_backend.get_device_count() == 1
    assert device.get_device_count() == 1
    assert device.get_device_id() == 0


def test_mps_probes(force_mps, monkeypatch):
    # bf16 is a runtime probe on the real hardware; pin the cached value so
    # the assertions are deterministic on any machine.
    monkeypatch.setattr(mps_backend, "_bf16_supported", True)
    assert mps_backend.supports_fp16() is True
    assert device.supports_native_bf16() is True
    assert device.supports_flash_attention_2() is False


def test_mps_probes_without_bf16(force_mps, monkeypatch):
    """Older PyTorch/macOS builds with no bf16 on MPS fall back to fp16."""
    monkeypatch.setattr(mps_backend, "_bf16_supported", False)
    assert device.supports_native_bf16() is False


def test_mps_describe(force_mps, monkeypatch):
    monkeypatch.setattr(mps_backend, "_bf16_supported", True)
    info = device.describe()
    assert info["device"] == "mps"
    assert info["count"] == 1
    assert info["native_bf16"] is True
    assert info["flash_attention_2"] is False
    assert "memory_gb" in info


def test_mps_compatibility_check_passes(force_mps):
    device.check_device_compatibility()  # must not raise


def test_cpu_path_unchanged(force_cpu):
    assert device.get_available_device() == "cpu"
    assert device.get_device_count() == 0
    with pytest.raises(RuntimeError, match="No GPU accelerator detected"):
        device.check_device_compatibility()


def test_attn_fallback_sdpa_on_mps_eager_on_cpu(monkeypatch):
    """SDPA works on MPS; keep the conservative eager fallback only on CPU."""
    from razordl.ops.hardware import device as hw_device
    from razordl.ops.model import huggingface

    monkeypatch.delenv("RAZORDL_DETERMINISTIC", raising=False)
    monkeypatch.setattr(hw_device, "get_available_device", lambda: "mps")
    assert huggingface.resolve_attn_implementation() == "sdpa"
    monkeypatch.setattr(hw_device, "get_available_device", lambda: "cpu")
    assert huggingface.resolve_attn_implementation() == "eager"
