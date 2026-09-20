"""ParallelModelGroup.get_device() must route through the hardware layer.

Regression guard for the hand-rolled cuda/cpu copy that silently dropped
MPS to CPU.  Device choice is forced by monkeypatching the dispatcher, so
no MPS (or CUDA) hardware is needed.
"""

import pytest
import torch

from razordl.core.engine.common.modelgroup import ParallelModelGroup
from razordl.ops.hardware import device as hw_device


class _StubGroup(ParallelModelGroup):
    """Satisfies the abstract methods; __init__ is skipped entirely."""

    def __init__(self):
        pass

    def build_processor(self):
        pass

    def build_model(self):
        pass

    def save_processor(self, d):
        pass

    def build_optimizer(self):
        pass

    def save_optimizer(self, d):
        pass

    def build_scheduler(self):
        pass

    def save_scheduler(self, d):
        pass

    def save_model(self, d):
        pass

    def save_model_and_processor(self, d):
        pass

    def save_checkpoint(self, d):
        pass


@pytest.mark.parametrize(
    "available,expected_type",
    [("cuda", "cuda"), ("mps", "mps"), ("cpu", "cpu")],
)
def test_get_device_follows_hardware_layer(monkeypatch, available, expected_type):
    monkeypatch.setattr(hw_device, "get_available_device", lambda: available)
    if available == "cuda":
        monkeypatch.setattr(hw_device, "get_device_id", lambda: 0)
        monkeypatch.setattr(torch.cuda, "set_device", lambda idx: None)

    stub = _StubGroup()
    stub.local_rank = 0
    dev = stub.get_device()
    assert dev.type == expected_type
