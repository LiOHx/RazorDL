"""The vllm-metal (MLX) adaptation guards must be inert off Apple Silicon.

All new logic in vllm_rollout.py sits behind import-detection guards that are
False wherever ``vllm_metal`` / ``mlx`` are not installed (every CUDA box).
These tests pin that inertness and the one widened line's equivalence.
"""

import torch
from safetensors.torch import load_file

from razordl.ops.model import vllm_rollout


def test_metal_detection_is_false_without_the_plugin():
    assert vllm_rollout._is_vllm_metal_backend() is False


def test_mlx_model_detection_is_false_without_mlx():
    assert vllm_rollout._is_mlx_model(object()) is False


def test_update_weights_full_keeps_cpu_tensors_passthrough():
    """`.cpu() if tensor.device.type != "cpu"` must keep CPU tensors untouched
    (the previous `is_cuda` test was equivalent for cuda/cpu; this pins the
    cpu case) and route the file through collective_rpc."""
    sent = {"w": torch.ones(2, 2)}
    received = {}

    class _FakeEngine:
        def collective_rpc(self, fn, kwargs):
            received.update(load_file(kwargs["file_path"]))

    rollout = vllm_rollout.GRPOVLLMRollout.__new__(vllm_rollout.GRPOVLLMRollout)
    rollout.engine = _FakeEngine()
    rollout._update_weights_full(iter(sent.items()))
    assert torch.equal(received["w"], sent["w"])
