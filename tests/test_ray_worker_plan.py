"""Ray worker-plan selection (local vs multi-node GPU count)."""

import ray

from razordl.core.engine.common import main as main_module
from razordl.ops.hardware import device as hw_device


def _plan(monkeypatch, *, accelerator, local_count, initialized, cluster_gpus):
    monkeypatch.setattr(hw_device, "get_available_device", lambda: accelerator)
    monkeypatch.setattr(hw_device, "get_device_count", lambda: local_count)
    monkeypatch.setattr(ray, "is_initialized", lambda: initialized)
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"GPU": cluster_gpus})
    return main_module._ray_worker_plan()


def test_cuda_uses_cluster_gpu_count(monkeypatch):
    assert _plan(
        monkeypatch,
        accelerator="cuda",
        local_count=1,
        initialized=True,
        cluster_gpus=8.0,
    ) == (8, True)


def test_cuda_falls_back_to_local_count_when_cluster_reports_zero(monkeypatch):
    assert _plan(
        monkeypatch,
        accelerator="cuda",
        local_count=2,
        initialized=True,
        cluster_gpus=0,
    ) == (2, True)


def test_cuda_without_ray_uses_local_count(monkeypatch):
    assert _plan(
        monkeypatch,
        accelerator="cuda",
        local_count=4,
        initialized=False,
        cluster_gpus=8.0,
    ) == (4, True)


def test_mps_is_single_worker(monkeypatch):
    assert _plan(
        monkeypatch,
        accelerator="mps",
        local_count=0,
        initialized=False,
        cluster_gpus=0,
    ) == (1, False)


def test_cpu_runs_in_driver(monkeypatch):
    assert _plan(
        monkeypatch,
        accelerator="cpu",
        local_count=0,
        initialized=False,
        cluster_gpus=0,
    ) == (0, False)
