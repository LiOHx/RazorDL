import pytest
import torch

from razordl.core.engine.common.flat_config import build_single_model_config_dict
from razordl.core.engine.common.parallel_backend import DDPBackend, FSDP2Backend, build_parallel_backend
from razordl.core.engine.single_model.config import Config


def _config(parallel_backend="fsdp2", **extra):
    flat = {
        "model": "dummy-model",
        "data_path": "./data",
        "parallel_backend": parallel_backend,
        **extra,
    }
    config_dict = build_single_model_config_dict(
        flat,
        data_config={"train_data_path": "./data", "max_length": 16, "sp_size": 1, "dataset_processor_path": ""},
        model_default="dummy-model",
        processor_max_length=16,
    )
    return Config.from_dict(config_dict)


class DummyModelGroup:
    def __init__(self, config):
        self.config = config
        self.model_group_config = config.worker_group_config.model_group_config
        self.local_rank = 0
        self.device = torch.device("cpu")
        self.is_trainable = True

    def _enable_gradient_checkpointing_after_wrap(self, model):
        self.gradient_checkpointing_called = True


def test_parallel_backend_flat_config_default_and_override():
    assert _config().worker_group_config.model_group_config.model_config.parallel_backend == "fsdp2"
    assert _config("ddp").worker_group_config.model_group_config.model_config.parallel_backend == "ddp"


def test_backend_registry_selects_backend_class():
    fsdp_group = DummyModelGroup(_config("fsdp2"))
    ddp_group = DummyModelGroup(_config("ddp"))

    assert isinstance(build_parallel_backend("fsdp2", fsdp_group), FSDP2Backend)
    assert isinstance(build_parallel_backend("ddp", ddp_group), DDPBackend)


def test_ddp_backend_rejects_fsdp_only_features():
    group = DummyModelGroup(_config("ddp", offload_param=True))
    backend = DDPBackend(group)

    try:
        backend.wrap_model(torch.nn.Linear(2, 2))
    except ValueError as e:
        assert "offload_param" in str(e)
    else:
        raise AssertionError("DDP backend should reject offload_param")


def test_ddp_backend_wraps_plain_model_on_single_process():
    group = DummyModelGroup(_config("ddp"))
    backend = DDPBackend(group)
    model = torch.nn.Linear(2, 2)

    wrapped = backend.wrap_model(model)

    assert wrapped is model
    assert next(wrapped.parameters()).device.type == "cpu"


def test_ddp_backend_accepts_frozen_model():
    group = DummyModelGroup(_config("ddp"))
    group.is_trainable = False
    backend = DDPBackend(group)
    model = torch.nn.Linear(2, 2)
    for param in model.parameters():
        param.requires_grad = False

    wrapped = backend.wrap_model(model)

    assert wrapped is model
    assert all(not param.requires_grad for param in wrapped.parameters())


def test_ddp_backend_unwraps_model_for_inference_methods():
    group = DummyModelGroup(_config("ddp"))
    backend = DDPBackend(group)

    class Raw(torch.nn.Module):
        def forward(self, x):
            return x

        def generate(self):
            return "ok"

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = Raw()

        def forward(self, x):
            return self.module(x)

    wrapped = Wrapper()

    assert not hasattr(wrapped, "generate")
    assert backend.unwrap_for_inference(wrapped).generate() == "ok"


class _FakeFsdpRoot:
    """Stands in for a fully_shard-ed PeftModel (FSDP2 root with a PEFT accessor)."""

    def __init__(self):
        self.calls = []

    def get_base_model(self):
        return self

    def _get_fsdp_state(self):
        root = self

        class _State:
            def _lazy_init(self):
                root.calls.append("lazy_init")

        return _State()

    def unshard(self, async_op=False):
        self.calls.append("unshard")

    def reshard(self):
        self.calls.append("reshard")


def test_fsdp2_generation_context_unshards_peft_root():
    backend = FSDP2Backend.__new__(FSDP2Backend)
    peft_root = _FakeFsdpRoot()
    with backend.generation_context(peft_root):
        assert peft_root.calls == ["lazy_init", "unshard"]
    assert peft_root.calls == ["lazy_init", "unshard", "reshard"]

    plain = _FakeFsdpRoot()
    plain.get_base_model = None  # a plain HF root runs its own forward: nothing to do
    with backend.generation_context(plain):
        pass
    assert plain.calls == []

    class _DdpUnwrapped:  # PeftModel without FSDP2 methods
        def get_base_model(self):
            return self

    with backend.generation_context(_DdpUnwrapped()):
        pass


def test_fsdp2_backend_rejects_non_cuda(monkeypatch):
    """FSDP2's DTensor stack is CUDA-only; off CUDA the failure must be a
    clear config error, not the raw AttributeError from init_device_mesh."""
    from razordl.ops.hardware import device as hw_device

    backend = FSDP2Backend.__new__(FSDP2Backend)
    for fake in ("mps", "cpu"):
        monkeypatch.setattr(hw_device, "get_available_device", lambda dev=fake: dev)
        with pytest.raises(RuntimeError, match="requires CUDA"):
            backend.wrap_model(object())


def test_ray_worker_plan_is_device_aware(monkeypatch):
    from razordl.core.engine.common import main as engine_main
    from razordl.ops.hardware import device as hw_device

    monkeypatch.setattr(hw_device, "get_available_device", lambda: "cuda")
    monkeypatch.setattr(hw_device, "get_device_count", lambda: 8)
    assert engine_main._ray_worker_plan() == (8, True)

    monkeypatch.setattr(hw_device, "get_available_device", lambda: "mps")
    assert engine_main._ray_worker_plan() == (1, False)

    monkeypatch.setattr(hw_device, "get_available_device", lambda: "cpu")
    assert engine_main._ray_worker_plan() == (0, False)
