"""vLLM engine init: absent package -> HF fallback; any other failure -> fatal.

Before this, every exception in `_init_vllm_engine` (OOM, bf16 on a Turing
GPU, a bad `max_model_len`) was logged at WARNING and the run silently
continued on HF generate, 20x slower. Only the ImportError branch keeps the
fallback; `SKIP_VLLM=1` is the explicit way to ask for it.

CPU-only: the vLLM module is replaced by a stub, so this covers the control
flow, not a real engine.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from razordl.presets.grpo.workgroup import GRPOPolicyModelGroup
from razordl.presets.opd.workgroup import OPDPolicyModelGroup

VLLM_MOD = "razordl.ops.model.vllm_rollout"


def _stub_model_group(cls):
    adapter = SimpleNamespace(use_adapter=False, lora_r=8)
    model_cfg = SimpleNamespace(model_path="/nonexistent", precision="fp32", adapter_config=adapter)
    wg = cls.__new__(cls)
    wg._vllm_engine = None
    wg.model_group_config = SimpleNamespace(
        model_config=model_cfg, processor_config=SimpleNamespace(max_length=64)
    )
    wg.config = SimpleNamespace(
        data_config=SimpleNamespace(max_completion_length=16),
        trainer_config=SimpleNamespace(seed=0),
    )
    return wg


def _install_failing_vllm(monkeypatch, exc):
    mod = types.ModuleType(VLLM_MOD)

    class _Rollout:
        def __init__(self, *a, **k):
            raise exc

    mod.GRPOVLLMRollout = _Rollout
    mod.InferenceConfig = lambda **k: SimpleNamespace(**k)
    mod.SamplingParamsConfig = lambda **k: SimpleNamespace(**k)
    monkeypatch.setitem(sys.modules, VLLM_MOD, mod)


@pytest.mark.parametrize("cls", [GRPOPolicyModelGroup, OPDPolicyModelGroup])
def test_missing_vllm_keeps_hf_fallback(monkeypatch, cls):
    monkeypatch.delenv("SKIP_VLLM", raising=False)
    monkeypatch.setitem(sys.modules, VLLM_MOD, None)  # `import` raises ImportError
    wg = _stub_model_group(cls)
    wg._init_vllm_engine()
    assert wg._vllm_engine is None


@pytest.mark.parametrize("cls", [GRPOPolicyModelGroup, OPDPolicyModelGroup])
def test_engine_failure_is_fatal(monkeypatch, cls):
    monkeypatch.delenv("SKIP_VLLM", raising=False)
    _install_failing_vllm(monkeypatch, RuntimeError("CUDA out of memory"))
    wg = _stub_model_group(cls)
    with pytest.raises(RuntimeError, match="SKIP_VLLM=1") as info:
        wg._init_vllm_engine()
    assert "CUDA out of memory" in str(info.value)
    assert isinstance(info.value.__cause__, RuntimeError)


@pytest.mark.parametrize("cls", [GRPOPolicyModelGroup, OPDPolicyModelGroup])
def test_skip_vllm_env_bypasses_engine(monkeypatch, cls):
    monkeypatch.setenv("SKIP_VLLM", "1")
    _install_failing_vllm(monkeypatch, RuntimeError("must not be reached"))
    wg = _stub_model_group(cls)
    wg._init_vllm_engine()
    assert wg._vllm_engine is None
