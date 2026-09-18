"""FSDP2 roots carry `_is_fsdp_managed_module`, which transformers' generate
uses to turn on `synced_gpus` (transformers>=5 checks that attribute for
FSDP2). `fully_shard` is stubbed: one GPU / CPU cannot host a real mesh.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.distributed.fsdp import is_fsdp_managed_module

from razordl.ops.parallel import fsdp2


@pytest.fixture
def stub_fully_shard(monkeypatch):
    wrapped = []
    monkeypatch.setattr(fsdp2, "fully_shard", lambda module, **kw: wrapped.append(module))
    monkeypatch.setattr(fsdp2, "maybe_patch_fsdp_module", lambda module: nullcontext())
    monkeypatch.setattr(fsdp2, "_reshard_after_forward", lambda mesh: True)
    monkeypatch.setattr(fsdp2, "get_shard_placement_fn", lambda fsdp_size: None)
    monkeypatch.setattr(fsdp2, "fsdp2_load_full_state_dict", lambda *a, **k: None)
    return wrapped


def _tiny_model():
    cfg = Qwen2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, vocab_size=64, max_position_embeddings=32)
    return Qwen2ForCausalLM(cfg)


def test_plain_root_is_marked(stub_fully_shard):
    model = _tiny_model()
    assert not is_fsdp_managed_module(model)
    fsdp2.model_to_fsdp2(model, device_mesh=SimpleNamespace(shape=(1,)), mp_policy=None)
    assert is_fsdp_managed_module(model)
    assert stub_fully_shard[-1] is model  # root wrapped last


def test_peft_root_and_its_base_model_are_marked(stub_fully_shard):
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(_tiny_model(), peft.LoraConfig(r=2, target_modules=["q_proj"]))
    fsdp2.model_to_fsdp2_with_lora(model, device_mesh=SimpleNamespace(shape=(1,)), mp_policy=None)
    assert is_fsdp_managed_module(model)
    # PeftModel.generate -> LoraModel.__getattr__ -> HF model.generate: that `self` must carry the flag
    assert is_fsdp_managed_module(model.get_base_model())
