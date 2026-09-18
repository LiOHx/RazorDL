"""Adapter checkpoint round trip, including modules_to_save.

A ``"lora_" in key`` filter once dropped the fully-trained ``lm_head`` /
``embed_tokens`` copies from ``adapter_model.safetensors``; with
``save_full_model=False`` nothing else was written, so resume and export
silently reloaded the untrained base head.
"""

import os

import pytest
import torch

from razordl.ops.model.peft import (
    adapter_state_dict_for_saving,
    get_adapter_state_dict,
    set_adapter_state_dict,
)


def _tiny_lora_model(modules_to_save=("lm_head",)):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
    )
    return get_peft_model(
        Qwen2ForCausalLM(cfg),
        LoraConfig(r=4, target_modules=["q_proj"], modules_to_save=list(modules_to_save)),
    )


def test_saving_filter_keeps_modules_to_save_in_peft_layout():
    model = _tiny_lora_model()
    saved = adapter_state_dict_for_saving(model.state_dict())

    assert "lm_head.weight" in saved
    assert "model.layers.0.self_attn.q_proj.lora_A.weight" in saved
    assert not any("modules_to_save" in k or ".default" in k for k in saved)
    # Base weights stay out: this is an adapter file, not a full model.
    assert "model.embed_tokens.weight" not in saved


@pytest.mark.parametrize("peft_loader_available", [True, False])
def test_adapter_round_trip_restores_lm_head(tmp_path, monkeypatch, peft_loader_available):
    from razordl.core.engine.common.parallel_state import save_full_or_adapter_model

    model = _tiny_lora_model()
    key = "base_model.model.lm_head.modules_to_save.default.weight"
    trained = model.state_dict()[key].clone()

    save_full_or_adapter_model(model, str(tmp_path), save_lora_separately=True, save_full_model=False)
    adapter_file = tmp_path / "adapter" / "adapter_model.safetensors"
    assert adapter_file.exists()
    assert not (tmp_path / "model.safetensors").exists()

    if not peft_loader_available:
        # Exercise the plain load_state_dict fallback too.
        import peft.utils

        def _boom(*a, **k):
            raise RuntimeError("simulated peft loader failure")

        monkeypatch.setattr(peft.utils, "set_peft_model_state_dict", _boom)

    with torch.no_grad():
        model.state_dict()[key].zero_()
    missing, unexpected = set_adapter_state_dict(model, get_adapter_state_dict(str(adapter_file)))

    assert unexpected == []
    assert torch.equal(model.state_dict()[key], trained)
