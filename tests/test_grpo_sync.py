"""Pins from the adversarial review of the Mac/MPS GRPO sync (342f56d..HEAD).

Each test closes a gap the reviewers confirmed with reproduced evidence:
the _iter_merged_full_weights HF-key bug (high), the disable_adapter
reference equivalence, the MPS GradScaler init_scale, and the micro-batch
knob plumbing.  All CPU-runnable; peft is required and installed.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM


def _tiny_lm():
    cfg = Qwen2Config(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=64,
        max_position_embeddings=32, tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(cfg)


def _lora_wrap(model):
    peft = pytest.importorskip("peft")
    return peft.get_peft_model(
        model,
        peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                        task_type="CAUSAL_LM"),
    )


class _StubGroup:
    """Not a ParallelModelGroup subclass on purpose -- the scaler test only
    needs _build_grad_scaler, which touches no other method.  Using __new__
    on the ABC raises; a plain object with the right attributes is enough."""


def test_iter_merged_full_weights_emits_hf_key_names():
    """Keys must equal the unwrapped model's HF names exactly.  The first
    version left peft's ``base_layer`` wrapper infix in place, so
    vllm-metal's HF-named loader would have silently kept stale weights for
    exactly the LoRA-adjacent layers."""
    from razordl.presets.grpo.workgroup import _iter_merged_full_weights

    expected = set(_tiny_lm().state_dict().keys())
    merged = _lora_wrap(_tiny_lm())
    out = dict(_iter_merged_full_weights(merged))
    assert out, "iterator produced nothing"
    assert set(out.keys()) == expected
    assert all("lora_" not in k for k in out)
    assert all("base_layer" not in k for k in out)


def test_disable_adapter_reference_matches_fresh_base():
    """GRPO's LoRA reference path: logp under disable_adapter must equal a
    separately loaded frozen base -- the k1 KL term is computed against it."""
    from razordl.ops.model.per_token_logp import compute_per_token_log_probs

    torch.manual_seed(0)
    ids = torch.randint(0, 64, (2, 8))
    mask = torch.ones_like(ids)

    torch.manual_seed(0)
    base = _tiny_lm().eval()
    torch.manual_seed(0)  # identical base weights -- the comparison is
    merged = _lora_wrap(_tiny_lm()).eval()  # disable_adapter vs fresh base
    with torch.no_grad():
        ref = compute_per_token_log_probs(base, ids, mask, no_grad=True)
        with merged.disable_adapter():
            via_policy = compute_per_token_log_probs(merged, ids, mask, no_grad=True)
    assert torch.equal(ref, via_policy)


def test_mps_grad_scaler_uses_measured_init_scale(monkeypatch):
    """fp16 on MPS must construct GradScaler('mps', init_scale=32) -- measured
    on Mac: 128 overflows Qwen3.5 hybrid-layer backward at step 5.  CUDA keeps
    the default init_scale (no kwargs)."""
    from razordl.core.engine.common import modelgroup as mg
    from razordl.ops.hardware import device as hw_device

    calls = []

    def fake_scaler(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(hw_device, "get_available_device", lambda: "mps")
    monkeypatch.setattr(torch.amp, "GradScaler", fake_scaler)

    stub = _StubGroup()
    stub.local_rank = 1  # silence the info log
    stub.model_group_name = "policy_model_group"
    stub.model_group_config = SimpleNamespace(
        model_config=SimpleNamespace(precision="fp16")
    )
    mg.ParallelModelGroup._build_grad_scaler(stub)
    assert len(calls) == 1
    assert calls[0][0] == ("mps",)
    assert calls[0][1].get("init_scale") == 32


def test_loss_micro_batch_size_config_plumbing():
    from razordl.presets.grpo.config import GRPOConfig

    cfg = GRPOConfig.from_flat_dict(
        {"model": "x", "data_path": "./d", "loss_micro_batch_size": 3}
    )
    assert cfg.data_config.loss_micro_batch_size == 3
    default = GRPOConfig.from_flat_dict({"model": "x", "data_path": "./d"})
    assert default.data_config.loss_micro_batch_size == 0
