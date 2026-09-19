"""Standalone save_fsdp2 must write even when no rank context exists.

The rank probe (Ray context -> RANK env -> process group) can all be absent
in a standalone single-process run; before the fallback the gates kept
``rank == -1`` and every ``if rank == 0`` check skipped, so the save
silently wrote nothing.
"""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from razordl.ops.parallel.fsdp2 import save_fsdp2


def _tiny_model():
    cfg = Qwen2Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(cfg)


def test_save_fsdp2_writes_without_rank_context(monkeypatch, tmp_path):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    model = _tiny_model()
    save_fsdp2(model, str(tmp_path))

    from safetensors import safe_open

    with safe_open(str(tmp_path / "model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    assert any("embed_tokens" in k for k in keys)
