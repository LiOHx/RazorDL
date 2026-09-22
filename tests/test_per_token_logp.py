"""Per-token logp: fake-LM wiring, gradient parity, and a real-model GPU pin.

The real-model test (Qwen3.5-0.8B) pins the load-bearing assumptions of the
streaming entry: ``hidden_states[-1]`` is post final norm, and the operator
path equals the full-logits composition on the same forward.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from razordl.ops.model.per_token_logp import compute_per_token_log_probs

VOCAB, DIM = 29, 8


class _Out:
    def __init__(self, logits, hidden_states=None):
        self.logits = logits
        self.hidden_states = hidden_states


class _FakeLM(torch.nn.Module):
    """Minimal causal LM exposing the HF surface the function relies on:
    output_hidden_states / logits_to_keep kwargs and get_output_embeddings.
    (Identity final norm -- immaterial for the math.)"""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = torch.nn.Embedding(VOCAB, DIM)
        self.head = torch.nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, logits_to_keep=0):
        hidden = self.emb(input_ids)
        logits = self.head(hidden)
        if logits_to_keep:
            logits = logits[:, -int(logits_to_keep):]
        return _Out(logits, (hidden,) if output_hidden_states else None)

    def get_output_embeddings(self):
        return self.head


def _ids(seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, VOCAB, (2, 6))


def _ref_from_fake(fake, ids, temperature):
    hidden = fake.emb(ids)
    logits = fake.head(hidden)
    if temperature != 1.0:
        logits = logits / temperature
    return torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(
        2, ids[:, 1:, None]
    ).squeeze(-1)


def test_temperature_divides_logits_before_log_softmax():
    ids = _ids()
    fake = _FakeLM(0)

    lp = compute_per_token_log_probs(fake, ids, torch.ones_like(ids), temperature=0.5)
    assert torch.allclose(lp, _ref_from_fake(fake, ids, 0.5), rtol=1e-5, atol=1e-6)

    # T=1 is the plain log-softmax; tempered values differ, so the rollout
    # temperature is not silently ignored.
    plain = compute_per_token_log_probs(fake, ids, torch.ones_like(ids))
    assert not torch.allclose(lp, plain)


def test_return_shape_and_alignment():
    ids = _ids()
    fake = _FakeLM(1)
    lp = compute_per_token_log_probs(fake, ids, torch.ones_like(ids))
    assert lp.shape == (2, 5)
    assert torch.allclose(lp, _ref_from_fake(fake, ids, 1.0), rtol=1e-5, atol=1e-6)


def test_gradients_match_full_logits_composition():
    """End-to-end through the fake's parameters: embedding + lm_head grads
    must equal the pre-streaming composition (pins the rolled-target trick —
    the discarded column must not leak gradient)."""
    ids = _ids(seed=3)

    fake = _FakeLM(7)
    lp = compute_per_token_log_probs(fake, ids, torch.ones_like(ids), temperature=0.9)
    lp.sum().backward()

    fake2 = _FakeLM(7)
    ref = _ref_from_fake(fake2, ids, 0.9)
    ref.sum().backward()

    assert torch.allclose(fake.emb.weight.grad, fake2.emb.weight.grad, rtol=1e-4, atol=1e-5)
    assert torch.allclose(fake.head.weight.grad, fake2.head.weight.grad, rtol=1e-4, atol=1e-4)


def test_no_grad_path_does_not_build_a_graph():
    ids = _ids()
    fake = _FakeLM(2)
    lp = compute_per_token_log_probs(fake, ids, torch.ones_like(ids), no_grad=True)
    assert lp.grad_fn is None


QWEN = "/home/lioh_wsl/Models/Qwen3.5-0.8B"


@pytest.mark.skipif(not os.path.isdir(QWEN), reason="local Qwen3.5 checkpoint not present")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real-model parity probe needs a GPU")
@pytest.mark.parametrize("dtype_ctx", ["fp32", "bf16_autocast"])
@pytest.mark.parametrize("temperature", [1.0, 0.9])
def test_real_model_matches_full_logits_composition(dtype_ctx, temperature):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).cuda().eval()
    enc = tok(["2+3=?", "The capital of France is"], return_tensors="pt",
              padding=True, padding_side="left")
    ids, mask = enc.input_ids.cuda(), enc.attention_mask.cuda()

    def run(fn):
        if dtype_ctx == "bf16_autocast":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return fn()
        return fn()

    with torch.no_grad():
        lp = run(lambda: compute_per_token_log_probs(
            model, ids, mask, temperature=temperature
        ))
        ref = run(lambda: _ref_from_model(model, ids, mask, temperature))
    assert lp.shape == ref.shape
    if dtype_ctx == "fp32":
        tol = dict(rtol=1e-5, atol=1e-5)
    else:
        # bf16 + T != 1: the operator scales hidden before the tiles, the
        # reference scales the logits -- mathematically identical, but the
        # division lands at different bf16 roundings (ulp at |logp|~12 is
        # 0.06, and softmax accumulates a few).  Still far below a sign
        # error (~2x|logp|) or any gross bug.
        tol = dict(rtol=5e-2, atol=5e-2)
    assert torch.allclose(lp.float(), ref, **tol)


def _ref_from_model(model, ids, mask, temperature):
    out = model(input_ids=ids, attention_mask=mask)
    logits = out.logits[:, :-1]
    if temperature != 1.0:
        logits = logits / temperature
    return torch.log_softmax(logits.float(), dim=-1).gather(2, ids[:, 1:, None]).squeeze(-1)
