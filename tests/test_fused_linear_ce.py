"""Parity matrix for FusedLinearCrossEntropy -- the acceptance gate.

The reference composition below is deliberately independent (full logits
materialized, fp32 CE) so the operator is checked against the plain math,
not against itself.
"""

import pytest
import torch
import torch.nn.functional as F

from razordl.ops.loss.fused_linear_ce import fused_linear_cross_entropy

VOCAB, DIM, B, L = 97, 16, 2, 6


def _ref_nll(hidden, weight, target, *, temperature=1.0, softcap=None, ignore_index=-100):
    logits = F.linear(hidden, weight)
    if temperature != 1.0:
        logits = logits / temperature
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        target.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view(target.shape)


def _case(device="cpu", dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    hidden = torch.randn(B, L, DIM, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(VOCAB, DIM, device=device, dtype=dtype, requires_grad=True)
    target = torch.randint(0, VOCAB, (B, L), device=device)
    return hidden, weight, target


@pytest.mark.parametrize("tile_size", [1, 3, 1024])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("softcap", [None, 20.0])
def test_forward_matches_reference(tile_size, temperature, softcap):
    hidden, weight, target = _case()
    out = fused_linear_cross_entropy(hidden, weight, target, tile_size,
                                     temperature=temperature, softcap=softcap)
    ref = _ref_nll(hidden.detach(), weight.detach(), target,
                   temperature=temperature, softcap=softcap)
    assert out.dtype is torch.float32
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("tile_size", [1, 3, 1024])
def test_gradients_match_reference(tile_size):
    hidden, weight, target = _case()
    out = fused_linear_cross_entropy(hidden, weight, target, tile_size)
    out.sum().backward()
    g_h, g_w = hidden.grad.clone(), weight.grad.clone()

    hidden2, weight2, target2 = _case()
    ref = _ref_nll(hidden2, weight2, target2)
    ref.sum().backward()

    assert torch.allclose(g_h, hidden2.grad, rtol=1e-4, atol=1e-5)
    # weight grad crosses tiles: different (both fp32) summation orders
    assert torch.allclose(g_w, weight2.grad, rtol=1e-4, atol=1e-4)
    assert weight.grad.dtype is torch.float32


def test_ignore_index_positions_return_zero_and_no_gradient():
    hidden, weight, target = _case(seed=1)
    target[0, :3] = -100
    out = fused_linear_cross_entropy(hidden, weight, target, 2, ignore_index=-100)
    assert (out[0, :3] == 0.0).all()
    ref = _ref_nll(hidden.detach(), weight.detach(), target)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)

    out.sum().backward()
    # gradient flows only through the valid rows (compare vs reference grads)
    hidden2, weight2, target2 = _case(seed=1)
    target2[0, :3] = -100
    _ref_nll(hidden2, weight2, target2).sum().backward()
    assert torch.allclose(hidden.grad, hidden2.grad, rtol=1e-4, atol=1e-5)
    assert torch.allclose(weight.grad, weight2.grad, rtol=1e-4, atol=1e-4)


def _mixed_case(seed):
    """Realistic training dtypes: bf16 activations, fp32 master weights."""
    torch.manual_seed(seed)
    hidden = torch.randn(B, L, DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(VOCAB, DIM, device="cuda", dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, VOCAB, (B, L), device="cuda")
    return hidden, weight, target


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 parity probe needs a GPU")
def test_bf16_matches_reference_under_autocast():
    for tile_size in (1, 4, 1024):
        hidden, weight, target = _mixed_case(2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fused_linear_cross_entropy(hidden, weight, target, tile_size)
            ref = _ref_nll(hidden.detach(), weight.detach(), target)
        assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2)  # ~1 ulp at |nll|~10

        out.sum().backward()
        hidden2, weight2, target2 = _mixed_case(2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref2 = _ref_nll(hidden2, weight2, target2)
        ref2.sum().backward()
        assert torch.allclose(hidden.grad, hidden2.grad, rtol=1e-2, atol=1e-2)
        assert torch.allclose(weight.grad, weight2.grad, rtol=1e-2, atol=1e-1)
        assert weight.grad.dtype is torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="memory probe needs a GPU")
def test_full_logits_are_never_materialized():
    """A [2, 512, 250k] fp32 logits tensor (plus its grad) would need >2 GiB;
    the operator must stay well under that."""
    V, D, B, L, tile = 250_000, 64, 2, 512, 128
    hidden = torch.randn(B, L, D, device="cuda", requires_grad=True)
    weight = torch.randn(V, D, device="cuda", requires_grad=True)
    target = torch.randint(0, V, (B, L), device="cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    nll = fused_linear_cross_entropy(hidden, weight, target, tile)
    nll.sum().backward()
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    # Tiled fp32 legitimately peaks around 1.1 GiB here (per-tile logits plus
    # CE's internal softmax, ~256 MiB each); materializing the full
    # [2, 512, 250k] logits would need well over 3 GiB before gradients.
    assert peak_gib < 1.5, f"logits were materialized (peak {peak_gib:.2f} GiB)"
