"""Fused linear cross-entropy (Liger Kernel / Cut Cross-Entropy style).

The full ``[B, L, V]`` logits tensor is never materialized: hidden states
stream through the lm_head in tiles and each tile's logits are recomputed in
backward.  This is the semantics the training community calls "fused linear
CE" (LigerFusedLinearCrossEntropyLoss, Apple's Cut Cross-Entropy) -- but this
is NOT a fused Triton kernel: a portable PyTorch implementation that also
runs on MPS / CPU, where Triton is unavailable.  Numerically equivalent to
``F.cross_entropy(F.linear(hidden, weight), target)`` with the CE in fp32.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F


class FusedLinearCrossEntropy(torch.autograd.Function):
    """Per-token NLL of ``F.linear(hidden, weight)`` without materializing logits.

    Args:
        hidden: ``[B, L, D]`` post-final-norm hidden states.
        weight: lm_head weight ``[V, D]``.
        target: ``[B, L]`` token ids, one per position.
        tile_size: sequence tiles.  ``tile_size >= L`` degenerates to a single
            pass -- the plain computation, just with a recomputed backward.
        softcap: optional Gemma-style final logit softcapping value.
        ignore_index: target value ignored by the CE (its nll is 0.0, same as
            ``F.cross_entropy(reduction="none")``).

    Returns:
        ``[B, L]`` fp32 per-token NLL.

    Backward recomputes each tile's logits from the saved hidden/weight and
    accumulates ``d(weight)`` in an fp32 buffer: per-tile grads are computed
    and summed in fp32, matching the fused kernels' internal fp32 reduction
    (a hand-rolled predecessor that accumulated per-tile bf16 grads drifted
    1-2 ulp from it).
    """

    @staticmethod
    def forward(ctx, hidden, weight, target, tile_size, softcap, ignore_index):
        B, L, _ = hidden.shape
        nll = hidden.new_empty(B, L, dtype=torch.float32)
        for start in range(0, L, tile_size):
            end = min(start + tile_size, L)
            logits = F.linear(hidden[:, start:end], weight)
            if softcap is not None:
                logits = torch.tanh(logits / softcap) * softcap
            tile_nll = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                target[:, start:end].reshape(-1),
                reduction="none",
                ignore_index=ignore_index,
            )
            nll[:, start:end] = tile_nll.view(B, end - start)
        ctx.save_for_backward(hidden, weight, target)
        ctx.tile_size = tile_size
        ctx.softcap = softcap
        ctx.ignore_index = ignore_index
        return nll

    @staticmethod
    def backward(ctx, grad_nll):
        hidden, weight, target = ctx.saved_tensors
        B, L, _ = hidden.shape
        V = weight.shape[0]

        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        # Recompute under the compute dtype the forward actually ran in: the
        # saved hidden already carries it (bf16 under autocast), so replaying
        # autocast with hidden.dtype reproduces the forward's F.linear
        # outside an autocast region (where mixed dtypes would error).
        recompute_ctx = (
            torch.autocast(hidden.device.type, dtype=hidden.dtype)
            if hidden.dtype in (torch.float16, torch.bfloat16)
            else contextlib.nullcontext()
        )
        for start in range(0, L, ctx.tile_size):
            end = min(start + ctx.tile_size, L)
            h_tile = hidden[:, start:end].detach().requires_grad_(True)
            w_tile = weight.detach().requires_grad_(True)
            # enable_grad: a Function's backward runs with autograd OFF, so
            # the recomputation below would build no graph without it (the
            # canonical pattern, same as torch.utils.checkpoint).
            with torch.enable_grad(), recompute_ctx:
                logits = F.linear(h_tile, w_tile)
                if ctx.softcap is not None:
                    logits = torch.tanh(logits / ctx.softcap) * ctx.softcap
                tile_nll = F.cross_entropy(
                    logits.float().reshape(-1, V),
                    target[:, start:end].reshape(-1),
                    reduction="none",
                    ignore_index=ctx.ignore_index,
                )
            tile_grad_nll = grad_nll[:, start:end].reshape(-1)
            grad_h, grad_w = torch.autograd.grad(
                tile_nll, (h_tile, w_tile), grad_outputs=tile_grad_nll
            )
            grad_hidden[:, start:end] = grad_h.to(grad_hidden.dtype)
            grad_weight += grad_w.float()
        return grad_hidden, grad_weight.to(weight.dtype), None, None, None, None


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    tile_size: int,
    *,
    temperature: float = 1.0,
    softcap: float | None = None,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Per-token NLL of the tempered lm_head without materializing logits.

    ``temperature`` divides the logits; it is applied to ``hidden`` before
    the tiles (mathematically identical to dividing the logits -- linear in
    the inputs -- and four orders of magnitude smaller than a ``[B, L, V]``
    division).
    """
    if temperature != 1.0:
        hidden = hidden / temperature
    if ignore_index is None:
        ignore_index = -100
    if tile_size <= 0:
        # A negative step makes the tile loops no-ops: forward would return
        # UNINITIALIZED nll (garbage loss) and backward zero gradients, with
        # no exception -- the adversarial review reproduced exactly that
        # silent-corruption shape.  Reject it loudly at the single choke
        # point every call site flows through.
        raise ValueError(f"tile_size must be a positive integer, got {tile_size}")
    return FusedLinearCrossEntropy.apply(
        hidden, weight, target, int(tile_size), softcap, ignore_index
    )
