"""Per-token log-probability gather, shared by on-policy presets.

The function takes a causal LM and an input batch, runs a forward pass, and
returns the log-prob of every next token in the input sequence — i.e. the
log-prob of token ``input_ids[:, t+1]`` under the model conditioned on
``input_ids[:, :t+1]``.  Callers are responsible for combining the result
with their own ``response_mask[:, 1:]`` to select the response tokens.

This helper is shared by GRPO and OPD; it lives in ``ops/`` rather than
either preset because both need it.

Memory: with a 248k-vocabulary model a [B, L, V] float tensor is several GB.
The implementation therefore evaluates the fused ``F.cross_entropy`` (whose
autograd recomputes the softmax from the saved logits) instead of
materializing ``log_softmax`` + ``gather``, and divides the temperature
in place — together removing three full-vocabulary copies per call.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F


def compute_per_token_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    logp_min_clamp: float | None = None,
    no_grad: bool = False,
) -> torch.Tensor:
    """Forward *model* and return next-token log-probabilities.

    Args:
        model: A causal LM with ``forward(input_ids, attention_mask) -> output.logits``.
        input_ids: ``[B, L]`` token ids.
        attention_mask: ``[B, L]`` 0/1 mask.
        temperature: The sampling temperature the rollout used.  The
            log-probs must describe the distribution the tokens were *drawn*
            from, so the logits are divided by it before the softmax (as TRL
            and verl do); at T=1 the untempered log-probs were a biased
            estimator for every T<1 rollout.  top-p / top-k cannot be matched
            exactly and are ignored, as elsewhere.
        logp_min_clamp: Optional lower bound clamp on the returned log-probs.
            Useful to keep PG ratios numerically sane when one of the two
            policies assigns near-zero probability.
        no_grad: When True, wrap the forward in ``torch.no_grad()``.  Used for
            reference / teacher forwards.

    Returns:
        ``[B, L-1]`` tensor of log-probs aligned with ``input_ids[:, 1:]``.
    """
    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits  # [B, L, V]
        if temperature != 1.0:
            # Out of place on purpose: mutating output.logits in place would
            # leak the tempering into any tensor that shares the storage
            # (the tests caught a fake LM returning its own buffer), and a
            # copy here is one full-vocabulary tensor, not three.
            logits = logits / temperature

        # Fused cross_entropy instead of log_softmax + gather: identical
        # mathematics (logp(target) == -CE(logits, target)) but its autograd
        # never materializes the [B, L, V] log-prob tensor — backward
        # recomputes the softmax from the saved logits.  With a 248k vocab
        # this removes three [B, L, V] float copies per call (measured peak
        # ~20G -> ~10G for a 2x2559 batch on MPS; CUDA benefits the same way).
        #
        # CE runs over ALL L positions with rolled targets so the big reshape
        # below is a free view of the contiguous logits (slicing first would
        # force a [B, L-1, V] copy).  The extra last column pairs position
        # L-1's logits with token 0 — garbage — but it is sliced off BEFORE
        # callers reduce, so it never receives gradient and cannot leak into
        # the real positions (pinned by a gradient-parity test).
        targets = torch.cat([input_ids[:, 1:], input_ids[:, :1]], dim=1)
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view(input_ids.size(0), input_ids.size(1))
    # CE returns the POSITIVE loss -logp; negate to restore the log-prob
    # contract (negative values), and slice the garbage last column off
    # before callers reduce — it never receives gradient that way.
    gathered = -per_token[:, :-1]
    if logp_min_clamp is not None:
        gathered = gathered.clamp(min=logp_min_clamp)
    return gathered
