"""Per-token log-probability gather, shared by on-policy presets.

The function takes a causal LM and an input batch, runs a forward pass, and
returns the log-prob of every next token in the input sequence — i.e. the
log-prob of token ``input_ids[:, t+1]`` under the model conditioned on
``input_ids[:, :t+1]``.  Callers are responsible for combining the result
with their own ``response_mask[:, 1:]`` to select the response tokens.

This helper is shared by GRPO and OPD; it lives in ``ops/`` rather than
either preset because both need it.
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
    logp_min_clamp: float | None = None,
    no_grad: bool = False,
) -> torch.Tensor:
    """Forward *model* and return next-token log-probabilities.

    Args:
        model: A causal LM with ``forward(input_ids, attention_mask) -> output.logits``.
        input_ids: ``[B, L]`` token ids.
        attention_mask: ``[B, L]`` 0/1 mask.
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
        logits_shifted = output.logits[:, :-1, :].contiguous()
        target_ids = input_ids[:, 1:].contiguous()
        log_probs = F.log_softmax(logits_shifted, dim=-1)
        gathered = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(-1)).squeeze(-1)
    if logp_min_clamp is not None:
        gathered = gathered.clamp(min=logp_min_clamp)
    return gathered
