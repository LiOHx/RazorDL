"""Per-token log-probability gather, shared by on-policy presets.

The function takes a causal LM and an input batch, runs a forward pass, and
returns the log-prob of every next token in the input sequence — i.e. the
log-prob of token ``input_ids[:, t+1]`` under the model conditioned on
``input_ids[:, :t+1]``.  Callers are responsible for combining the result
with their own ``response_mask[:, 1:]`` to select the response tokens.

Memory: with a 248k-vocabulary model a [B, L, V] float tensor is several GB,
so the forward runs with ``logits_to_keep=1`` (the model never computes more
than one lm_head row) and the per-token CE streams through
:class:`razordl.ops.loss.FusedLinearCrossEntropy` — the full logits tensor
is never materialized.  This helper is shared by GRPO and OPD; it lives in
``ops/`` rather than either preset because both need it.
"""
from __future__ import annotations

import contextlib

import torch

from razordl.ops.loss.fused_linear_ce import fused_linear_cross_entropy

# Sequence tiles for the streaming lm_head; the CE's transient memory scales
# with the tile, not the sequence.
_TILE_SIZE = 1024


def compute_per_token_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    logp_min_clamp: float | None = None,
    no_grad: bool = False,
    tile_size: int | None = None,
) -> torch.Tensor:
    """Forward *model* and return next-token log-probabilities.

    Args:
        model: A causal LM with ``forward(input_ids, attention_mask,
            output_hidden_states=..., logits_to_keep=...) -> output`` where
            ``output.hidden_states[-1]`` is the post-final-norm hidden state
            (standard HF behaviour) and ``get_output_embeddings()`` returns
            the lm_head.
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
        tile_size: sequence tiles for the streaming CE; defaults to the
            module constant (callers with a config pass
            ``model_config.fused_linear_tile_size`` explicitly).

    Returns:
        ``[B, L-1]`` tensor of log-probs aligned with ``input_ids[:, 1:]``.
    """
    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        hidden = output.hidden_states[-1]  # [B, L, D], post final norm
        raw_model = model.module if hasattr(model, "module") else model
        weight = raw_model.get_output_embeddings().weight  # [V, D]

        # CE runs over ALL L positions with rolled targets so no big slice or
        # reshape copy is needed; the extra last column pairs position L-1's
        # hidden with token 0 — garbage — but it is sliced off BEFORE callers
        # reduce, so it never receives gradient and cannot leak into the real
        # positions (pinned by a gradient-parity test).
        targets = torch.cat([input_ids[:, 1:], input_ids[:, :1]], dim=1)
        nll = fused_linear_cross_entropy(
            hidden, weight, targets,
            _TILE_SIZE if tile_size is None else tile_size,
            temperature=temperature,
        )
    gathered = -nll[:, :-1]
    if logp_min_clamp is not None:
        gathered = gathered.clamp(min=logp_min_clamp)
    return gathered
