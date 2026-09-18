"""Helpers shared by the RL presets' rollout code (GRPO, OPD).

Kept in ``ops`` so both presets import one copy instead of each carrying its
own (the two ``_remove_left_padding_batch`` copies had already started to
drift in docstrings only, but the HF-generate mask bug below was duplicated
verbatim).
"""

import torch


def remove_left_padding_batch(prompt_ids: torch.Tensor, pad_id: int) -> list[list[int]]:
    """Strip the leading pad run of every row; vLLM wants unpadded token lists."""
    result = []
    for i in range(prompt_ids.size(0)):
        ids = prompt_ids[i]
        non_pad = (ids != pad_id).nonzero(as_tuple=False)
        if len(non_pad) > 0:
            result.append(ids[non_pad[0].item():].tolist())
        else:
            result.append(ids.tolist())
    return result


def hf_generate_masks(
    generated: torch.Tensor,
    prompt_mask: torch.Tensor,
    eos_token_id,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention and response masks for the output of ``model.generate``.

    ``generated`` is ``[B, P + L]`` (prompt incl. left padding, then ``L``
    generated tokens); ``prompt_mask`` is the ``[B, P]`` prompt attention
    mask that was passed to ``generate``.  The response mask covers each
    row's generated tokens up to **and including** its first EOS (all of
    them when the row hit ``max_new_tokens``); the attention mask is the
    prompt mask followed by the response mask.

    The old ``generated != pad_token_id`` rule dropped the EOS token whenever
    the tokenizer's pad and EOS ids coincide (every Llama-style chat
    template, and Qwen when ``pad_token = eos_token`` is set for batching), so
    the policy was never rewarded for stopping and responses grew to
    ``max_completion_length``.  ``eos_token_id`` may be an int or a list, as
    ``GenerationConfig`` allows.
    """
    if generated.dim() != 2 or prompt_mask.dim() != 2 or generated.size(0) != prompt_mask.size(0):
        raise ValueError(f"expected [B, P+L] generated and [B, P] prompt_mask, got {tuple(generated.shape)} / {tuple(prompt_mask.shape)}")
    prompt_len = prompt_mask.size(1)
    response = generated[:, prompt_len:]
    n_resp = response.size(1)
    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id or [])
    if eos_ids:
        is_eos = torch.isin(response, torch.tensor(eos_ids, device=response.device))
        has_eos = is_eos.any(dim=1)
        # argmax over a bool row gives the first True; rows without EOS keep everything
        first_eos = torch.where(has_eos, is_eos.int().argmax(dim=1), torch.full_like(has_eos, n_resp, dtype=torch.long))
    else:
        first_eos = torch.full((response.size(0),), n_resp, dtype=torch.long, device=response.device)
    positions = torch.arange(n_resp, device=response.device)
    response_mask_gen = (positions[None, :] <= first_eos[:, None]).long()
    prompt_mask = prompt_mask.long()
    attention_mask = torch.cat([prompt_mask, response_mask_gen], dim=1)
    response_mask = torch.cat([torch.zeros_like(prompt_mask), response_mask_gen], dim=1)
    return attention_mask, response_mask


def truncate_chat_prompt(
    tokenizer,
    messages: list[dict],
    max_length: int,
    add_generation_prompt: bool = True,
    **template_kwargs,
) -> list[int]:
    """Render ``messages`` through the chat template into at most ``max_length`` ids.

    ``tokenizer.encode(..., truncation=True)`` cuts the *end* of the rendered
    prompt, which for a generation prompt is ``<|im_start|>assistant\\n``: the
    policy then continued the user's text instead of answering it, and the
    reward was garbage for exactly the long prompts.  When the rendering is
    too long, this shortens the content of the longest user message (its
    first tokens are kept, decoded back to text) and re-renders, so the
    system prompt, the turn markers and the generation prompt all survive.
    Raises ``ValueError`` when even an empty user message does not fit.
    """

    def _render(msgs):
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt, **template_kwargs
        )
        return tokenizer.encode(text, add_special_tokens=False)

    ids = _render(messages)
    if len(ids) <= max_length:
        return ids

    user_idx = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not user_idx:
        raise ValueError(f"prompt is {len(ids)} tokens > max_length={max_length} and has no user message to shorten")
    longest = max(user_idx, key=lambda i: len(tokenizer.encode(str(messages[i]["content"]), add_special_tokens=False)))
    content_ids = tokenizer.encode(str(messages[longest]["content"]), add_special_tokens=False)
    overhead = len(ids) - len(content_ids)
    budget = max_length - overhead
    if budget <= 0:
        raise ValueError(
            f"chat template overhead is {overhead} tokens, leaving no room for user text under max_length={max_length}"
        )

    # decode/re-encode can shift the count by a token or two at the cut; shrink until it fits
    for _ in range(8):
        shortened = [dict(m) for m in messages]
        shortened[longest]["content"] = tokenizer.decode(content_ids[:budget], skip_special_tokens=True)
        ids = _render(shortened)
        if len(ids) <= max_length:
            return ids
        budget -= len(ids) - max_length
        if budget <= 0:
            break
    raise ValueError(f"could not fit the prompt into max_length={max_length}")
