"""
Ulysses-style Sequence Parallel (SP) for long-context training.

Each GPU holds a chunk of the input sequence (seq_len / sp_size tokens).
In attention layers, all-to-all redistributes Q/K/V from
    (B, num_heads, seq_local, head_dim) -> (B, num_heads/sp, seq_full, head_dim)
so every head sees the complete context. After attention, a reverse all-to-all
restores the original layout.

Requires: num_q_heads % sp_size == 0 AND num_kv_heads % sp_size == 0.

Currently supported: Qwen2, Qwen3, Qwen3.5 (including Qwen3.5-MoE).
To extend support to other model families, add their model_type to
_SUPPORTED_MODEL_TYPES and verify correctness.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional
from torch.distributed import ProcessGroup

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported model families for Ulysses SP
# Add new families here after verifying correctness with the attention patch.
# ---------------------------------------------------------------------------
_SUPPORTED_MODEL_TYPES = frozenset({
    "qwen2", "qwen2_moe",
    "qwen3", "qwen3_moe",
    "qwen3_5", "qwen3_5_text", "qwen3_5_moe",
})


# ---------------------------------------------------------------------------
# Global SP state
# ---------------------------------------------------------------------------
_sp_group: Optional[ProcessGroup] = None
_sp_rank: int = 0
_sp_world_size: int = 1


def init_sp_group(sp_group: ProcessGroup):
    global _sp_group, _sp_rank, _sp_world_size
    _sp_group = sp_group
    _sp_rank = dist.get_rank(sp_group)
    _sp_world_size = dist.get_world_size(sp_group)


def get_sp_group() -> Optional[ProcessGroup]:
    return _sp_group


def get_sp_rank() -> int:
    return _sp_rank


def get_sp_world_size() -> int:
    return _sp_world_size


def get_sp_data_parallel_info(global_rank: int, world_size: int, sp_size: int):
    """Return (dp_rank, dp_size) for data loading with SP.

    Ranks within the same SP group share the same dp_rank so they
    receive identical data from the sampler.
    """
    dp_size = world_size // sp_size
    dp_rank = global_rank // sp_size
    return dp_rank, dp_size


# ---------------------------------------------------------------------------
# SP process-group creation
# ---------------------------------------------------------------------------
def create_sp_process_groups(world_size: int, sp_size: int) -> ProcessGroup:
    """Create SP sub-groups. Returns the group this rank belongs to."""
    assert world_size % sp_size == 0, (
        f"world_size ({world_size}) must be divisible by sp_size ({sp_size})"
    )
    my_rank = dist.get_rank()
    my_group = None
    for start in range(0, world_size, sp_size):
        ranks = list(range(start, start + sp_size))
        group = dist.new_group(ranks)
        if my_rank in ranks:
            my_group = group
    return my_group


# ---------------------------------------------------------------------------
# All-to-all with autograd
# ---------------------------------------------------------------------------
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, scatter_dim, gather_dim, group):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = group
        ws = dist.get_world_size(group)
        if ws == 1:
            return inp
        chunks_in = [c.contiguous() for c in inp.chunk(ws, dim=scatter_dim)]
        chunks_out = [torch.empty_like(c) for c in chunks_in]
        dist.all_to_all(chunks_out, chunks_in, group=group)
        return torch.cat(chunks_out, dim=gather_dim)

    @staticmethod
    def backward(ctx, grad):
        return (
            _AllToAll.apply(grad, ctx.gather_dim, ctx.scatter_dim, ctx.group),
            None, None, None,
        )


def all_to_all(tensor, scatter_dim, gather_dim, group):
    return _AllToAll.apply(tensor.contiguous(), scatter_dim, gather_dim, group)


# ---------------------------------------------------------------------------
# Model-agnostic RoPE (inline, no model-specific imports)
# ---------------------------------------------------------------------------
def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin, unsqueeze_dim=2):
    """Apply rotary position embedding.

    Works with x in (B, S, H, D) format when unsqueeze_dim=2,
    or in (B, H, S, D) format when unsqueeze_dim=1.

    Supports partial rotary (e.g. Qwen3.5 with partial_rotary_factor=0.25):
    when cos/sin last dim < x last dim, only rotate the first rotary_dim
    elements and leave the rest unchanged.
    """
    rotary_dim = cos.shape[-1]
    head_dim = x.shape[-1]

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    if rotary_dim < head_dim:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_rot = (x_rot * cos) + (_rotate_half(x_rot) * sin)
        return torch.cat([x_rot, x_pass], dim=-1)

    return (x * cos) + (_rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# Input splitting
# ---------------------------------------------------------------------------
def split_for_sp(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
) -> dict:
    """Split a batch along the sequence dim for SP.

    Returns a dict ready to be fed to the model, including correct
    ``position_ids`` so that RoPE sees the true absolute positions.
    """
    sp_rank = get_sp_rank()
    sp_size = get_sp_world_size()

    if sp_size <= 1:
        result = {"input_ids": input_ids, "attention_mask": attention_mask}
        if labels is not None:
            result["labels"] = labels
        return result

    seq_len = input_ids.shape[1]
    assert seq_len % sp_size == 0, (
        f"seq_len ({seq_len}) must be divisible by sp_size ({sp_size}). "
        f"Pad sequences to a multiple of sp_size."
    )

    chunk = seq_len // sp_size
    lo = sp_rank * chunk
    hi = lo + chunk
    B = input_ids.shape[0]

    result = {
        "input_ids": input_ids[:, lo:hi].contiguous(),
        "attention_mask": attention_mask[:, lo:hi].contiguous(),
        "position_ids": torch.arange(lo, hi, device=input_ids.device)
                             .unsqueeze(0).expand(B, -1),
    }
    if labels is not None:
        result["labels"] = labels[:, lo:hi].contiguous()
    return result


# ---------------------------------------------------------------------------
# Compatibility validation
# ---------------------------------------------------------------------------
def _validate_sp_compatibility(model, sp_size: int):
    """Validate that the model is compatible with Ulysses SP before patching.

    Checks:
      1. At least one attention layer is found
      2. num_q_heads and num_kv_heads are divisible by sp_size
      3. Attention forward signature accepts position_embeddings
      4. The model has a rotary embedding (RoPE) module
      5. Required attributes (config, scaling, etc.) exist on each attn module

    Raises RuntimeError with a clear message on failure.
    """
    import inspect

    attn_modules = []
    for name, m in model.named_modules():
        if _is_attention(m):
            attn_modules.append((name, m))

    # --- Check 1: at least one attention layer found -------------------------
    if not attn_modules:
        all_classes = {type(m).__name__ for _, m in model.named_modules()}
        raise RuntimeError(
            f"[SP] No compatible attention layers found in {type(model).__name__}. "
            f"Ulysses SP requires attention modules with (q_proj, k_proj, o_proj, head_dim). "
            f"Module classes in model: {sorted(all_classes)}"
        )

    first_name, first_attn = attn_modules[0]
    errors = []

    # --- Check 2: head counts divisible by sp_size --------------------------
    config = getattr(first_attn, "config", None)
    if config is None:
        errors.append(
            f"Attention module '{first_name}' has no .config attribute — "
            f"cannot verify head counts."
        )
    else:
        num_q_heads = getattr(config, "num_attention_heads", None)
        num_kv_heads = getattr(config, "num_key_value_heads", num_q_heads)
        # Gemma4 may use num_global_key_value_heads for non-sliding layers
        if hasattr(first_attn, "use_alternative_attention") and first_attn.use_alternative_attention:
            num_kv_heads = getattr(config, "num_global_key_value_heads", num_kv_heads)

        if num_q_heads is None:
            errors.append("Cannot determine num_attention_heads from config.")
        elif num_q_heads % sp_size != 0:
            errors.append(
                f"num_attention_heads ({num_q_heads}) is not divisible by sp_size ({sp_size}). "
                f"Valid sp_size values: {[s for s in range(2, num_q_heads + 1) if num_q_heads % s == 0]}"
            )
        if num_kv_heads is not None and num_kv_heads % sp_size != 0:
            errors.append(
                f"num_key_value_heads ({num_kv_heads}) is not divisible by sp_size ({sp_size}). "
                f"Valid sp_size values: {[s for s in range(2, num_kv_heads + 1) if num_kv_heads % s == 0]}"
            )

    # --- Check 3: forward signature ------------------------------------------
    orig_forward = first_attn.forward
    try:
        sig = inspect.signature(orig_forward)
        params = list(sig.parameters.keys())
        if "position_embeddings" not in params:
            errors.append(
                f"Attention forward signature {params} does not contain "
                f"'position_embeddings'. The model may use a different "
                f"positional encoding API that Ulysses SP cannot patch."
            )
    except (ValueError, TypeError):
        pass  # can't inspect, skip this check

    # --- Check 4: rotary embedding exists ------------------------------------
    has_rope = any(
        "rotary" in type(m).__name__.lower() or "rotary" in n.lower()
        for n, m in model.named_modules()
    )
    if not has_rope:
        errors.append(
            "No rotary embedding module found in the model. "
            "Ulysses SP relies on RoPE for correct position handling "
            "across sequence chunks."
        )

    # --- Check 5: required attributes on attention modules -------------------
    for attr in ("scaling",):
        if not hasattr(first_attn, attr):
            errors.append(
                f"Attention module missing required attribute '{attr}'."
            )
    attn_impl = getattr(config, "_attn_implementation", None) if config else None
    if attn_impl is None:
        errors.append(
            "config._attn_implementation not set — cannot determine "
            "attention backend (flash_attention_2 / sdpa / eager)."
        )

    # --- Report --------------------------------------------------------------
    if errors:
        header = (
            f"[SP] Model {type(model).__name__} failed Ulysses SP compatibility check "
            f"(sp_size={sp_size}):\n"
        )
        detail = "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(header + detail)

    # All good — log summary
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        model_cls = type(model).__name__
        attn_cls = type(first_attn).__name__
        num_q = getattr(config, "num_attention_heads", "?") if config else "?"
        num_kv = getattr(config, "num_key_value_heads", num_q) if config else "?"
        has_v_norm = hasattr(first_attn, "v_norm")
        has_v_proj = getattr(first_attn, "v_proj", None) is not None
        logger.info(
            f"[SP] Compatibility check passed for {model_cls}:\n"
            f"      attention_cls  = {attn_cls}\n"
            f"      q_heads={num_q}, kv_heads={num_kv}, sp_size={sp_size}\n"
            f"      attn_impl={attn_impl}, head_dim={first_attn.head_dim}\n"
            f"      v_proj={'yes' if has_v_proj else 'NO (k_eq_v)'}, "
            f"v_norm={'yes' if has_v_norm else 'no'}\n"
            f"      num_layers={len(attn_modules)}"
        )


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------
def _detect_model_family(model) -> str:
    """Return the model_type string from the model's config.

    Falls back to the model class name if config.model_type is unavailable.
    """
    config = getattr(model, "config", None)
    if config is not None:
        model_type = getattr(config, "model_type", None)
        if model_type is not None:
            return model_type
    return type(model).__name__


def _check_model_support(model_type: str):
    """Validate that the model family is supported by Ulysses SP.

    Raises RuntimeError for unsupported families with a clear message.
    Logs an info message for supported ones.
    """
    if model_type in _SUPPORTED_MODEL_TYPES:
        logger.info(f"[SP] Detected supported model family: {model_type}")
        return

    supported = ", ".join(sorted(_SUPPORTED_MODEL_TYPES))
    raise RuntimeError(
        f"[SP] Model type '{model_type}' is not supported for Sequence Parallel.\n"
        f"     Currently supported families: {supported}\n"
        f"     To add support: verify the model's attention module works with\n"
        f"     the Ulysses SP patch (q/k/v/o_proj, head_dim, RoPE), fix any\n"
        f"     quirk (gate, partial_rotary, etc.), then add '{model_type}' to\n"
        f"     _SUPPORTED_MODEL_TYPES in ops/parallel/sequence_parallel.py."
    )


# ---------------------------------------------------------------------------
# Monkey-patch attention layers
# ---------------------------------------------------------------------------
def apply_ulysses_sp(model, sp_group: ProcessGroup):
    """Replace every compatible attention forward with Ulysses SP.

    Works with any HuggingFace model whose attention modules have
    q_proj, k_proj, o_proj, and head_dim (Qwen2/3, Gemma3/4, Llama, etc.).
    """
    sp_size = dist.get_world_size(sp_group)
    if sp_size <= 1:
        return

    init_sp_group(sp_group)

    model_type = _detect_model_family(model)
    _check_model_support(model_type)

    _validate_sp_compatibility(model, sp_size)

    count = 0
    for _name, module in model.named_modules():
        if _is_attention(module):
            _patch(module, sp_group)
            count += 1

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        logger.info(f"[SP] Ulysses patched {count} attention layers (sp_size={sp_size})")


def _is_attention(m):
    """Detect attention modules by checking for standard projection layers."""
    return all(hasattr(m, a) for a in ("q_proj", "k_proj", "o_proj", "head_dim"))


def _patch(attn, sp_group):
    """Build a model-agnostic Ulysses forward closure."""

    def _ulysses_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]             # (B, S_local)
        hidden_shape = (*input_shape, -1, attn.head_dim)   # (B, S_local, H, D)

        cos, sin = position_embeddings

        # --- QKV projection → (B, S_local, H, D) ----------------------------
        q_proj_out = attn.q_proj(hidden_states)
        # Detect gate mechanism (e.g. Qwen3.5): q_proj outputs 2x normal dim,
        # where the second half is an output gate applied before o_proj.
        o_proj_in = attn.o_proj.in_features  # normal dim = num_heads * head_dim
        _gate = None
        if q_proj_out.shape[-1] > o_proj_in:
            q = q_proj_out[..., :o_proj_in].view(hidden_shape)
            _gate = q_proj_out[..., o_proj_in:]
        else:
            q = q_proj_out.view(hidden_shape)
        k = attn.k_proj(hidden_states).view(hidden_shape)

        # Gemma4 attention_k_eq_v: v_proj can be None → v = k (before norms)
        v_proj = getattr(attn, "v_proj", None)
        v_raw = v_proj(hidden_states).view(hidden_shape) if v_proj is not None else k

        # --- Norms -----------------------------------------------------------
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        if hasattr(attn, "k_norm"):
            k = attn.k_norm(k)
        # v_norm exists in Gemma4 but not Qwen3
        v = attn.v_norm(v_raw) if hasattr(attn, "v_norm") else v_raw

        # --- RoPE in (B, S, H, D) then transpose to (B, H, S, D) -----------
        q = _apply_rope(q, cos, sin, unsqueeze_dim=2)
        k = _apply_rope(k, cos, sin, unsqueeze_dim=2)

        q = q.transpose(1, 2)   # (B, Hq,  S_local, D)
        k = k.transpose(1, 2)   # (B, Hkv, S_local, D)
        v = v.transpose(1, 2)   # (B, Hkv, S_local, D)

        # --- Ulysses all-to-all FORWARD: scatter heads, gather seq -----------
        q = all_to_all(q, scatter_dim=1, gather_dim=2, group=sp_group)
        k = all_to_all(k, scatter_dim=1, gather_dim=2, group=sp_group)
        v = all_to_all(v, scatter_dim=1, gather_dim=2, group=sp_group)

        # --- Attention (memory-efficient, handles large head_dim) -----------
        num_kv_groups = getattr(attn, "num_key_value_groups", 1)
        attn_out, attn_weights = _sp_attention(
            q, k, v,
            scaling=attn.scaling,
            num_key_value_groups=num_kv_groups,
        )

        # --- Ulysses all-to-all REVERSE: scatter seq, gather heads -----------
        attn_out = attn_out.transpose(1, 2)   # (B, H_local, S_full, D)
        attn_out = all_to_all(attn_out, scatter_dim=2, gather_dim=1, group=sp_group)
        attn_out = attn_out.transpose(1, 2)   # (B, S_local, H, D)

        # --- Output projection (with optional gate) -------------------------
        attn_out = attn_out.reshape(*input_shape, -1).contiguous()
        if _gate is not None:
            _gate = _gate.reshape(*input_shape, -1).to(dtype=attn_out.dtype)
            attn_out = attn_out * torch.sigmoid(_gate)
        attn_out = attn.o_proj(attn_out)
        return attn_out, attn_weights

    attn.forward = _ulysses_forward


# ---------------------------------------------------------------------------
# GQA helper
# ---------------------------------------------------------------------------
def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)


# ---------------------------------------------------------------------------
# SP attention dispatcher: avoids O(S²) memory for large head_dim
# ---------------------------------------------------------------------------
def _sp_attention(q, k, v, scaling, num_key_value_groups=1, chunk_size=2048):
    """Memory-efficient attention for SP.

    Fallback chain:
      1. SDPA Flash / Efficient kernels (fastest, head_dim ≤ 256)
      2. xformers memory_efficient_attention (fast, supports larger head_dim)
      3. Chunked attention with online softmax (always works, O(chunk) memory)
    """
    if num_key_value_groups > 1:
        k = _repeat_kv(k, num_key_value_groups)
        v = _repeat_kv(v, num_key_value_groups)

    # --- Strategy 1: SDPA efficient kernels (exclude math to avoid S×S) ---
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=scaling,
            )
        return out.transpose(1, 2).contiguous(), None
    except (RuntimeError, ImportError):
        pass

    # --- Strategy 2: xformers (supports arbitrary head_dim) ---------------
    try:
        import xformers.ops as xops
        out = xops.memory_efficient_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_bias=xops.LowerTriangularMask(),
            scale=scaling,
        )
        return out.contiguous(), None
    except (ImportError, RuntimeError):
        pass

    # --- Strategy 3: chunked attention with online softmax ----------------
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        if not getattr(_sp_attention, "_warned", False):
            logger.warning(
                "[SP] SDPA efficient kernels and xformers unavailable for "
                f"head_dim={q.shape[-1]}. Using chunked attention fallback "
                f"(chunk_size={chunk_size}). Training will be slower."
            )
            _sp_attention._warned = True
    return _chunked_causal_attention(q, k, v, scaling, chunk_size)


def _chunked_causal_attention(q, k, v, scaling, chunk_size=2048):
    """Causal attention via chunked online softmax.

    Memory per step: O(H × chunk_size × max(chunk_size, D)) instead of O(S²).
    """
    B, H, S, D = q.shape
    device, dtype = q.device, q.dtype

    output = torch.empty(B, H, S, D, device=device, dtype=dtype)

    for q_start in range(0, S, chunk_size):
        q_end = min(q_start + chunk_size, S)
        q_chunk = q[:, :, q_start:q_end].float()
        cs_q = q_end - q_start

        m = torch.full((B, H, cs_q, 1), float("-inf"), device=device)
        l = torch.zeros((B, H, cs_q, 1), device=device)
        o = torch.zeros((B, H, cs_q, D), device=device)

        for kv_start in range(0, q_end, chunk_size):
            kv_end = min(kv_start + chunk_size, q_end)
            k_chunk = k[:, :, kv_start:kv_end].float()
            v_chunk = v[:, :, kv_start:kv_end].float()

            scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scaling

            if kv_end > q_start:
                q_idx = torch.arange(q_start, q_end, device=device).unsqueeze(1)
                kv_idx = torch.arange(kv_start, kv_end, device=device).unsqueeze(0)
                scores = scores.masked_fill(kv_idx > q_idx, float("-inf"))

            chunk_max = scores.amax(dim=-1, keepdim=True)
            chunk_max = torch.clamp(chunk_max, min=-1e30)

            m_new = torch.maximum(m, chunk_max)
            alpha = torch.exp(m - m_new)
            p = torch.exp(scores - m_new)

            o = o * alpha + torch.matmul(p, v_chunk)
            l = l * alpha + p.sum(dim=-1, keepdim=True)
            m = m_new

        output[:, :, q_start:q_end] = (o / l.clamp(min=1e-6)).to(dtype)

    return output.transpose(1, 2).contiguous(), None
