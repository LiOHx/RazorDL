"""Precision policy — resolves a user request into a concrete compute dtype.

Split from `device.py` on purpose: `device.py` dispatches capability *probes*
to per-backend modules, this module holds the backend-independent *policy*
built on top of them.  Adding a new accelerator means touching `device.py`
and a new backend file, never this one.
"""

import logging

import torch

from razordl.ops.hardware import device

logger = logging.getLogger(__name__)

PRECISION_CHOICES = ("auto", "bf16", "fp16", "fp32")

_warned_emulated_bf16 = False

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

# Legacy `use_bf16: bool` maps onto the enum.  Kept for configs written before
# `precision` existed; `flat_config.py` emits a deprecation warning.
_LEGACY_USE_BF16 = {True: "bf16", False: "fp16"}


def resolve_precision(requested: str = "auto") -> str:
    """Resolve *requested* into a concrete "bf16" / "fp16" / "fp32".

    ``auto`` picks the fastest numerically-sound dtype for the hardware:

    ==========================  ========
    hardware                    result
    ==========================  ========
    CUDA, every GPU sm >= 80    bf16      (native)
    CUDA, some GPU sm < 80      bf16      (emulated in software by PyTorch)
    no accelerator              fp32
    ==========================  ========

    Emulated bf16 is still warned about once -- it is otherwise invisible -- but
    it is the *default* on pre-Ampere rather than a fallback.  Both half
    precisions now run the same recipe (fp32 master weights + autocast), and
    measured that way they cost the same; bf16 wins on simplicity, because it
    needs no loss scaling.

    Measured on an RTX 2080 Ti (sm_75), Qwen3.5-0.8B + LoRA, demo data at
    max_length 256 x batch 1, 20 steps, fsdp2 (peak = nvidia-smi total used,
    which includes ~1.4 GB of unrelated allocations):

    =====================================  =========  ===========
    path                                   step time  peak memory
    =====================================  =========  ===========
    bf16 (emulated, fp32 masters+autocast)   0.67 s     7.5 GB
    fp16 (fp32 masters+autocast+GradScaler)  0.66 s     7.5 GB
    bf16 params updated in place (old)       0.63 s     5.8 GB
    fp16 params, no masters                  fastest    unusable
    =====================================  =========  ===========

    The old in-place bf16 row was cheaper but rounded away every update below
    ``|w| / 256`` (see :func:`needs_fp32_master_weights`), so it is gone.  The
    last row is why fp16 is not simply the fast option on old cards: raw fp16
    compute *is* faster than emulated bf16, but it produces NaN gradients (see
    :func:`needs_autocast`) and ``GradScaler.unscale_`` refuses fp16 gradients
    outright.  fp16 additionally skips its first optimizer steps while the
    scaler finds a scale (two of twenty in this run) and carries scaler state
    in every checkpoint.

    ``precision: fp16`` remains fully supported; it is simply not what ``auto``
    picks for you.
    """
    if requested is None:
        requested = "auto"
    requested = str(requested).lower()
    if requested not in PRECISION_CHOICES:
        raise ValueError(
            f"precision must be one of {list(PRECISION_CHOICES)}, got {requested!r}"
        )

    if requested == "auto":
        # Half precision where the accelerator executes it, chosen by
        # capability probe rather than a raw CUDA check (this file is
        # policy, not capability — see the module docstring):
        #   cuda -> bf16 on ANY CUDA GPU, native or emulated (on pre-Ampere
        #     the emulation measured on par with fp16 once both carry fp32
        #     masters, see the table above, and bf16 needs no loss scaling);
        #   mps  -> bf16 when the runtime probe says bf16 executes natively
        #     (current PyTorch/macOS, measured -- see mps.py), fp16 on older
        #     builds where only fp16 is native;
        #   else -> fp32.
        available = device.get_available_device()
        if available == "cuda":
            resolved = "bf16"
        elif available == "mps" and device._backend("mps").supports_native_bf16():
            resolved = "bf16"
        elif available == "mps" and device._backend("mps").supports_fp16():
            resolved = "fp16"
        else:
            resolved = "fp32"
    else:
        resolved = requested

    if (
        resolved == "bf16"
        and torch.cuda.is_available()
        and not device.supports_native_bf16()
        and not _warned_emulated_bf16
    ):
        # resolve_precision() is called per model group, per backend wrap and
        # once per training step; warn once per process or it drowns the log.
        globals()["_warned_emulated_bf16"] = True
        cap = device.describe().get("compute_capability")
        logger.warning(
            "[PRECISION] running bf16 on a GPU (%s) with no native bf16 -- "
            "PyTorch emulates it in software, which is slower than a native bf16 "
            "card. It measured on par with fp16 here and needs no loss scaling: "
            "see the trade-off in ops/hardware/precision.py.",
            cap,
        )

    return resolved


def to_torch_dtype(precision: str) -> torch.dtype:
    """Map a resolved precision string to its torch dtype."""
    try:
        return _DTYPES[precision]
    except KeyError:
        raise ValueError(
            f"expected a resolved precision {list(_DTYPES)}, got {precision!r}. "
            "Call resolve_precision() first."
        ) from None


_VLLM_DTYPE_NAMES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
}


def to_vllm_dtype_name(precision: str) -> str:
    """Map a resolved precision to the dtype string vLLM expects.

    Rollout must agree with training: resolving through the same policy keeps
    the RL presets from running the engine in bf16 while the trainer fell back
    to fp16 on pre-Ampere hardware.
    """
    try:
        return _VLLM_DTYPE_NAMES[precision]
    except KeyError:
        raise ValueError(
            f"expected a resolved precision {list(_VLLM_DTYPE_NAMES)}, got {precision!r}. "
            "Call resolve_precision() first."
        ) from None


def resolve_vllm_dtype_name(precision: str) -> str:
    """vLLM dtype for a *resolved* training precision on the GPU actually present.

    Training runs bf16 everywhere CUDA exists, emulated in software on
    pre-Ampere cards (see :func:`resolve_precision`).  vLLM has no emulated
    bf16: a ``bfloat16`` engine on sm < 80 refuses to start (or, with its check
    disabled, runs garbage), which used to take the whole rollout down and drop
    the preset into the slow HF fallback.  On such a GPU rollout uses
    ``float16`` instead, the nearest dtype the card executes natively; the
    policy weights are cast on the way in, so the mismatch only affects rollout
    numerics, not the master weights.  Everywhere else this is the plain
    :func:`to_vllm_dtype_name` mapping.
    """
    name = to_vllm_dtype_name(precision)
    if precision == "bf16" and not device.supports_native_bf16():
        cap = device.describe().get("compute_capability")
        logger.warning(
            f"vLLM cannot run bf16 on this GPU ({cap}); rollout engine uses float16 "
            "while training stays on emulated bf16"
        )
        return "float16"
    return name


def to_storage_dtype(precision: str) -> torch.dtype:
    """Map a resolved precision to the dtype the *parameters* live in.

    Distinct from :func:`to_torch_dtype`, which gives the *compute* dtype:

    ==========  ==========  ==========
    precision   storage     compute
    ==========  ==========  ==========
    bf16        **fp32**    bf16
    fp16        **fp32**    fp16
    fp32        fp32        fp32
    ==========  ==========  ==========

    Both half precisions keep fp32 master weights (see
    :func:`needs_fp32_master_weights`); FSDP2's
    ``MixedPrecisionPolicy(param_dtype=...)`` casts the all-gathered copy down
    for the forward/backward, so the shard stays fp32 without costing speed,
    and under DDP :func:`needs_autocast` provides the down-cast instead.

    Every parameter of a sharded module must share this dtype -- FSDP2 asserts
    ``"FSDP expects uniform original parameter dtype"`` per sharded unit, and a
    LoRA layer holds the frozen base weight and the trainable adapter in the
    same unit, so "fp32 for trainable, fp16 for frozen" is not representable.
    """
    if needs_fp32_master_weights(precision):
        return torch.float32
    return to_torch_dtype(precision)


def needs_grad_scaler(precision: str) -> bool:
    """fp16 needs loss scaling; bf16 and fp32 do not."""
    return precision == "fp16"


def needs_autocast(precision: str) -> bool:
    """fp16 and bf16 run the forward under ``torch.autocast``; fp32 does not.

    fp16: blanket casting -- what FSDP2's ``MixedPrecisionPolicy`` does -- puts
    *every* op in fp16.  Measured on Qwen3.5-0.8B, the gated linear-attention
    recurrence produces NaN gradients from layer 14 downwards with every op in
    fp16, at *any* loss scale, while the loss itself stays finite.
    ``torch.autocast`` keeps the numerically sensitive ops (softmax, norms,
    reductions) in fp32 and the gradients stay finite.

    bf16: the parameters are fp32 master weights (see
    :func:`needs_fp32_master_weights`), so under DDP -- which has no
    ``MixedPrecisionPolicy`` -- autocast is what makes the forward compute in
    bf16 at all; under FSDP2 the policy already casts and autocast only adds
    the fp32 islands above.
    """
    return precision in ("fp16", "bf16")


def needs_fp32_master_weights(precision: str) -> bool:
    """Half-precision training keeps fp32 master weights for trainable params.

    fp16: without them the optimizer updates fp16 weights directly and
    ``clip_grad_norm_`` computes the total norm in fp16, silently zeroing
    every gradient once it overflows 65504.

    bf16: the exponent range is fine, but bf16 has only 8 mantissa bits
    (~3 significant digits).  An in-place update ``w -= lr * u`` is rounded
    away whenever ``|lr * u| < |w| / 256``, which at lr=1e-5..5e-5 is most
    updates -- the weights stop moving long before the loss plateaus.  Every
    mixed-precision recipe (Megatron, DeepSpeed, FSDP mixed precision) keeps
    fp32 masters for bf16 for that reason; RazorDL used to train bf16 in place
    and only fp16 with masters, so the two precisions did not even agree on
    what the optimizer state was.
    """
    return precision in ("fp16", "bf16")


def legacy_use_bf16_to_precision(use_bf16: bool) -> str:
    """Map the deprecated ``use_bf16`` boolean onto the precision enum."""
    return _LEGACY_USE_BF16[bool(use_bf16)]
