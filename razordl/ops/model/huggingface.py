import re
import os

import torch
from transformers import PreTrainedModel


def convert_weight_keys(state_dict: dict[str, torch.Tensor], model: PreTrainedModel):
    """
    这是一个用于模型权重键名对齐的工具函数，主要用于处理 Hugging Face Transformers 库中某些模型因结构更新或命名变更导致的权重加载不匹配问题（注释中提到的 PR #38385 也佐证了这一点，通常与 Qwen2 或 Llama 等模型的不同实现版本有关）。
    """
    # convert state dict keys: https://github.com/huggingface/transformers/pull/38385
    if not hasattr(model, "_checkpoint_conversion_mapping"):
        return state_dict

    reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
    original_weights = {}
    for key, value in state_dict.items():
        for pattern, replacement in reverse_key_mapping.items():
            replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
            replacement = re.sub(r"\(.*\)", "", replacement)
            key, n_replace = re.subn(pattern, replacement, key)
            # Early exit of the loop
            if n_replace > 0:
                break

        original_weights[key] = value

    return original_weights


def resolve_compute_dtype(precision: str = "auto"):
    """Resolve a `precision` config value into the *compute* dtype.

    This is what the forward/backward runs in -- i.e. what FSDP2 gets as
    `MixedPrecisionPolicy(param_dtype=...)`.  To load weights, use
    :func:`resolve_storage_dtype` instead: under fp16 the two differ.

    Delegates to `razordl.ops.hardware.precision` -- the single source of truth
    for dtype selection.  Never probe capabilities here.
    """
    from razordl.ops.hardware.precision import resolve_precision, to_torch_dtype

    return to_torch_dtype(resolve_precision(precision))


def resolve_storage_dtype(precision: str = "auto", *, trainable: bool = True):
    """Resolve a `precision` config value into the dtype weights are held in.

    Equals the compute dtype except under fp16, where parameters stay fp32 as
    master weights.  Model loading must use this so every parameter of a
    sharded unit shares one dtype (FSDP2 asserts on mixed dtypes).

    ``trainable=False`` (reference / teacher models) skips the promotion: with
    no optimizer there is nothing to keep a master copy for, and loading at the
    compute dtype halves both peak and resident memory.
    """
    from razordl.ops.hardware.precision import (
        resolve_precision,
        to_storage_dtype,
        to_torch_dtype,
    )

    resolved = resolve_precision(precision)
    return to_storage_dtype(resolved) if trainable else to_torch_dtype(resolved)


def resolve_attn_implementation(local_rank: int = 0, logger=None, deterministic_env: bool = True) -> str:
    if deterministic_env and os.environ.get("RAZORDL_DETERMINISTIC") in {"1", "true", "True"}:
        return "eager"

    from razordl.ops.hardware import device

    fallback = "sdpa" if torch.cuda.is_available() else "eager"

    # flash-attn 2 requires Ampere (sm_80+). It imports fine on Turing/Volta but
    # crashes at forward, so importability alone is not enough of a criterion.
    if not device.supports_flash_attention_2():
        if logger is not None and local_rank == 0:
            cap = device.describe().get("compute_capability")
            logger.info(
                "[MODEL] flash_attention_2 needs sm_80+ (this GPU is %s), using %s",
                cap,
                fallback,
            )
        return fallback

    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        if logger is not None and local_rank == 0:
            logger.warning("[MODEL] flash_attn not installed, falling back to %s", fallback)
        return fallback


def build_left_padding_tokenizer(processor_path: str | None, model_path: str, *, ensure_pad_token: bool = False):
    from transformers import AutoTokenizer

    if not processor_path:
        processor_path = model_path

    processor = AutoTokenizer.from_pretrained(processor_path, trust_remote_code=True)
    if ensure_pad_token and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    processor.padding_side = "left"
    return processor


def enforce_model_profile(model_path: str):
    """Look up the profile for *model_path*'s ``model_type`` and validate.

    Returns the (possibly-rewritten) ``PreTrainedConfig`` to pass to
    ``from_pretrained(config=...)``.  Raises
    :class:`razordl.ops.model.profiles.UnsupportedModelError` if no profile is
    registered for the checkpoint's ``model_type``.

    Every preset's ``build_model()`` MUST call this before loading the HF
    model — see the Key Invariants section in ``CLAUDE.md``.  To add support
    for a new family, drop a file at
    ``razordl/ops/model/profiles/<model_type>.py`` and ``@register`` a
    ``ModelProfile``.
    """
    from transformers import AutoConfig

    from razordl.ops.model.profiles import PROFILES, UnsupportedModelError

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = cfg.model_type
    profile = PROFILES.get(model_type)
    if profile is None:
        raise UnsupportedModelError(
            f"Model type {model_type!r} (loaded from {model_path}) is not in the "
            f"supported set {sorted(PROFILES)}. To add support, create "
            f"razordl/ops/model/profiles/{model_type}.py and @register a ModelProfile."
        )
    profile.validate(cfg)
    return profile.prepare_config(cfg)


def build_causal_lm(
    model_path: str,
    *,
    device=None,
    precision: str = "auto",
    trainable: bool = True,
    local_rank: int = 0,
    logger=None,
    deterministic_attn: bool = True,
):
    from transformers import AutoModelForCausalLM

    cfg = enforce_model_profile(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=cfg,
        torch_dtype=resolve_storage_dtype(precision, trainable=trainable),
        attn_implementation=resolve_attn_implementation(
            local_rank=local_rank,
            logger=logger,
            deterministic_env=deterministic_attn,
        ),
        trust_remote_code=True,
    )
    if device is not None:
        model = model.to(device)
    model.config.use_cache = False
    return model