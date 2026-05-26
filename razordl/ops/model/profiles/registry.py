"""Per-model-type profile registry for HF model loading.

Each ``ModelProfile`` subclass declares explicit support for one HF
``model_type`` and gets a chance to adjust the ``PretrainedConfig`` (or veto
the load) before ``from_pretrained`` instantiates the model.  Models without
a registered profile are rejected by ``enforce_model_profile`` in
``razordl.ops.model.huggingface``.

Why this exists: composing vLLM + transformers + FSDP2 occasionally requires
ad-hoc adjustment to the HF config object — e.g. vLLM 0.19+ rewrites the
``AutoConfig`` registry for some Qwen3.5 variants, which causes
``Qwen3_5ForCausalLM`` to receive the top-level multimodal config instead of
the text-only one it expects.  Profiles capture those adjustments in one
named place, instead of growing inline branches in ``build_causal_lm``.
"""
from __future__ import annotations

import abc


class UnsupportedModelError(RuntimeError):
    """Raised when a checkpoint's ``model_type`` has no registered profile."""


class ModelProfile(abc.ABC):
    """Per-model-type customization invoked right before ``from_pretrained``.

    Profiles are stateless, idempotent, and only responsible for HF-load-time
    concerns: ``PretrainedConfig`` adjustments and hard precondition checks.
    They MUST NOT contain training logic.

    Subclasses set the ``model_type`` ClassVar to a value matching
    ``cfg.model_type`` (e.g. ``"qwen3"``, ``"qwen3_5"``).
    """

    model_type: str = ""

    def validate(self, cfg) -> None:
        """Raise :class:`UnsupportedModelError` on incompatible variants.

        Default: accept anything.  Override to reject e.g. quantization
        formats the rest of the pipeline can't handle.
        """

    def prepare_config(self, cfg):
        """Return the config object to pass to ``from_pretrained(config=...)``.

        Default: identity.  Override to unwrap nested configs etc.
        """
        return cfg


PROFILES: dict[str, ModelProfile] = {}


def register(profile_cls: type[ModelProfile]) -> type[ModelProfile]:
    """Class decorator: instantiate *profile_cls* and add it to ``PROFILES``.

    Raises ``ValueError`` if the same ``model_type`` is registered twice
    (catches copy-paste bugs early).
    """
    if not profile_cls.model_type:
        raise ValueError(f"{profile_cls.__name__} must set model_type")
    if profile_cls.model_type in PROFILES:
        raise ValueError(
            f"Profile for model_type {profile_cls.model_type!r} already registered "
            f"by {type(PROFILES[profile_cls.model_type]).__name__}"
        )
    PROFILES[profile_cls.model_type] = profile_cls()
    return profile_cls


def get(model_type: str) -> ModelProfile | None:
    return PROFILES.get(model_type)
