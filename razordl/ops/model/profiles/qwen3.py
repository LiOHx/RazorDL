"""Profile for Qwen3 (model_type: ``qwen3``).

No special handling needed — transformers loads Qwen3 correctly out of the
box.  This file exists solely to declare that Qwen3 is in the supported set,
so ``enforce_model_profile`` lets it through.
"""
from razordl.ops.model.profiles.registry import ModelProfile, register


@register
class Qwen3Profile(ModelProfile):
    model_type = "qwen3"
