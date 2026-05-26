"""Profile for Qwen2-VL (model_type: ``qwen2_vl``).

Used by the ``video_embedding`` preset, whose default checkpoint
(``OpenSearch-AI/Ops-MM-embedding-v1-2B``) is a Qwen2-VL derivative.
No special config handling needed — this file declares explicit support.
"""
from razordl.ops.model.profiles.registry import ModelProfile, register


@register
class Qwen2VLProfile(ModelProfile):
    model_type = "qwen2_vl"
