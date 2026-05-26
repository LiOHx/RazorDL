"""Model-profile registry.

To add a new profile: create ``razordl/ops/model/profiles/<model_type>.py``
with a ``@register`` decorated subclass of ``ModelProfile``, then add an
explicit absolute-import line below.  The explicit import is required so the
full-mode export's AST dependency walker (which scans absolute ``razordl.*``
imports) sees the file.
"""
from razordl.ops.model.profiles.registry import (
    ModelProfile,
    PROFILES,
    UnsupportedModelError,
    get,
    register,
)

# Shipped profiles — each import triggers @register at module load.
# Keep this list alphabetical.
from razordl.ops.model.profiles import qwen2_vl  # noqa: F401
from razordl.ops.model.profiles import qwen3     # noqa: F401
from razordl.ops.model.profiles import qwen3_5   # noqa: F401

__all__ = ["ModelProfile", "PROFILES", "UnsupportedModelError", "get", "register"]

