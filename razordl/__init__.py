"""RazorDL: A flexible distributed training framework for LLMs and multimodal models."""

__version__ = "0.1.0"

# Lazy imports to avoid pulling in heavy dependencies (tensordict, torch, etc.)
# when only simple operations (e.g. CLI init) are needed.

def __getattr__(name: str):
    if name == "BaseConfig":
        from razordl.core.base.config import BaseConfig
        return BaseConfig
    if name == "BaseTrainer":
        from razordl.core.base.trainer import BaseTrainer
        return BaseTrainer
    if name == "BaseWorkGroup":
        from razordl.core.base.workgroup import BaseWorkGroup
        return BaseWorkGroup
    if name == "BaseModelGroup":
        from razordl.core.base.workgroup import BaseModelGroup
        return BaseModelGroup
    raise AttributeError(f"module 'razordl' has no attribute '{name}'")


__all__ = [
    "__version__",
    "BaseConfig",
    "BaseTrainer",
    "BaseWorkGroup",
    "BaseModelGroup",
]
