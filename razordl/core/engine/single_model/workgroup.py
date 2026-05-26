from razordl.core.engine.common.modelgroup import FSDPModelGroup
from razordl.core.engine.common.workgroup import EngineWorkGroup
from razordl.core.engine.single_model.config import Config, ModelGroupConfig, WorkerGroupConfig
from razordl.ops.model.peft import load_lora_adapter_compatible  # noqa: F401 (re-export)


class ModelGroup(FSDPModelGroup):
    """Single-model engine ModelGroup.

    The FSDP2/LoRA/resume/optimizer lifecycle lives in the shared engine
    implementation; presets only provide ``build_processor`` and ``build_model``.
    """

    config: Config
    model_group_config: ModelGroupConfig


class WorkGroup(EngineWorkGroup):
    """Single-model engine WorkGroup wrapper."""

    config: Config
    worker_group_config: WorkerGroupConfig
