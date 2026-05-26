from razordl.core.engine.common.modelgroup import FSDPModelGroup
from razordl.core.engine.on_policy_single_model.config import Config, ModelGroupConfig


class ModelGroup(FSDPModelGroup):
    """On-policy single-model ModelGroup.

    In addition to the shared FSDP2 lifecycle, this engine uses
    ``model_config.is_trainable=False`` for frozen reference models.
    """

    config: Config
    model_group_config: ModelGroupConfig
