from razordl.core.engine.common.modelgroup import ParallelModelGroup
from razordl.core.engine.on_policy_single_model.config import Config, ModelGroupConfig


class ModelGroup(ParallelModelGroup):
    """On-policy single-model ModelGroup.

    In addition to the shared parallel lifecycle, this engine uses
    ``model_config.is_trainable=False`` for frozen reference models.
    """

    config: Config
    model_group_config: ModelGroupConfig
