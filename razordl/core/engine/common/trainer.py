from tensordict.tensordict import TensorDict

from razordl.core.base.config import BaseConfig
from razordl.core.base.trainer import BaseTrainer
from razordl.core.base.workgroup import BaseWorkGroup


class EngineTrainer(BaseTrainer):
    """Shared trainer adapter used by engine variants."""

    def __init__(
        self,
        config: BaseConfig,
        workgroup_class: BaseWorkGroup,
        train_dataset_class,
        train_collator_class,
    ):
        self.workgroup_class = workgroup_class
        self.train_dataset_class = train_dataset_class
        self.train_collator_class = train_collator_class
        super().__init__(config)

    def prepare_data_and_workgroup(self):
        self.workgroup = self.workgroup_class(self.config)
        self.train_dataset = self.train_dataset_class(self.config)
        self.dataset_processor = self.train_dataset.dataset_processor
        sp_size = getattr(self.config.data_config, "sp_size", 1)
        self.train_collator = self.train_collator_class(self.dataset_processor, sp_size)

    def update_step(self, input_dict: TensorDict, step: int):
        return self.workgroup.update_step(input_dict, step)
