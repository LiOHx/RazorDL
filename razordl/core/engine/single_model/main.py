from torch.utils.data import Dataset

from razordl.core.base.workgroup import BaseWorkGroup
from razordl.core.engine.common.main import main as _engine_main
from razordl.core.engine.single_model.config import Config
from razordl.core.engine.single_model.trainer import Trainer


def main(config: Config, workgroup_class: BaseWorkGroup, train_dataset_class: Dataset, train_collator_class: type):
    _engine_main(
        config,
        workgroup_class,
        train_dataset_class,
        train_collator_class,
        trainer_class=Trainer,
    )


if __name__ == "__main__":
    config = Config()
    main(config, None, None, None)
