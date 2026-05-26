import os
import random

import torch
from torch.utils.data import Dataset

from razordl.core.base.workgroup import BaseWorkGroup
from razordl.core.engine.common.main import main as _engine_main
from razordl.core.engine.on_policy_single_model.config import Config
from razordl.core.engine.on_policy_single_model.trainer import Trainer


def _configure_determinism(config: Config):
    if os.environ.get("RAZORDL_DETERMINISTIC") not in {"1", "true", "True"}:
        return

    import numpy as np

    seed = config.trainer_config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def main(config: Config, workgroup_class: BaseWorkGroup, train_dataset_class: Dataset, train_collator_class: type):
    _engine_main(
        config,
        workgroup_class,
        train_dataset_class,
        train_collator_class,
        trainer_class=Trainer,
        worker_setup=_configure_determinism,
    )


if __name__ == "__main__":
    config = Config()
    main(config, None, None, None)
