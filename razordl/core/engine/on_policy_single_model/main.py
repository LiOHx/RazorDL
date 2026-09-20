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
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _validate_config(config: Config) -> None:
    """Reject configs the on-policy engine cannot run correctly.

    The Ulysses SP patch (FSDP2Backend.wrap_model) fires whenever
    ``sp_size > 1``, but on-policy rollout and loss never call
    ``split_for_sp`` — SP-group ranks would run attention over sp_size
    concatenated copies of the same full sequence, silently wrong on
    nonzero ranks.  SP is single-model-only until rollout splitting exists.
    """
    sp_size = getattr(config.data_config, "sp_size", 1) or 1
    if sp_size > 1:
        raise ValueError(
            f"on-policy engine (GRPO/OPD) does not support sp_size={sp_size}: "
            "rollout and loss never split the sequence for Ulysses SP, so ranks "
            "> 0 train on silently wrong attention. Use the single_model engine "
            "(SFT/DFT) for sequence parallel, or sp_size: 1."
        )


def main(config: Config, workgroup_class: BaseWorkGroup, train_dataset_class: Dataset, train_collator_class: type):
    _validate_config(config)
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
