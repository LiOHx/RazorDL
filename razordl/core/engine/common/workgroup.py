import functools

import torch

from razordl.core.base.workgroup import AutoSetModelGroupNameWorkGroup, BaseModelGroup
from razordl.ops.parallel.fsdp2 import load_fsdp2_model_to_gpu


class EngineWorkGroup(AutoSetModelGroupNameWorkGroup):
    """Shared update-step wrapper for engine workgroups."""

    def __init__(self, config):
        self.config = config
        self.worker_group_config = config.worker_group_config

    def _pre_update_step(self, step: int):
        import random

        import numpy as np

        seed = self.config.trainer_config.seed + step
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for _name, model_group in self.__dict__.items():
            if isinstance(model_group, BaseModelGroup):
                if model_group.model_group_config.model_config._is_offload_param:
                    load_fsdp2_model_to_gpu(model_group.model)

    def _post_update_step(self, input_dict, step: int):
        step_info = {}
        for model_group_name, model_group in self.__dict__.items():
            if isinstance(model_group, BaseModelGroup) and getattr(model_group, "optimizer", None) is not None:
                step_info[model_group_name] = model_group.update_step(step)
        return step_info

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "update_step" in cls.__dict__:
            original_update_step = cls.update_step

            @functools.wraps(original_update_step)
            def wrapper(self, input_dict, step: int, *args, **kwargs):
                step_info = {}
                self._pre_update_step(step)
                step_info.update(original_update_step(self, input_dict, step, *args, **kwargs))
                step_info.update(self._post_update_step(input_dict, step))
                return step_info

            cls.update_step = wrapper
            return

        if hasattr(cls, "_run_update_step"):
            original_run_update_step = cls._run_update_step

            @functools.wraps(original_run_update_step)
            def wrapper(self, input_dict, step: int, *args, **kwargs):
                step_info = {}
                self._pre_update_step(step)
                step_info.update(original_run_update_step(self, input_dict, step, *args, **kwargs))
                step_info.update(self._post_update_step(input_dict, step))
                return step_info

            cls.update_step = wrapper
