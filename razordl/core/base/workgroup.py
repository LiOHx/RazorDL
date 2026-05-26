from abc import ABC, abstractmethod
from razordl.core.base.config import BaseConfig, BaseModelGroupConfig
import os
import torch.distributed as dist


__all__ = ["BaseModelGroup", "BaseWorkGroup"]



class BaseModelGroup(ABC):
    
    @abstractmethod
    def __init__(self, config: BaseConfig):
        pass


    @abstractmethod
    def get_device(self):
        pass
    
    
    @abstractmethod
    def build_processor(self):
        pass


    @abstractmethod
    def save_processor(self, checkpoint_dir: str):
        pass


    @abstractmethod
    def build_model(self):
        pass


    @abstractmethod
    def save_model(self, checkpoint_dir: str):
        pass


    @abstractmethod
    def build_optimizer(self):
        pass


    @abstractmethod
    def save_optimizer(self, checkpoint_dir: str):
        pass    
    
    @abstractmethod
    def build_scheduler(self):
        pass


    @abstractmethod
    def save_scheduler(self, checkpoint_dir: str):
        pass

    def _find_checkpoint_file(self, filename: str) -> str | None:
        """Find *filename* inside the resume checkpoint directory tree.

        If ``self.model_group_name`` is set, only that subdirectory is
        eligible.  This avoids loading policy checkpoints into reference
        models in multi-model trainers.  Unnamed single-model trainers fall
        back to the first match that also contains ``optimizer.pt``.
        """
        ckpt_dir = getattr(
            getattr(self, "config", None), "trainer_config", None
        )
        if ckpt_dir is None:
            return None
        ckpt_dir = ckpt_dir.resume_checkpoint_dir
        if not ckpt_dir:
            return None
        first = None
        with_optimizer = None
        mg_name = getattr(self, "model_group_name", None)
        for root, _dirs, files in os.walk(ckpt_dir):
            if filename in files:
                candidate = os.path.join(root, filename)
                if mg_name:
                    if mg_name in root.split(os.sep):
                        return candidate
                    continue
                else:
                    if first is None:
                        first = candidate
                    if with_optimizer is None and "optimizer.pt" in files:
                        with_optimizer = candidate
        if mg_name:
            return None
        return with_optimizer or first


    @abstractmethod
    def save_model_and_processor(self, checkpoint_dir: str):
        pass


    @abstractmethod
    def save_checkpoint(self, checkpoint_dir: str):
        pass



class BaseWorkGroup(ABC):

    @abstractmethod
    def __init__(self, config: BaseConfig):
        pass
            

    @abstractmethod
    def update_step(self, input_dict) -> dict:
        pass


    def save_model_and_processor(self, checkpoint_dir: str):
        _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model_group_fields = [
            name
            for name in dir(self)
            if not name.startswith('__') and
                isinstance(getattr(self, name), BaseModelGroup)
        ]
        for model_group_field in model_group_fields:
            model_group: BaseModelGroup = getattr(self, model_group_field)
            model_group_checkpoint_dir = os.path.join(checkpoint_dir, model_group_field)
            if _local_rank == 0:
                os.makedirs(model_group_checkpoint_dir, exist_ok=True)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            model_group.save_model_and_processor(model_group_checkpoint_dir)

    def save_checkpoint(self, checkpoint_dir: str):
        _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model_group_fields = [
            name
            for name in dir(self)
            if not name.startswith('__') and
                isinstance(getattr(self, name), BaseModelGroup)
        ]
        for model_group_field in model_group_fields:
            model_group: BaseModelGroup = getattr(self, model_group_field)
            model_group_checkpoint_dir = os.path.join(checkpoint_dir, model_group_field)
            if _local_rank == 0:
                os.makedirs(model_group_checkpoint_dir, exist_ok=True)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            model_group.save_checkpoint(model_group_checkpoint_dir)



class AutoSetModelGroupNameWorkGroup(BaseWorkGroup):

    def __post_init__(self):
        self.auto_set_model_group_name()


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Goal: __post_init__ fires exactly once per instance, AFTER the
        # outermost __init__ in the chain returns.
        #
        # The pitfall: a subclass's __init__ may call super().__init__()
        # before assigning its own attributes (e.g. SFTWorkGroup creates
        # self.model_group AFTER super().__init__()).  If the inner wrapper
        # fires __post_init__ right after super() returns, the model_group
        # attribute does not yet exist and auto_set_model_group_name() walks
        # an empty set.
        #
        # Solution: wrap every level, but use a per-instance depth counter
        # to dispatch __post_init__ only when the outermost wrapper returns.
        current_init = cls.__init__
        if getattr(current_init, "_auto_post_init_wrapped", False):
            return

        def wrapped_init(self, *args, **kwargs):
            depth = getattr(self, "_auto_post_init_depth", 0)
            self._auto_post_init_depth = depth + 1
            try:
                current_init(self, *args, **kwargs)
            finally:
                self._auto_post_init_depth -= 1
            if self._auto_post_init_depth == 0:
                self.__post_init__()

        wrapped_init._auto_post_init_wrapped = True
        cls.__init__ = wrapped_init


    def auto_set_model_group_name(self):
        model_group_fields = [
            name 
            for name in dir(self) 
            if not name.startswith('__') and 
                isinstance(getattr(self, name), BaseModelGroup)
        ]
        for model_group_field in model_group_fields:
            model_group = getattr(self, model_group_field)
            model_group.model_group_name = model_group_field
