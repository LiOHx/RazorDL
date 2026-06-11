from abc import abstractmethod
import functools
import os

import torch

from razordl.core.base import logging
from razordl.core.base.workgroup import BaseModelGroup
from razordl.core.engine.common.parallel_backend import build_parallel_backend

logger = logging.getLogger(__name__)


class ParallelModelGroup(BaseModelGroup):
    """Shared model lifecycle with pluggable distributed parallel backends."""

    def __init__(self, config):
        self.config = config
        self.model_group_config = config.worker_group_config.model_group_config
        self.model_group_name = self.model_group_config.model_group_name
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = self.get_device()
        self.parallel_backend = build_parallel_backend(
            self.model_group_config.model_config.parallel_backend,
            self,
        )
        self.processor = self.build_processor()
        self.model = self.build_model()

        if self.is_trainable:
            self.optimizer = self.build_optimizer()
            self.scheduler = self.build_scheduler()
        else:
            self.optimizer = None
            self.scheduler = None
            for param in self.model.parameters():
                param.requires_grad = False
            if self.local_rank == 0:
                logger.info("[%s] Model frozen (is_trainable=False), no optimizer", self.model_group_name)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "build_model" in cls.__dict__:
            original_build_model = cls.build_model

            @functools.wraps(original_build_model)
            def wrapper(self, *args, **kwargs):
                result = original_build_model(self, *args, **kwargs)
                return self._post_build_model(result)

            cls.build_model = wrapper

    @property
    def is_trainable(self) -> bool:
        return getattr(self.model_group_config.model_config, "is_trainable", True)

    def get_device(self):
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    @abstractmethod
    def build_processor(self):
        pass

    def save_processor(self, checkpoint_dir: str):
        self.parallel_backend.save_processor(self.processor, checkpoint_dir)

    def _post_build_model(self, model):
        model = self._apply_adapter(model)
        model = self._resume_model_checkpoint(model)
        if not self.is_trainable:
            for param in model.parameters():
                param.requires_grad = False
        model = self._cast_trainable_params_to_compute_dtype(model)
        model = self.parallel_backend.wrap_model(model)
        return model

    def _apply_adapter(self, model):
        adapter_config = self.model_group_config.model_config.adapter_config
        if not adapter_config.use_adapter:
            return model
        if not self.is_trainable:
            if self.local_rank == 0:
                logger.info("[%s] Skipping LoRA for non-trainable model", self.model_group_name)
            return model

        from razordl.ops.model.peft import load_lora_adapter_compatible

        return load_lora_adapter_compatible(
            model,
            adapter_config.adapter_path,
            adapter_config,
            self.local_rank,
        )

    def _resume_model_checkpoint(self, model):
        adapter_path = self._find_checkpoint_file("adapter_model.safetensors")
        model_path = self._find_checkpoint_file("model.safetensors")

        if adapter_path:
            if self.local_rank == 0:
                logger.info("[RESUME] Preloading LoRA adapter from %s", adapter_path)

            from razordl.ops.model.peft import get_adapter_state_dict, set_adapter_state_dict

            adapter_state_dict = get_adapter_state_dict(adapter_path)
            missing, unexpected = set_adapter_state_dict(model, adapter_state_dict)
            if self.local_rank == 0:
                if unexpected:
                    logger.warning("[RESUME WARNING] Unexpected keys when loading adapter: %s", unexpected)
                logger.info(
                    "[RESUME] LoRA adapter preloaded (%s tensors)%s",
                    len(adapter_state_dict),
                    f" — {len(missing)} base-model keys skipped (expected)" if missing else "",
                )
        elif model_path:
            if self.local_rank == 0:
                logger.info("[RESUME] Preloading full model from %s", model_path)

            from safetensors.torch import load_file

            model_state_dict = load_file(model_path)
            missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
            if self.local_rank == 0:
                if missing:
                    logger.warning("[RESUME WARNING] Missing keys when loading full model: %s", missing)
                if unexpected:
                    logger.warning("[RESUME WARNING] Unexpected keys when loading full model: %s", unexpected)
                logger.info("[RESUME] Full model preloaded (%s tensors)", len(model_state_dict))
        elif self.local_rank == 0 and self.config.trainer_config.resume_checkpoint_dir:
            logger.warning("[RESUME WARNING] No model file found in checkpoint")

        return model

    def _cast_trainable_params_to_compute_dtype(self, model):
        use_bf16 = self.model_group_config.model_config.use_bf16
        if use_bf16:
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        target_dtype = torch.bfloat16 if use_bf16 else torch.float16
        for _name, param in model.named_parameters():
            if param.requires_grad and param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)
        return model

    def _enable_gradient_checkpointing_after_wrap(self, model):
        if not self.model_group_config.model_config.enable_gradient_checkpointing:
            return

        from torch.utils.checkpoint import checkpoint as ckpt_fn

        gc_func = functools.partial(ckpt_fn, use_reentrant=False)
        gc_count = 0
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                module._gradient_checkpointing_func = gc_func
                gc_count += 1
        model.train()
        if self.local_rank == 0:
            logger.info("Gradient checkpointing force-enabled on %s modules (post-wrap)", gc_count)

    @abstractmethod
    def build_model(self):
        pass

    def save_model(self, checkpoint_dir: str):
        self.parallel_backend.save_model(self.model, checkpoint_dir)

    def build_optimizer(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.model_group_config.optimizer_config.learning_rate,
            weight_decay=self.model_group_config.optimizer_config.weight_decay,
        )
        return self._resume_optimizer_checkpoint(optimizer)

    def _resume_optimizer_checkpoint(self, optimizer):
        if getattr(self.config.trainer_config, "init_from", None):
            logger.info("[INIT_FROM] Skipping optimizer state — starting fresh")
            return optimizer
        optimizer_path = self._find_checkpoint_file("optimizer.pt")
        if optimizer_path:
            try:
                self.parallel_backend.load_optimizer(self.model, optimizer, optimizer_path)
            except Exception:
                logger.exception("[RESUME] Failed to load optimizer from %s", optimizer_path)
                raise
        elif self.config.trainer_config.resume_checkpoint_dir:
            logger.warning("[RESUME WARNING] Optimizer file not found in checkpoint")
        return optimizer

    def save_optimizer(self, checkpoint_dir: str):
        if self.optimizer is not None:
            self.parallel_backend.save_optimizer(self.model, self.optimizer, checkpoint_dir)

    def build_scheduler(self):
        pass

    def save_scheduler(self, checkpoint_dir):
        pass

    def save_model_and_processor(self, checkpoint_dir: str):
        if not self.is_trainable:
            return
        self.save_processor(checkpoint_dir)
        self.save_model(checkpoint_dir)

    def save_checkpoint(self, checkpoint_dir: str):
        if not self.is_trainable:
            return
        self.save_processor(checkpoint_dir)
        self.save_model(checkpoint_dir)
        self.save_optimizer(checkpoint_dir)
        self.save_scheduler(checkpoint_dir)

    def update_step(self, step: int):
        if self.optimizer is None:
            return dict(grad_norm=0.0)

        accumulate_grad_steps = self.model_group_config.optimizer_config.accumulate_grad_steps
        grad_norm = 0.0
        if step % accumulate_grad_steps == 0 or step == -1:
            self.parallel_backend.load_for_optimizer_step(self.model, self.optimizer)
            max_grad_norm = getattr(self.model_group_config.optimizer_config, "max_grad_norm", None)
            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                grad_norm = grad_norm.item()

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.parallel_backend.offload_after_optimizer_step(self.model, self.optimizer)

        return dict(grad_norm=grad_norm)

    def trainer_context(self):
        return self.parallel_backend.trainer_context(self.model, self.optimizer)

    def inference_context(self):
        return self.parallel_backend.inference_context(self.model)

    @property
    def inference_model(self):
        return self.parallel_backend.unwrap_for_inference(self.model)

    def iter_vllm_weights(self, *, lora_only: bool):
        return self.parallel_backend.iter_vllm_weights(self.model, lora_only=lora_only)
