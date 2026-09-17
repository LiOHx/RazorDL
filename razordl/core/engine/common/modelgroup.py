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
        self.scaler = None  # fp16 loss scaler; set by build_optimizer()
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
        return self.model_group_config.model_config.is_trainable

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
        model = self._cast_params_to_storage_dtype(model)
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
        elif self.config.trainer_config.resume_checkpoint_dir:
            if self.config.trainer_config.init_from:
                # A fork that finds no weights would silently train from the
                # base model.  Typical cause: forking across presets whose
                # model_group_name differs (model_group vs policy_model_group).
                raise RuntimeError(
                    f"[INIT_FROM] No model/adapter file for model group "
                    f"{self.model_group_name!r} under "
                    f"{self.config.trainer_config.resume_checkpoint_dir}"
                )
            if self.local_rank == 0:
                logger.warning("[RESUME WARNING] No model file found in checkpoint")

        return model

    def _cast_params_to_storage_dtype(self, model):
        """Make every parameter share the precision's storage dtype.

        Storage dtype equals the compute dtype except under fp16, where it is
        fp32: fp16 master weights lose updates below fp16 relative precision
        (lr=5e-5 sits right at that boundary) and make ``clip_grad_norm_``
        compute the total norm in fp16, which silently zeroes every gradient
        once it overflows 65504.  FSDP2's
        ``MixedPrecisionPolicy(param_dtype=fp16)`` casts down for compute, so
        the fp32 shard costs memory, not speed.

        Within a trainable group the frozen params are promoted too, even
        though they need no master copy: FSDP2 asserts on mixed dtypes within a
        sharded unit, and a LoRA layer keeps the frozen base weight next to the
        trainable adapter.  A wholly frozen group (reference / teacher model)
        has no optimizer at all, so it stays at the compute dtype -- uniform,
        and half the memory.
        """
        from razordl.ops.hardware.precision import (
            resolve_precision,
            to_storage_dtype,
            to_torch_dtype,
        )

        precision = resolve_precision(self.model_group_config.model_config.precision)
        target_dtype = (
            to_storage_dtype(precision) if self.is_trainable else to_torch_dtype(precision)
        )
        for _name, param in model.named_parameters():
            if param.is_floating_point() and param.dtype != target_dtype:
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
        self._build_grad_scaler()
        return self._resume_optimizer_checkpoint(optimizer)

    def _build_grad_scaler(self):
        """Create the fp16 loss scaler, or leave it None for bf16 / fp32.

        fp16 gradients underflow to zero without loss scaling. bf16 has the
        same exponent range as fp32 and needs none.
        """
        from razordl.ops.hardware.precision import needs_grad_scaler, resolve_precision

        precision = resolve_precision(self.model_group_config.model_config.precision)
        if not needs_grad_scaler(precision):
            self.scaler = None
            return

        self.scaler = torch.amp.GradScaler("cuda")
        if self.local_rank == 0:
            logger.info(
                "[%s] fp16: GradScaler enabled (initial scale %.0f)",
                self.model_group_name,
                self.scaler.get_scale(),
            )

    def scale_loss(self, loss):
        """Scale *loss* before backward when fp16 loss scaling is active.

        Returns *loss* unchanged for bf16 / fp32 so callers need no branching.
        """
        scaler = getattr(self, "scaler", None)
        return loss if scaler is None else scaler.scale(loss)

    def _resume_optimizer_checkpoint(self, optimizer):
        if self.config.trainer_config.init_from:
            logger.info("[INIT_FROM] Skipping optimizer state — starting fresh")
            return optimizer
        self._resume_grad_scaler()
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
        self.save_grad_scaler(checkpoint_dir)

    def save_grad_scaler(self, checkpoint_dir: str):
        """Persist the fp16 scale factor next to the optimizer state.

        Without it a resumed run restarts from the initial scale and burns
        several steps re-probing for the right one.
        """
        if self.scaler is None or self.local_rank != 0:
            return
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.scaler.state_dict(), os.path.join(checkpoint_dir, "scaler.pt"))

    def _resume_grad_scaler(self):
        """Restore the scaler state; a checkpoint without one is not an error."""
        if self.scaler is None:
            return
        scaler_path = self._find_checkpoint_file("scaler.pt")
        if not scaler_path:
            return
        try:
            self.scaler.load_state_dict(torch.load(scaler_path, map_location="cpu"))
            if self.local_rank == 0:
                logger.info("[RESUME] GradScaler scale restored: %.0f", self.scaler.get_scale())
        except Exception:
            logger.exception("[RESUME] Failed to load GradScaler state from %s", scaler_path)
            raise

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
        grad_overflow = False
        if step % accumulate_grad_steps == 0 or step == -1:
            self.parallel_backend.load_for_optimizer_step(self.model, self.optimizer)

            # Unscale before clipping: clip_grad_norm_ must see true gradient
            # magnitudes, and unscale_ may be called only once per optimizer
            # step -- which the accumulation gate above already guarantees.
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            max_grad_norm = getattr(self.model_group_config.optimizer_config, "max_grad_norm", None)
            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                grad_norm = grad_norm.item()

            if self.scaler is not None:
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # A dropped scale means the step was skipped on non-finite
                # grads. Surface it: otherwise this looks exactly like a loss
                # curve that simply refuses to move.
                grad_overflow = self.scaler.get_scale() < scale_before
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.parallel_backend.offload_after_optimizer_step(self.model, self.optimizer)

        return dict(grad_norm=grad_norm, grad_overflow=grad_overflow)

    def trainer_context(self):
        return self.parallel_backend.trainer_context(self.model, self.optimizer)

    def inference_context(self):
        return self.parallel_backend.inference_context(self.model)

    @property
    def inference_model(self):
        return self.parallel_backend.unwrap_for_inference(self.model)

    def iter_vllm_weights(self, *, lora_only: bool):
        return self.parallel_backend.iter_vllm_weights(self.model, lora_only=lora_only)
