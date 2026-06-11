from __future__ import annotations

import contextlib
import os
from abc import ABC, abstractmethod
from typing import Iterator

import torch
import torch.distributed as dist

from razordl.core.base import logging
from razordl.ops.distributed.torch import get_device_id

logger = logging.getLogger(__name__)


class ParallelBackend(ABC):
    """Backend-specific distributed model lifecycle.

    Algorithm presets should never call these methods directly; they are owned
    by ``ParallelModelGroup``.
    """

    name: str

    def __init__(self, model_group):
        self.model_group = model_group
        self.config = model_group.config
        self.model_group_config = model_group.model_group_config
        self.local_rank = model_group.local_rank
        self.device = model_group.device

    @abstractmethod
    def wrap_model(self, model):
        pass

    @abstractmethod
    def save_processor(self, processor, checkpoint_dir: str):
        pass

    @abstractmethod
    def save_model(self, model, checkpoint_dir: str):
        pass

    @abstractmethod
    def save_optimizer(self, model, optimizer, checkpoint_dir: str):
        pass

    @abstractmethod
    def load_optimizer(self, model, optimizer, optimizer_path: str):
        pass

    def load_for_compute(self, model):
        return

    def load_for_optimizer_step(self, model, optimizer):
        return

    def offload_after_optimizer_step(self, model, optimizer):
        return

    @contextlib.contextmanager
    def trainer_context(self, model, optimizer) -> Iterator[None]:
        model.train()
        yield

    @contextlib.contextmanager
    def inference_context(self, model) -> Iterator[None]:
        model.eval()
        yield

    def unwrap_for_inference(self, model):
        return model

    def iter_vllm_weights(self, model, *, lora_only: bool):
        from razordl.core.engine.common.parallel_state import iter_model_weights_for_sync

        yield from iter_model_weights_for_sync(model, lora_only=lora_only)


class FSDP2Backend(ParallelBackend):
    name = "fsdp2"

    def wrap_model(self, model):
        from torch.distributed.fsdp import MixedPrecisionPolicy
        from razordl.ops.parallel.fsdp2 import create_device_mesh

        mc = self.model_group_config.model_config
        use_bf16 = mc.use_bf16
        if use_bf16:
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.model_group.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=-1)

        sp_size = getattr(mc, "sp_size", 1)
        self.model_group.sp_size = sp_size
        self.model_group.sp_group = None
        if sp_size > 1:
            from razordl.ops.parallel.sequence_parallel import create_sp_process_groups, apply_ulysses_sp

            self.model_group.sp_group = create_sp_process_groups(world_size, sp_size)
            apply_ulysses_sp(model, self.model_group.sp_group)
            if self.local_rank == 0:
                logger.info("[SP] Sequence Parallel enabled: sp_size=%s", sp_size)

        adapter_enabled = mc.adapter_config.use_adapter
        if self.model_group.is_trainable and adapter_enabled:
            if self.local_rank == 0:
                model.print_trainable_parameters()

            if mc.enable_gradient_checkpointing:
                model.enable_input_require_grads()

            from razordl.ops.parallel.fsdp2 import model_to_fsdp2_with_lora

            model_to_fsdp2_with_lora(model, self.model_group.device_mesh, mp_policy)
            self.model_group._enable_gradient_checkpointing_after_wrap(model)
        else:
            from razordl.ops.parallel.fsdp2 import model_to_fsdp2

            model_to_fsdp2(model, self.model_group.device_mesh, mp_policy)
            if mc.enable_gradient_checkpointing:
                self.model_group._enable_gradient_checkpointing_after_wrap(model)

            if getattr(mc, "enable_activation_offload", False):
                from razordl.ops.parallel.activation import enable_activation_offloading

                enable_activation_offloading(
                    model,
                    strategy="fsdp2",
                    enable_ckpt=mc.enable_gradient_checkpointing,
                )

        return model

    def save_processor(self, processor, checkpoint_dir: str):
        from razordl.ops.parallel.fsdp2 import save_processor_fsdp2

        save_processor_fsdp2(processor, checkpoint_dir)

    def save_model(self, model, checkpoint_dir: str):
        from razordl.ops.parallel.fsdp2 import save_fsdp2

        use_adapter = self.model_group_config.model_config.adapter_config.use_adapter
        save_fsdp2(
            model,
            save_dir=checkpoint_dir,
            save_lora_separately=use_adapter,
            save_full_model=not use_adapter,
        )

    def save_optimizer(self, model, optimizer, checkpoint_dir: str):
        from razordl.ops.parallel.fsdp2 import save_optimizer_fsdp2

        save_optimizer_fsdp2(model, optimizer, save_dir=checkpoint_dir)

    def load_optimizer(self, model, optimizer, optimizer_path: str):
        optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

        try:
            set_optimizer_state_dict(
                model,
                optimizer,
                optimizer_state,
                options=StateDictOptions(full_state_dict=True),
            )
            logger.info("[RESUME] FSDP2 optimizer state loaded from %s", optimizer_path)
        except Exception as e:
            state_keys = optimizer_state.get("state", {}).keys() if isinstance(optimizer_state, dict) else []
            if any(isinstance(k, str) for k in state_keys):
                raise RuntimeError("Failed to load FSDP2 optimizer checkpoint with named parameters") from e
            logger.warning(
                "[RESUME WARNING] Falling back to standard optimizer.load_state_dict for %s: %s",
                optimizer_path,
                e,
            )
            optimizer.load_state_dict(optimizer_state)
            logger.info("[RESUME] Standard optimizer state loaded from %s", optimizer_path)

    def load_for_compute(self, model):
        mc = self.model_group_config.model_config
        if mc._is_offload_param:
            from razordl.ops.parallel.fsdp2 import load_fsdp2_model_to_gpu

            load_fsdp2_model_to_gpu(model)

    def load_for_optimizer_step(self, model, optimizer):
        mc = self.model_group_config.model_config
        if mc._is_offload_optimizer:
            from razordl.ops.parallel.fsdp2 import load_fsdp_optimizer

            load_fsdp_optimizer(optimizer, get_device_id())
        self.load_for_compute(model)

    def offload_after_optimizer_step(self, model, optimizer):
        mc = self.model_group_config.model_config
        if mc._is_offload_param:
            from razordl.ops.parallel.fsdp2 import offload_fsdp2_model_to_cpu

            offload_fsdp2_model_to_cpu(model)
        if mc._is_offload_optimizer:
            from razordl.ops.parallel.fsdp2 import offload_fsdp_optimizer

            offload_fsdp_optimizer(optimizer)

    @contextlib.contextmanager
    def trainer_context(self, model, optimizer):
        self.load_for_optimizer_step(model, optimizer)
        model.train()
        try:
            yield
        finally:
            self.offload_after_optimizer_step(model, optimizer)

    @contextlib.contextmanager
    def inference_context(self, model):
        mc = self.model_group_config.model_config
        if mc._is_offload_param:
            from razordl.ops.parallel.fsdp2 import offload_fsdp2_model_to_cpu

            self.load_for_compute(model)
        model.eval()
        try:
            yield
        finally:
            if mc._is_offload_param:
                offload_fsdp2_model_to_cpu(model)


class DDPBackend(ParallelBackend):
    name = "ddp"

    def _validate_config(self):
        mc = self.model_group_config.model_config
        unsupported = []
        if mc._is_offload_param:
            unsupported.append("offload_param")
        if mc._is_offload_optimizer:
            unsupported.append("offload_optimizer")
        if getattr(mc, "enable_activation_offload", False):
            unsupported.append("enable_activation_offload")
        if getattr(mc, "sp_size", 1) != 1:
            unsupported.append("sp_size > 1")
        if unsupported:
            raise ValueError(
                "parallel_backend='ddp' does not support: " + ", ".join(unsupported)
            )

    def wrap_model(self, model):
        self._validate_config()
        mc = self.model_group_config.model_config
        model = model.to(self.device)

        if (
            self.model_group.is_trainable
            and mc.adapter_config.use_adapter
            and self.local_rank == 0
            and hasattr(model, "print_trainable_parameters")
        ):
            model.print_trainable_parameters()

        if mc.enable_gradient_checkpointing:
            if mc.adapter_config.use_adapter and hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            self.model_group._enable_gradient_checkpointing_after_wrap(model)

        if (
            self.model_group.is_trainable
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            from torch.nn.parallel import DistributedDataParallel as DDP

            kwargs = {}
            if torch.cuda.is_available():
                kwargs = {"device_ids": [self.local_rank], "output_device": self.local_rank}
            model = DDP(model, find_unused_parameters=False, **kwargs)
            if self.local_rank == 0:
                logger.info("[DDP] DistributedDataParallel enabled")
        return model

    def _rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return int(os.environ.get("LOCAL_RANK", "0"))

    def _unwrap(self, model):
        return model.module if hasattr(model, "module") else model

    def unwrap_for_inference(self, model):
        return self._unwrap(model)

    def save_processor(self, processor, checkpoint_dir: str):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self._rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            processor.save_pretrained(checkpoint_dir)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def save_model(self, model, checkpoint_dir: str):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self._rank() == 0:
            from razordl.core.engine.common.parallel_state import save_full_or_adapter_model

            save_full_or_adapter_model(
                self._unwrap(model),
                checkpoint_dir,
                save_lora_separately=self.model_group_config.model_config.adapter_config.use_adapter,
                save_full_model=not self.model_group_config.model_config.adapter_config.use_adapter,
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def save_optimizer(self, model, optimizer, checkpoint_dir: str):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self._rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def load_optimizer(self, model, optimizer, optimizer_path: str):
        optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(optimizer_state)
        logger.info("[RESUME] DDP optimizer state loaded from %s", optimizer_path)


_BACKENDS = {
    "fsdp2": FSDP2Backend,
    "ddp": DDPBackend,
}


def build_parallel_backend(name: str, model_group) -> ParallelBackend:
    try:
        cls = _BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown parallel_backend: {name!r}. Available: {sorted(_BACKENDS)}")
    return cls(model_group)
