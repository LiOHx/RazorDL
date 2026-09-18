import contextlib
import functools

import torch

from razordl.core.base.workgroup import AutoSetModelGroupNameWorkGroup, BaseModelGroup


def _with_update_hooks(original):
    """Wrap an update-step method with the pre/post hooks and the autocast region.

    Guarded by a per-instance depth counter: a preset that overrides
    ``update_step`` and calls ``super().update_step()`` reaches the parent's
    wrapped method too, which used to run ``_pre_update_step`` twice and the
    optimizer step twice per batch. Only the outermost call runs the hooks;
    nested calls run the original method directly.
    """

    @functools.wraps(original)
    def wrapper(self, input_dict, step: int, *args, **kwargs):
        if getattr(self, "_update_step_depth", 0) > 0:
            return original(self, input_dict, step, *args, **kwargs)
        self._update_step_depth = 1
        try:
            step_info = {}
            self._pre_update_step(step)
            with self._autocast_context():
                step_info.update(original(self, input_dict, step, *args, **kwargs))
            step_info.update(self._post_update_step(input_dict, step))
            return step_info
        finally:
            self._update_step_depth = 0

    return wrapper


class EngineWorkGroup(AutoSetModelGroupNameWorkGroup):
    """Shared update-step wrapper for engine workgroups."""

    def __init__(self, config):
        self.config = config
        self.worker_group_config = config.worker_group_config

    def _pre_update_step(self, step: int):
        from razordl.core.base.trainer import set_seed

        set_seed(self.config.trainer_config.seed + step)

        for _name, model_group in self.__dict__.items():
            if isinstance(model_group, BaseModelGroup):
                backend = getattr(model_group, "parallel_backend", None)
                optimizer = getattr(model_group, "optimizer", None)
                if backend is not None:
                    backend.load_for_compute(model_group.model)

    def _post_update_step(self, input_dict, step: int):
        step_info = {}
        for model_group_name, model_group in self.__dict__.items():
            if isinstance(model_group, BaseModelGroup) and getattr(model_group, "optimizer", None) is not None:
                step_info[model_group_name] = model_group.update_step(step)
        return step_info

    def _autocast_context(self):
        """Autocast the forward/backward when any model group runs fp16.

        fp16 is only numerically sound under ``torch.autocast``; relying on the
        parallel backend to cast every op down produces NaN gradients at any
        loss scale (see ``ops/hardware/precision.py::needs_autocast``).  bf16 and
        fp32 get a no-op context.

        Engine-level on purpose: the forward lives in preset code, so this is
        the only layer that can wrap every preset without each of them opting
        in.  ``_post_update_step`` (optimizer step, clipping) stays outside.
        """
        from razordl.ops.hardware.precision import (
            needs_autocast,
            resolve_precision,
            to_torch_dtype,
        )

        # Resolved once and cached: this runs on every training step.
        if not hasattr(self, "_autocast_dtype"):
            self._autocast_dtype = None
            if torch.cuda.is_available():
                for _name, model_group in self.__dict__.items():
                    if not isinstance(model_group, BaseModelGroup):
                        continue
                    precision = resolve_precision(
                        model_group.model_group_config.model_config.precision
                    )
                    if needs_autocast(precision):
                        self._autocast_dtype = to_torch_dtype(precision)
                        break

        if self._autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self._autocast_dtype)

    def _backward_loss(self, loss, model_group: BaseModelGroup):
        """Backward with standard gradient-accumulation scaling."""
        accumulate_grad_steps = model_group.model_group_config.optimizer_config.accumulate_grad_steps
        if accumulate_grad_steps < 1:
            raise ValueError(f"accumulate_grad_steps must be >= 1, got {accumulate_grad_steps}")
        # scale_loss is a no-op unless fp16 loss scaling is active.
        model_group.scale_loss(loss / accumulate_grad_steps).backward()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "update_step" in cls.__dict__:
            cls.update_step = _with_update_hooks(cls.update_step)
            return

        if hasattr(cls, "_run_update_step"):
            cls.update_step = _with_update_hooks(cls._run_update_step)
