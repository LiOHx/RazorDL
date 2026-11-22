"""Optional learning-rate schedule shared by every engine variant.

The default is a constant learning rate (``lr_scheduler: "none"``), which is
the behaviour RazorDL had before the scheduler existed.  ``cosine`` adds a
linear warmup followed by cosine decay to ``learning_rate * min_lr_ratio``.

Only the trainer knows the total number of optimizer steps (it depends on the
dataset length, ``num_epochs`` and gradient accumulation), so the schedule
stays unresolved until ``BaseTrainer`` calls
``set_scheduler_total_steps``.  Until then the multiplier is 1.0 -- the peak
LR -- which is exactly what optimizer step 0 should use anyway.

Steps are **optimizer steps**, not micro-batches: with
``accumulate_grad_steps > 1`` the schedule advances once per applied update.
"""

from __future__ import annotations

import math

import torch

SCHEDULER_CHOICES = ("none", "cosine")


def _as_int(value, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None


def validate_scheduler_config(
    *,
    lr_scheduler="none",
    warmup_steps=0,
    warmup_ratio=0.0,
    min_lr_ratio=0.0,
    total_steps=None,
) -> str:
    """Validate the scheduler knobs and return the normalised schedule name.

    Called both from the flat-config parser (fail before Ray starts) and from
    ``build_lr_scheduler`` (guard direct config construction).
    """
    name = str(lr_scheduler or "none").lower()
    if name not in SCHEDULER_CHOICES:
        raise ValueError(
            f"lr_scheduler must be one of {list(SCHEDULER_CHOICES)}, got {lr_scheduler!r}"
        )

    warmup_steps = _as_int(warmup_steps, name="warmup_steps")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

    warmup_ratio = float(warmup_ratio)
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1], got {warmup_ratio}")

    min_lr_ratio = float(min_lr_ratio)
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

    if total_steps is not None:
        total_steps = _as_int(total_steps, name="total_steps")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")

    if warmup_steps > 0 and warmup_ratio > 0:
        raise ValueError(
            "warmup_steps and warmup_ratio are mutually exclusive; set exactly one "
            "(warmup_steps wins when both are non-zero, which hides the other)"
        )

    if name == "none":
        configured = {
            key: value
            for key, value in {
                "warmup_steps": warmup_steps,
                "warmup_ratio": warmup_ratio,
                "min_lr_ratio": min_lr_ratio,
                "total_steps": total_steps,
            }.items()
            if value
        }
        if configured:
            raise ValueError(
                f"{sorted(configured)} require lr_scheduler: cosine "
                "(the default 'none' keeps a constant learning rate)"
            )

    return name


def warmup_steps_for(*, total_steps, warmup_steps=0, warmup_ratio=0.0) -> int:
    """Absolute warmup length in optimizer steps (``warmup_steps`` wins)."""
    if warmup_steps > 0:
        return int(warmup_steps)
    if total_steps is not None and warmup_ratio > 0:
        return int(round(float(warmup_ratio) * int(total_steps)))
    return 0


def lr_multiplier(
    step: int,
    *,
    total_steps,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> float:
    """LR multiplier at *step* (0-based optimizer steps).

    Linear warmup from ``1/warmup_steps`` to 1.0, then cosine decay to
    ``min_lr_ratio``.  The ``+1`` in the warmup branch keeps the first
    optimizer step from running at LR 0 (a wasted update).
    """
    step = int(step)
    warmup_steps = int(warmup_steps)
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)

    if total_steps is None:
        # total_steps is resolved by the trainer after the dataloader exists.
        # Until then stay at the peak LR (step 0 of any schedule is 1.0).
        return 1.0

    min_lr_ratio = min(1.0, max(0.0, float(min_lr_ratio)))
    decay_steps = max(1, int(total_steps) - warmup_steps)
    progress = (step - warmup_steps) / float(decay_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_lr_scheduler(optimizer, scheduler_config):
    """``LambdaLR`` for *scheduler_config*, or ``None`` when disabled."""
    name = validate_scheduler_config(
        lr_scheduler=getattr(scheduler_config, "lr_scheduler", "none"),
        warmup_steps=getattr(scheduler_config, "warmup_steps", 0),
        warmup_ratio=getattr(scheduler_config, "warmup_ratio", 0.0),
        min_lr_ratio=getattr(scheduler_config, "min_lr_ratio", 0.0),
        total_steps=getattr(scheduler_config, "total_steps", None),
    )
    if name == "none":
        return None

    def _multiplier(step: int) -> float:
        return lr_multiplier(
            step,
            total_steps=scheduler_config.total_steps,
            warmup_steps=warmup_steps_for(
                total_steps=scheduler_config.total_steps,
                warmup_steps=scheduler_config.warmup_steps,
                warmup_ratio=scheduler_config.warmup_ratio,
            ),
            min_lr_ratio=scheduler_config.min_lr_ratio,
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_multiplier)
