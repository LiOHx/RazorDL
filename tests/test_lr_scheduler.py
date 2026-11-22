"""Optional cosine/warmup LR schedule: config plumbing, math, and resume."""

from types import SimpleNamespace

import pytest
import torch

from razordl.core.engine.common.flat_config import build_single_model_config_dict
from razordl.core.engine.common.lr_scheduler import (
    build_lr_scheduler,
    lr_multiplier,
    validate_scheduler_config,
    warmup_steps_for,
)
from razordl.core.engine.common.modelgroup import ParallelModelGroup


def _scheduler_config(**overrides):
    values = {
        "lr_scheduler": "none",
        "warmup_steps": 0,
        "warmup_ratio": 0.0,
        "min_lr_ratio": 0.0,
        "total_steps": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _flat_config(extra):
    return build_single_model_config_dict(
        {"model": "x", **extra},
        data_config={
            "train_data_path": "./d",
            "max_length": 8,
            "sp_size": 1,
            "dataset_processor_path": "",
        },
        model_default="x",
        processor_max_length=8,
    )


# --- validation ---------------------------------------------------------------


def test_defaults_are_disabled():
    assert validate_scheduler_config() == "none"
    assert build_lr_scheduler(torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
                              _scheduler_config()) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"lr_scheduler": "linear"},
        {"warmup_steps": -1, "lr_scheduler": "cosine"},
        {"warmup_ratio": 1.5, "lr_scheduler": "cosine"},
        {"min_lr_ratio": -0.1, "lr_scheduler": "cosine"},
        {"total_steps": 0, "lr_scheduler": "cosine"},
        {"warmup_steps": 5, "warmup_ratio": 0.1, "lr_scheduler": "cosine"},
    ],
)
def test_invalid_scheduler_config_rejected(overrides):
    with pytest.raises(ValueError):
        validate_scheduler_config(**overrides)


@pytest.mark.parametrize(
    "key,value",
    [
        ("warmup_steps", 10),
        ("warmup_ratio", 0.1),
        ("min_lr_ratio", 0.1),
        ("total_steps", 100),
    ],
)
def test_scheduler_knobs_require_cosine(key, value):
    """Dead keys would otherwise be silently ignored (same policy as the
    retired optimizer_path/scheduler_path keys)."""
    with pytest.raises(ValueError, match="lr_scheduler: cosine"):
        validate_scheduler_config(**{key: value})


# --- schedule math ------------------------------------------------------------


def test_linear_warmup_then_cosine():
    kwargs = dict(total_steps=10, warmup_steps=2, min_lr_ratio=0.0)
    assert lr_multiplier(0, **kwargs) == pytest.approx(0.5)
    assert lr_multiplier(1, **kwargs) == pytest.approx(1.0)
    assert lr_multiplier(2, **kwargs) == pytest.approx(1.0)
    # midpoint of the decay phase (step 6 of 2..10)
    assert lr_multiplier(6, **kwargs) == pytest.approx(0.5)
    assert lr_multiplier(10, **kwargs) == pytest.approx(0.0)


def test_cosine_floor_and_clamp():
    kwargs = dict(total_steps=4, warmup_steps=0, min_lr_ratio=0.1)
    assert lr_multiplier(0, **kwargs) == pytest.approx(1.0)
    assert lr_multiplier(4, **kwargs) == pytest.approx(0.1)
    # past the end stays at the floor
    assert lr_multiplier(99, **kwargs) == pytest.approx(0.1)


def test_warmup_ratio_resolves_against_total_steps():
    assert warmup_steps_for(total_steps=100, warmup_ratio=0.1) == 10
    assert warmup_steps_for(total_steps=100, warmup_steps=3, warmup_ratio=0.1) == 3
    assert warmup_steps_for(total_steps=None, warmup_ratio=0.1) == 0


def test_unresolved_total_stays_at_peak_lr():
    assert lr_multiplier(0, total_steps=None) == 1.0
    assert lr_multiplier(50, total_steps=None) == 1.0


# --- LambdaLR integration -----------------------------------------------------


def test_lambdalr_steps_match_multiplier():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    cfg = _scheduler_config(lr_scheduler="cosine", total_steps=4, warmup_steps=0)
    scheduler = build_lr_scheduler(optimizer, cfg)
    assert scheduler is not None

    observed = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])
    assert observed == pytest.approx(
        [lr_multiplier(i, total_steps=4, warmup_steps=0) for i in range(5)]
    )


# --- flat config plumbing -----------------------------------------------------


def test_flat_config_plumbs_scheduler_keys():
    cfg = _flat_config(
        {
            "lr_scheduler": "cosine",
            "warmup_ratio": 0.05,
            "min_lr_ratio": 0.1,
            "total_steps": 200,
        }
    )
    scheduler = cfg["worker_group_config"]["model_group_config"]["scheduler_config"]
    assert scheduler == {
        "lr_scheduler": "cosine",
        "warmup_steps": 0,
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.1,
        "total_steps": 200,
    }


def test_flat_config_rejects_warmup_without_cosine():
    with pytest.raises(ValueError, match="lr_scheduler: cosine"):
        _flat_config({"warmup_steps": 10})


# --- checkpoint resume --------------------------------------------------------


class _Harness:
    _completed_optimizer_steps = ParallelModelGroup._completed_optimizer_steps
    _resume_scheduler_checkpoint = ParallelModelGroup._resume_scheduler_checkpoint


def _harness(tmp_path, *, resume_dir=None, scheduler_path=None):
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    cfg = _scheduler_config(lr_scheduler="cosine", total_steps=10, warmup_steps=2)
    harness = _Harness()
    harness.config = SimpleNamespace(
        trainer_config=SimpleNamespace(
            init_from=None, resume_checkpoint_dir=resume_dir
        )
    )
    harness.model_group_config = SimpleNamespace(
        optimizer_config=SimpleNamespace(accumulate_grad_steps=2),
        scheduler_config=cfg,
    )
    harness.local_rank = 0
    harness.optimizer = optimizer
    harness.scheduler = build_lr_scheduler(optimizer, cfg)
    harness._find_checkpoint_file = lambda name: (
        str(scheduler_path)
        if name == "scheduler.pt" and scheduler_path is not None
        else None
    )
    return harness


def test_scheduler_state_roundtrip(tmp_path):
    source = _harness(tmp_path)
    for _ in range(4):
        source.optimizer.step()
        source.scheduler.step()
    expected_epoch = source.scheduler.last_epoch
    expected_lr = source.optimizer.param_groups[0]["lr"]

    state_path = tmp_path / "scheduler.pt"
    torch.save(source.scheduler.state_dict(), state_path)

    target = _harness(tmp_path, scheduler_path=state_path)
    target._resume_scheduler_checkpoint()
    assert target.scheduler.last_epoch == expected_epoch
    assert target.optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)


def test_missing_scheduler_state_fast_forwards_from_checkpoint(tmp_path):
    ckpt = tmp_path / "checkpoint_000010"
    ckpt.mkdir()
    harness = _harness(tmp_path, resume_dir=str(ckpt))
    # checkpoint_000010 with grad_accum=2 -> 5 optimizer steps completed
    assert harness._completed_optimizer_steps() == 5

    harness._resume_scheduler_checkpoint()
    assert harness.scheduler.last_epoch == 5
    assert harness.optimizer.param_groups[0]["lr"] == pytest.approx(
        lr_multiplier(5, total_steps=10, warmup_steps=2)
    )


def test_warmup_longer_than_run_rejected(tmp_path):
    harness = _harness(tmp_path)
    harness.model_group_config.scheduler_config = _scheduler_config(
        lr_scheduler="cosine", total_steps=3, warmup_steps=10
    )
    harness._scheduler_ready = False
    harness.model_group_name = "model_group"
    with pytest.raises(ValueError, match="exceeds"):
        ParallelModelGroup._activate_scheduler(harness)


def test_init_from_ignores_scheduler_state(tmp_path):
    ckpt = tmp_path / "checkpoint_000010"
    ckpt.mkdir()
    state_path = tmp_path / "scheduler.pt"
    torch.save({}, state_path)

    harness = _harness(tmp_path, resume_dir=str(ckpt), scheduler_path=state_path)
    harness.config.trainer_config.init_from = str(ckpt)
    harness._resume_scheduler_checkpoint()
    assert harness.scheduler.last_epoch == 0
