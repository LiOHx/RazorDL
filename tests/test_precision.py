"""Precision resolution, the use_bf16 → precision migration, and the fp16 path.

The fp16 regression tests here cover the failure that motivated the hardware
capability layer: `precision: fp16` was selectable but not trainable, because
fp16 master weights make `clip_grad_norm_` overflow to inf and zero every
gradient.  `auto` now stays on bf16 everywhere CUDA is present (emulated bf16
measured faster and lighter than the corrected fp16 path), so these tests guard
the explicit fp16 escape hatch rather than a fallback.
"""

import warnings

import pytest
import torch

from razordl.core.engine.common.flat_config import (
    _resolve_precision_key,
    build_single_model_config_dict,
)
from razordl.core.engine.single_model.config import Config
from razordl.ops.hardware import precision as prec


# --- resolve_precision ---------------------------------------------------------


@pytest.mark.parametrize(
    "available, mps_fp16, expected",
    [
        ("cuda", None, "bf16"),   # any CUDA GPU, native or emulated bf16
        ("mps", True, "fp16"),    # Apple Silicon executes fp16 natively
        ("mps", False, "fp32"),   # defensive: MPS without the fp16 probe
        ("cpu", None, "fp32"),    # no accelerator
    ],
)
def test_auto_picks_device_appropriate_dtype(monkeypatch, available, mps_fp16, expected):
    """`auto` keys off the capability probes, not a raw CUDA check: bf16 on
    any CUDA GPU, fp16 on MPS, fp32 elsewhere."""
    from types import SimpleNamespace

    monkeypatch.setattr(prec.device, "get_available_device", lambda: available)
    if available == "mps":
        monkeypatch.setattr(
            prec.device, "_backend",
            lambda name: SimpleNamespace(supports_fp16=lambda: mps_fp16),
        )
    monkeypatch.setattr(prec, "_warned_emulated_bf16", True)
    assert prec.resolve_precision("auto") == expected


@pytest.mark.parametrize("requested", ["bf16", "fp16", "fp32"])
def test_explicit_precision_is_honoured(monkeypatch, requested):
    """An explicit dtype is an escape hatch and must never be silently rewritten."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(prec.device, "supports_native_bf16", lambda: False)
    monkeypatch.setattr(prec.device, "describe", lambda: {"compute_capability": "sm_75"})
    assert prec.resolve_precision(requested) == requested


def test_emulated_bf16_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(prec.device, "supports_native_bf16", lambda: False)
    monkeypatch.setattr(prec.device, "describe", lambda: {"compute_capability": "sm_75"})
    monkeypatch.setattr(prec, "_warned_emulated_bf16", False)
    with caplog.at_level("WARNING"):
        prec.resolve_precision("bf16")
        prec.resolve_precision("bf16")
        # `auto` resolves to bf16 too, and must warn through the same guard.
        prec.resolve_precision("auto")
    # Resolved per model group, per backend wrap and once per step -- warning on
    # every call buries the rest of the log.
    assert caplog.text.count("no native bf16") == 1


def test_unknown_precision_rejected():
    with pytest.raises(ValueError, match="precision must be one of"):
        prec.resolve_precision("int8")


def test_to_torch_dtype_rejects_unresolved():
    """`auto` must be resolved first -- it is not a dtype."""
    with pytest.raises(ValueError, match="resolved precision"):
        prec.to_torch_dtype("auto")


def test_dtype_and_policy_mapping():
    assert prec.to_torch_dtype("bf16") is torch.bfloat16
    assert prec.to_torch_dtype("fp16") is torch.float16
    assert prec.to_torch_dtype("fp32") is torch.float32

    # Only fp16 needs loss scaling (bf16 has fp32's exponent range), but both
    # half precisions keep fp32 master weights and run under autocast: bf16's
    # 8 mantissa bits round away in-place updates below |w| / 256.
    assert [prec.needs_grad_scaler(p) for p in ("bf16", "fp16", "fp32")] == [False, True, False]
    assert [prec.needs_fp32_master_weights(p) for p in ("bf16", "fp16", "fp32")] == [True, True, False]
    assert [prec.needs_autocast(p) for p in ("bf16", "fp16", "fp32")] == [True, True, False]


def test_vllm_dtype_names_track_training_precision():
    """Rollout dtype must come from the same policy as training."""
    assert prec.to_vllm_dtype_name("bf16") == "bfloat16"
    assert prec.to_vllm_dtype_name("fp16") == "float16"
    assert prec.to_vllm_dtype_name("fp32") == "float32"


# --- flat config: precision key + use_bf16 migration ---------------------------


def test_precision_defaults_to_auto():
    assert _resolve_precision_key({}) == "auto"


def test_precision_key_passthrough():
    assert _resolve_precision_key({"precision": "fp32"}) == "fp32"
    assert _resolve_precision_key({"precision": "FP16"}) == "fp16"


@pytest.mark.parametrize("use_bf16, expected", [(True, "bf16"), (False, "fp16")])
def test_legacy_use_bf16_migrates_with_warning(use_bf16, expected):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _resolve_precision_key({"use_bf16": use_bf16}) == expected
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_precision_wins_over_legacy_use_bf16():
    """A stale use_bf16 left in the file must not override an explicit precision."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _resolve_precision_key({"precision": "fp32", "use_bf16": True}) == "fp32"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_bad_precision_in_flat_config_rejected():
    with pytest.raises(ValueError, match="precision must be one of"):
        _resolve_precision_key({"precision": "tf32"})


def test_precision_reaches_model_config():
    """The flat key must land on model_config -- that is what every consumer reads."""
    config_dict = build_single_model_config_dict(
        {"model": "dummy", "precision": "fp16"},
        data_config={"train_data_path": "./d", "max_length": 8, "sp_size": 1, "dataset_processor_path": ""},
        model_default="dummy",
        processor_max_length=8,
    )
    config = Config.from_dict(config_dict)
    assert config.worker_group_config.model_group_config.model_config.precision == "fp16"


# --- fp16 numerical regressions ------------------------------------------------


def test_fp16_grads_overflow_clip_to_zero():
    """Documents *why* fp32 master weights are required.

    With fp16 params, clip_grad_norm_ accumulates the total norm in fp16; once
    it passes 65504 the norm is inf, the clip coefficient is 0, and every
    gradient is silently zeroed while the optimizer step becomes a no-op.
    """
    # ||g|| = 200 * sqrt(512*512) = 102400, past fp16's 65504 ceiling.
    param = torch.nn.Parameter(torch.randn(512, 512, dtype=torch.float16))
    param.grad = torch.full_like(param, 200.0)

    total_norm = torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)

    assert torch.isinf(total_norm)
    assert bool((param.grad == 0).all())


def test_fp32_master_weights_survive_the_same_gradients():
    param = torch.nn.Parameter(torch.randn(512, 512, dtype=torch.float32))
    param.grad = torch.full_like(param, 200.0)

    total_norm = torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)

    assert torch.isfinite(total_norm)
    assert not bool((param.grad == 0).all())


def _cast_with_precision(monkeypatch, precision, trainable=True):
    """Run ParallelModelGroup's dtype cast against a tiny model, unbound."""
    from razordl.core.engine.common.modelgroup import ParallelModelGroup

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),  # trainable
        torch.nn.Linear(4, 4),  # frozen below
    ).to(torch.float16)
    for p in model[1].parameters():
        p.requires_grad = False

    config_dict = build_single_model_config_dict(
        {"model": "dummy", "precision": precision},
        data_config={"train_data_path": "./d", "max_length": 8, "sp_size": 1, "dataset_processor_path": ""},
        model_default="dummy",
        processor_max_length=8,
    )
    config = Config.from_dict(config_dict)

    class _Stub:
        model_group_config = config.worker_group_config.model_group_config
        is_trainable = trainable

    ParallelModelGroup._cast_params_to_storage_dtype(_Stub(), model)
    return model


def test_fp16_holds_every_param_in_fp32(monkeypatch):
    """Trainable params need an fp32 master copy; frozen ones follow.

    FSDP2 asserts "uniform original parameter dtype" per sharded unit, and a
    LoRA layer holds the frozen base weight beside the trainable adapter, so
    promoting only the trainable half is not representable.
    """
    model = _cast_with_precision(monkeypatch, "fp16")
    assert model[0].weight.dtype is torch.float32
    assert model[1].weight.dtype is torch.float32


def test_bf16_holds_every_param_in_fp32(monkeypatch):
    """bf16 gets the same fp32 master weights as fp16 (in-place bf16 updates
    below |w| / 256 are rounded away)."""
    model = _cast_with_precision(monkeypatch, "bf16")
    assert model[0].weight.dtype is torch.float32
    assert model[1].weight.dtype is torch.float32


def test_bf16_in_place_update_is_lost_below_mantissa_resolution():
    """Documents *why* bf16 needs master weights: lr * grad below |w| / 256
    leaves a bf16 weight untouched, while the fp32 master moves."""
    w_bf16 = torch.full((1024,), 1.0, dtype=torch.bfloat16)
    w_fp32 = torch.full((1024,), 1.0, dtype=torch.float32)
    update = 1e-3  # 1/1000 < 1/256
    w_bf16 -= update
    w_fp32 -= update
    assert torch.equal(w_bf16, torch.full((1024,), 1.0, dtype=torch.bfloat16))
    assert torch.allclose(w_fp32, torch.full((1024,), 1.0 - update))


@pytest.mark.parametrize("precision, dtype", [("fp16", torch.float16), ("bf16", torch.bfloat16)])
def test_frozen_group_stays_at_compute_dtype(monkeypatch, precision, dtype):
    """A reference/teacher group has no optimizer, so no master copy to keep."""
    model = _cast_with_precision(monkeypatch, precision, trainable=False)
    assert model[0].weight.dtype is dtype
    assert model[1].weight.dtype is dtype


def test_storage_dtype_is_fp32_for_both_half_precisions():
    from razordl.ops.hardware.precision import to_storage_dtype, to_torch_dtype

    assert to_storage_dtype("fp32") is to_torch_dtype("fp32")
    assert to_storage_dtype("fp16") is torch.float32
    assert to_storage_dtype("bf16") is torch.float32
    assert to_torch_dtype("fp16") is torch.float16
    assert to_torch_dtype("bf16") is torch.bfloat16


# --- vLLM rollout dtype ---------------------------------------------------------


@pytest.mark.parametrize(
    "precision, native_bf16, expected",
    [
        ("bf16", True, "bfloat16"),   # Ampere+: rollout matches training
        ("bf16", False, "float16"),   # Turing: training emulates bf16, vLLM cannot
        ("fp16", False, "float16"),
        ("fp32", False, "float32"),
    ],
)
def test_resolve_vllm_dtype_downgrades_emulated_bf16(monkeypatch, precision, native_bf16, expected):
    monkeypatch.setattr(prec.device, "supports_native_bf16", lambda: native_bf16)
    monkeypatch.setattr(prec.device, "describe", lambda: {"compute_capability": "sm_75"})
    assert prec.resolve_vllm_dtype_name(precision) == expected


def test_resolve_vllm_dtype_rejects_unresolved(monkeypatch):
    monkeypatch.setattr(prec.device, "supports_native_bf16", lambda: True)
    with pytest.raises(ValueError):
        prec.resolve_vllm_dtype_name("auto")
