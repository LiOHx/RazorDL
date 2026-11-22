"""LoRA target-module normalisation: all-linear, aliases, exclusion, expansion."""

import pytest
import torch.nn as nn
from peft.tuners.lora import LoraLayer
from transformers import PretrainedConfig, PreTrainedModel

from razordl.core.base.config import BaseAdapterConfig
from razordl.core.engine.common.flat_config import build_single_model_config_dict
from razordl.ops.model.lora_targets import (
    ALL_LINEAR,
    normalize_lora_exclude_modules,
    normalize_lora_targets,
)
from razordl.ops.model.peft import load_lora_adapter_compatible


# --- normalisation ------------------------------------------------------------


@pytest.mark.parametrize("value", ["all", "ALL", "all-linear", " all-linear ", ["all"], ["all-linear"]])
def test_all_aliases_become_the_peft_wildcard(value):
    assert normalize_lora_targets(value) == ALL_LINEAR


def test_comma_separated_string_becomes_a_list():
    assert normalize_lora_targets("q_proj, v_proj") == ["q_proj", "v_proj"]


def test_defaults_when_unset():
    assert normalize_lora_targets(None) == ["q_proj", "v_proj", "k_proj", "o_proj"]


@pytest.mark.parametrize(
    "value",
    ["", "   ", [], ["q_proj", "all"], "all,q_proj", 5],
)
def test_invalid_targets_rejected(value):
    with pytest.raises(ValueError):
        normalize_lora_targets(value)


def test_exclude_modules_normalisation():
    assert normalize_lora_exclude_modules(None) is None
    assert normalize_lora_exclude_modules("") is None
    assert normalize_lora_exclude_modules(".*visual.*") == ".*visual.*"
    assert normalize_lora_exclude_modules(["visual", " merger "]) == ["visual", "merger"]
    with pytest.raises(ValueError):
        normalize_lora_exclude_modules(3)


# --- flat config --------------------------------------------------------------


def _adapter_config(extra):
    cfg = build_single_model_config_dict(
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
    return cfg["worker_group_config"]["model_group_config"]["model_config"]["adapter_config"]


def test_flat_config_accepts_all_linear_and_llamafactory_alias():
    assert _adapter_config({"lora_target_modules": "all-linear"})["lora_target_modules"] == ALL_LINEAR
    assert _adapter_config({"lora_target": "all"})["lora_target_modules"] == ALL_LINEAR
    assert _adapter_config({"lora_target": "q_proj,v_proj"})["lora_target_modules"] == [
        "q_proj",
        "v_proj",
    ]


def test_flat_config_rejects_both_alias_spellings():
    with pytest.raises(ValueError, match="not both"):
        _adapter_config({"lora_target": "all", "lora_target_modules": "all"})


def test_flat_config_plumbs_exclude_modules():
    adapter = _adapter_config(
        {"lora_target_modules": "all", "lora_exclude_modules": ".*visual.*"}
    )
    assert adapter["lora_exclude_modules"] == ".*visual.*"


# --- PEFT expansion -----------------------------------------------------------


class _Config(PretrainedConfig):
    model_type = "tiny_lora_targets"


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(8, 16)
        self.linear_attn = nn.Module()
        self.linear_attn.in_proj_qkv = nn.Linear(8, 24)
        self.linear_attn.out_proj = nn.Linear(24, 8)
        self.visual = nn.Module()
        self.visual.proj = nn.Linear(8, 8)


class _TinyModel(PreTrainedModel):
    config_class = _Config

    def __init__(self):
        super().__init__(_Config())
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block()])
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


def _lora_names(wrapped):
    return {name for name, module in wrapped.named_modules() if isinstance(module, LoraLayer)}


def _adapter(**overrides):
    values = dict(
        use_adapter=True,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_target_modules=ALL_LINEAR,
        lora_exclude_modules=None,
        modules_to_save=None,
        task_type=None,
    )
    values.update(overrides)
    return BaseAdapterConfig(**values)


def test_all_linear_covers_mlp_and_hybrid_linear_attention():
    wrapped = load_lora_adapter_compatible(_TinyModel(), None, _adapter(), local_rank=0)
    names = _lora_names(wrapped)

    assert any(name.endswith("mlp.gate_proj") for name in names)
    assert any(name.endswith("linear_attn.in_proj_qkv") for name in names)
    assert any(name.endswith("linear_attn.out_proj") for name in names)
    # PEFT excludes the output head for a PreTrainedModel.
    assert not any(name.endswith("lm_head") for name in names)


def test_all_linear_can_exclude_the_vision_tower():
    wrapped = load_lora_adapter_compatible(
        _TinyModel(),
        None,
        _adapter(lora_exclude_modules=".*visual.*"),
        local_rank=0,
    )
    names = _lora_names(wrapped)
    assert names
    assert not any(".visual." in name for name in names)


def test_explicit_list_still_targets_only_those_names():
    wrapped = load_lora_adapter_compatible(
        _TinyModel(), None, _adapter(lora_target_modules=["q_proj"]), local_rank=0
    )
    names = _lora_names(wrapped)
    assert names
    assert all(name.endswith("q_proj") for name in names)


class _WrappedLinear(nn.Module):
    """Custom linear container: PEFT must target the inner ``.linear``."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)


class _WrapperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _WrappedLinear()


def test_custom_linear_wrapper_still_auto_resolves():
    wrapped = load_lora_adapter_compatible(
        _WrapperModel(), None, _adapter(lora_target_modules=["q_proj"]), local_rank=0
    )
    names = _lora_names(wrapped)
    assert any(name.endswith("q_proj.linear") for name in names)
