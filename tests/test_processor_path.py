"""Processor-path resolution contracts (no model download required)."""

from razordl.ops.model.huggingface import resolve_processor_path
from razordl.presets.sft.config import SFTConfig


def _config(**flat):
    values = {"model": "base-model", "data_path": "./data"}
    values.update(flat)
    return SFTConfig.from_flat_dict(values)


def test_dataset_processor_path_becomes_canonical_processor_path():
    config = _config(dataset_processor_path="tokenizer-dir")

    assert config.data_config.dataset_processor_path == "tokenizer-dir"
    assert (
        config.worker_group_config.model_group_config.processor_config.processor_path
        == "tokenizer-dir"
    )
    assert resolve_processor_path(config) == "tokenizer-dir"


def test_explicit_processor_path_wins_over_dataset_processor_path():
    config = _config(processor_path="explicit", dataset_processor_path="legacy")

    assert (
        config.worker_group_config.model_group_config.processor_config.processor_path
        == "explicit"
    )
    assert resolve_processor_path(config) == "explicit"


def test_processor_path_defaults_to_model_path():
    config = _config()

    assert (
        config.worker_group_config.model_group_config.processor_config.processor_path
        == "base-model"
    )
    assert resolve_processor_path(config) == "base-model"
