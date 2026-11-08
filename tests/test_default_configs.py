"""Every shipped default config must parse through its preset config class."""

import importlib.util
import os
import sys

import pytest
import yaml

from razordl.cli.discovery import available_presets


def _load_config_module(preset, config_path):
    # Load config.py by file path: preset __init__.py may pull optional heavy
    # dependencies (video_embedding needs PIL, for example).
    spec = importlib.util.spec_from_file_location(
        f"_razordl_preset_config_{preset}", config_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("preset", available_presets())
def test_default_config_parses(preset):
    preset_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "razordl", "presets", preset
    )
    config_path = os.path.join(preset_dir, "default_config.yaml")
    with open(config_path) as f:
        flat = yaml.safe_load(f)

    module = _load_config_module(preset, os.path.join(preset_dir, "config.py"))
    config_classes = [
        obj
        for name, obj in vars(module).items()
        if name.endswith("Config")
        and not name.startswith("_")
        and isinstance(obj, type)
        and hasattr(obj, "from_flat_dict")
    ]
    assert len(config_classes) == 1, f"expected one top-level config in {preset}/config.py"
    config = config_classes[0].from_flat_dict(flat)

    assert config.data_config.train_data_path == flat["data_path"]
    assert config.trainer_config.outputs_dir == flat["outputs_dir"]
    assert config.worker_group_config.model_group_config.model_config.model_path == flat["model"]
