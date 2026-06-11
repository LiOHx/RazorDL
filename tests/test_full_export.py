import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

from razordl.cli.init import handle_init


def _load_module(path: Path):
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


def test_full_mode_uses_canonical_config_class_names(tmp_path):
    cases = [
        ("sft", "SFTConfig"),
        ("video_embedding", "VideoEmbeddingConfig"),
        ("dft", "DFTConfig"),
    ]
    for preset, config_class in cases:
        handle_init(Namespace(
            project_name=f"{preset}_full",
            preset=preset,
            path=str(tmp_path),
            mode="full",
        ))

        project = tmp_path / f"{preset}_full"
        main_py = project / "src" / "main.py"
        config_py = project / "src" / "config.py"

        assert f"from config import {config_class}" in main_py.read_text()
        config_mod = _load_module(config_py)
        assert hasattr(config_mod, config_class)


def test_dft_config_preserves_dft_specific_field():
    from razordl.presets.dft import DFTConfig

    cfg = DFTConfig.from_flat_dict({
        "model": "dummy-model",
        "data_path": "./data",
        "dft_mini_scale": 0.25,
    })

    assert cfg.data_config.dft_mini_scale == 0.25


def test_full_export_includes_accumulation_scaled_backward(tmp_path):
    handle_init(Namespace(
        project_name="sft_full_backward",
        preset="sft",
        path=str(tmp_path),
        mode="full",
    ))

    project = tmp_path / "sft_full_backward"
    common_workgroup = (project / "src" / "engine" / "common" / "workgroup.py").read_text()
    sft_workgroup = (project / "src" / "workgroup.py").read_text()

    assert (project / "src" / "engine" / "common" / "parallel_backend.py").exists()
    assert (project / "src" / "engine" / "common" / "parallel_state.py").exists()
    assert "def _backward_loss" in common_workgroup
    assert "loss / accumulate_grad_steps" in common_workgroup
    assert "self._backward_loss(loss, self.model_group)" in sft_workgroup
