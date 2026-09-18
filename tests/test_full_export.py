import importlib.util
import os
import re
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

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


def _presets():
    presets_dir = Path(__file__).resolve().parent.parent / "razordl" / "presets"
    return sorted(p.name for p in presets_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))


_OPTIONAL_THIRD_PARTY = {"PIL", "decord", "cv2", "qwen_vl_utils", "vllm", "av"}


@pytest.mark.parametrize("preset", _presets())
def test_full_project_runs_without_razordl_installed(tmp_path, preset):
    """A full-mode project must import and probe the device with `razordl` absent.

    ops/hardware/device.py used to load its backends through a hardcoded
    "razordl.ops.hardware.<name>" string, which the export's AST scan could
    not see (cuda.py was never shipped) and which named a package the
    exported project no longer has.
    """
    handle_init(Namespace(project_name=preset, preset=preset, path=str(tmp_path), mode="full"))
    src = tmp_path / preset / "src"
    assert (src / "ops" / "hardware" / "cuda.py").exists()

    code = (
        "import sys; sys.modules['razordl'] = None\n"
        "import main\n"
        "from ops.hardware import device\n"
        "device.get_available_device(); device.describe()\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=src, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(src)},
    )
    missing = re.findall(r"No module named '([^'.]+)", proc.stderr)
    if proc.returncode != 0 and missing and set(missing) <= _OPTIONAL_THIRD_PARTY:
        pytest.skip(f"optional dependency not installed: {missing}")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OK" in proc.stdout


def test_dft_export_does_not_carry_sft_docstring(tmp_path):
    from razordl.core.export.ast_utils import replace_class_docstring

    handle_init(Namespace(project_name="dft_full", preset="dft", path=str(tmp_path), mode="full"))
    workgroup = (tmp_path / "dft_full" / "src" / "workgroup.py").read_text()
    assert "SFT preset" not in workgroup
    assert "Base LM workgroup" in workgroup

    # Helper inserts when there is no docstring, replaces when there is.
    assert replace_class_docstring("class A:\n    x = 1\n", "doc") == 'class A:\n    """doc"""\n    x = 1'
    assert replace_class_docstring('class A:\n    """old\n    more"""\n    x = 1\n', "doc") == 'class A:\n    """doc"""\n    x = 1'
