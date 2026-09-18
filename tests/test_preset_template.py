"""presets/_template must be a working preset when copied verbatim."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def package_with_new_preset(tmp_path_factory):
    """A private copy of the razordl package with _template copied to presets/new."""
    root = tmp_path_factory.mktemp("pkg")
    shutil.copytree(REPO / "razordl", root / "razordl", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(root / "razordl" / "presets" / "_template", root / "razordl" / "presets" / "new")
    if (REPO / "datasets").is_dir():
        shutil.copytree(REPO / "datasets", root / "datasets")
    return root


def _run(code, cwd, pythonpath):
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc.stdout


def test_new_preset_is_discovered_with_cli_aliases(package_with_new_preset):
    root = package_with_new_preset
    out = _run(
        "import importlib\n"
        "from razordl.cli.discovery import available_presets\n"
        "assert 'new' in available_presets(), available_presets()\n"
        "mod = importlib.import_module('razordl.presets.new')\n"
        "for cls in ('Config', 'WorkGroup', 'Dataset', 'Collator'):\n"
        "    getattr(mod, f'New{cls}')\n"
        "print(mod.ENGINE, mod.DATASET)\n",
        cwd=root, pythonpath=str(root),
    )
    assert out.split() == ["single_model", "demo_chat"]


@pytest.mark.parametrize("mode", ["custom", "full"])
def test_new_preset_inits_and_imports(package_with_new_preset, tmp_path, mode):
    root = package_with_new_preset
    _run(
        "from argparse import Namespace\n"
        "from razordl.cli.init import handle_init\n"
        f"handle_init(Namespace(project_name='proj', preset='new', path={str(tmp_path)!r}, mode={mode!r}))\n",
        cwd=root, pythonpath=str(root),
    )
    src = tmp_path / "proj" / "src"
    assert (src / "main.py").exists() and (src / "workgroup.py").exists()
    block = "import sys; sys.modules['razordl'] = None\n" if mode == "full" else ""
    pythonpath = str(src) if mode == "full" else f"{src}{os.pathsep}{root}"
    _run(block + "import main\nfrom workgroup import WorkGroup\nfrom config import NEWConfig\n"
         if mode == "full" else
         block + "import main\nfrom workgroup import WorkGroup\nfrom razordl.presets.new import NEWConfig\n",
         cwd=src, pythonpath=pythonpath)
