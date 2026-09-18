"""`razordl train` dispatch: project code first, preset from config.yaml."""

from argparse import Namespace

import pytest

from razordl.cli import train as train_cli


def test_custom_project_runs_src_main(tmp_path, monkeypatch):
    """A custom-mode project keeps its code in src/; train must run it, not the built-in preset."""
    (tmp_path / "config.yaml").write_text("preset: sft\nmodel: x\n")
    src = tmp_path / "src"
    src.mkdir()
    marker = tmp_path / "ran.txt"
    (src / "main.py").write_text(
        "import os\n"
        f"open({str(marker)!r}, 'w').write(os.path.basename(__file__))\n"
    )
    monkeypatch.setattr(train_cli, "_run_simple_mode", lambda *a: pytest.fail("built-in preset ran"))

    train_cli.handle_train(Namespace(config=str(tmp_path / "config.yaml"), preset=None))
    assert marker.read_text() == "main.py"


def test_simple_project_preset_comes_from_config(tmp_path, monkeypatch):
    """Bare `razordl train` in a GRPO project used to train SFT on the GRPO config."""
    (tmp_path / "config.yaml").write_text("preset: grpo\nmodel: x\n")
    seen = []
    monkeypatch.setattr(train_cli, "_run_simple_mode", lambda config_dir, preset: seen.append(preset))

    train_cli.handle_train(Namespace(config=str(tmp_path / "config.yaml"), preset=None))
    assert seen == ["grpo"]

    # An explicit --preset still wins.
    train_cli.handle_train(Namespace(config=str(tmp_path / "config.yaml"), preset="dft"))
    assert seen == ["grpo", "dft"]


def test_unknown_preset_in_config_is_rejected(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("preset: nope\n")
    monkeypatch.setattr(train_cli, "_run_simple_mode", lambda *a: pytest.fail("should not run"))
    with pytest.raises(SystemExit):
        train_cli.handle_train(Namespace(config=str(tmp_path / "config.yaml"), preset=None))


def test_every_default_config_names_its_preset():
    import os

    import yaml

    from razordl.cli.discovery import available_presets

    presets_dir = os.path.join(os.path.dirname(train_cli.__file__), "..", "presets")
    for preset in available_presets():
        with open(os.path.join(presets_dir, preset, "default_config.yaml")) as f:
            assert yaml.safe_load(f).get("preset") == preset, preset
