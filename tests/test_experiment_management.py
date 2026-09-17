"""Experiment-management contracts: init_from forks and snapshot copy recovery.

Both features were documented in engine/CLAUDE.md but did not work; these
tests pin the behaviour so the driver / worker split cannot drift again.
"""

import os

import pytest

from razordl.core.engine.common.flat_config import build_single_model_config_dict
from razordl.core.engine.common.main import _resolve_experiment
from razordl.core.engine.single_model.config import Config


def _config(tmp_path, **flat_extra):
    flat = {"model": "dummy-model", "outputs_dir": str(tmp_path / "outputs"), **flat_extra}
    config_dict = build_single_model_config_dict(
        flat,
        data_config={"train_data_path": "./data", "max_length": 16, "sp_size": 1, "dataset_processor_path": ""},
        model_default="dummy-model",
        processor_max_length=16,
    )
    return Config.from_dict(config_dict)


def _fake_checkpoint(path, *, adapter=True, optimizer=False):
    """Lay out <path>/workgroup/model_group/{adapter_model.safetensors,optimizer.pt}."""
    mg = path / "workgroup" / "model_group"
    mg.mkdir(parents=True)
    if adapter:
        (mg / "adapter_model.safetensors").write_bytes(b"")
    if optimizer:
        (mg / "optimizer.pt").write_bytes(b"")
    return path


class _StubTrainer:
    """BaseTrainer with model/data construction stubbed out.

    Only the resume-resolution methods are exercised, so the heavy
    prepare_data_and_workgroup / build_data_loader steps are replaced.
    """

    def __new__(cls, config):
        from razordl.core.base.trainer import BaseTrainer

        class Stub(BaseTrainer):
            def prepare_data_and_workgroup(self):
                pass

            def build_data_loader(self):
                return None

            def update_step(self, input_dict, step):
                return {}

        return Stub(config)


# --- init_from -----------------------------------------------------------------


def test_init_from_sets_resume_checkpoint_dir_on_worker(tmp_path):
    src = _fake_checkpoint(tmp_path / "src_ckpt")
    config = _config(tmp_path, init_from=str(src))
    config.trainer_config.output_dir = str(tmp_path / "outputs" / "new_exp")

    trainer = _StubTrainer(config)

    # The fork source survives the worker-side scan of the (empty) new experiment dir.
    assert config.trainer_config.resume_checkpoint_dir == str(src)
    # ...but step counter starts from zero.
    assert trainer.get_resume_state().completed_step == 0


def test_init_from_missing_dir_raises(tmp_path):
    config = _config(tmp_path, init_from=str(tmp_path / "nope"))
    config.trainer_config.output_dir = str(tmp_path / "outputs" / "new_exp")
    with pytest.raises(FileNotFoundError):
        _StubTrainer(config)


def test_init_from_dir_without_weights_raises(tmp_path):
    src = tmp_path / "empty_ckpt"
    src.mkdir()
    config = _config(tmp_path, init_from=str(src))
    config.trainer_config.output_dir = str(tmp_path / "outputs" / "new_exp")
    with pytest.raises(ValueError):
        _StubTrainer(config)


def test_plain_resume_still_scans_experiment_dir(tmp_path):
    exp = tmp_path / "outputs" / "exp"
    ckpt = _fake_checkpoint(exp / "checkpoint_000010", optimizer=True)
    (ckpt / "checkpoint_info.json").write_text("{}")
    config = _config(tmp_path)
    config.trainer_config.output_dir = str(exp)

    _StubTrainer(config)

    assert config.trainer_config.resume_checkpoint_dir == str(ckpt)


def test_resolve_experiment_init_from_always_creates_new_dir(tmp_path):
    outputs = tmp_path / "outputs"
    # An incomplete latest experiment that auto mode would otherwise try to resume.
    latest = outputs / "2026-01-01_00-00-00"
    latest.mkdir(parents=True)

    exp_dir, is_new = _resolve_experiment(
        str(outputs), "auto", None, str(tmp_path), init_from=str(tmp_path / "src")
    )

    assert is_new
    assert os.path.abspath(exp_dir) != os.path.abspath(latest)
    assert os.path.isdir(exp_dir)
