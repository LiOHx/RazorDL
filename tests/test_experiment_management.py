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


# --- copy recovery: snapshot output_dir pin ---------------------------------------


def _flat_to_trainer_config(flat):
    config_dict = build_single_model_config_dict(
        {"model": "dummy-model", **flat},
        data_config={"train_data_path": "./data", "max_length": 16, "sp_size": 1, "dataset_processor_path": ""},
        model_default="dummy-model",
        processor_max_length=16,
    )
    return config_dict["trainer_config"]


def test_snapshot_output_dir_pin_passes_through():
    tc = _flat_to_trainer_config({"outputs_dir": "./outputs", "output_dir": "/abs/exp/2026-01-01_00-00-00"})
    assert tc["outputs_dir"] == "./outputs"
    assert tc["output_dir"] == "/abs/exp/2026-01-01_00-00-00"


def test_user_config_has_no_output_dir():
    tc = _flat_to_trainer_config({"outputs_dir": "./outputs"})
    assert tc["output_dir"] is None


def test_legacy_output_dir_migrates_to_outputs_dir():
    with pytest.warns(DeprecationWarning):
        tc = _flat_to_trainer_config({"output_dir": "./old_outputs"})
    assert tc["outputs_dir"] == "./old_outputs"
    assert tc["output_dir"] is None


# --- parallel_backend recorded in topology; switching on resume is refused ------


def test_topology_records_parallel_backend(tmp_path):
    from razordl.core.base import checkpoint_info as ckpt_info

    config = _config(tmp_path, parallel_backend="ddp")
    info = ckpt_info.build_info(
        completed_step=1, config=config, elapsed_seconds=0.0, last_step_info=None,
        resumed_from=None, ckpt_dir=str(tmp_path), kind="checkpoint",
    )
    assert info["topology"]["parallel_backend"] == "ddp"


def test_resume_across_backends_raises(tmp_path):
    from razordl.core.base import checkpoint_info as ckpt_info

    exp = tmp_path / "outputs" / "exp"
    ckpt = _fake_checkpoint(exp / "checkpoint_000010", optimizer=True)
    saved = _config(tmp_path, parallel_backend="fsdp2")
    ckpt_info.write_info(str(ckpt), ckpt_info.build_info(
        completed_step=10, config=saved, elapsed_seconds=0.0, last_step_info=None,
        resumed_from=None, ckpt_dir=str(ckpt), kind="checkpoint",
    ))

    config = _config(tmp_path, parallel_backend="ddp")
    config.trainer_config.output_dir = str(exp)
    trainer = _StubTrainer(config)
    with pytest.raises(ValueError, match="parallel_backend mismatch"):
        trainer.get_resume_state()


# --- snapshot / hash must not descend into the configured outputs dir ------------


def test_snapshot_and_hash_skip_configured_outputs_dir(tmp_path):
    from razordl.ops.snapshot import compute_code_hash, snapshot_code

    project = tmp_path / "proj"
    project.mkdir()
    (project / "config.yaml").write_text("outputs_dir: ./runs\n")
    (project / "main.py").write_text("print('hi')\n")
    runs = project / "runs"
    old_exp = runs / "2026-01-01_00-00-00"
    old_exp.mkdir(parents=True)
    (old_exp / "step_info.jsonl").write_text('{"step": 1}\n')

    h_before = compute_code_hash(str(project), exclude_paths=[str(runs)])
    exp_dir = runs / "2026-01-02_00-00-00"
    exp_dir.mkdir()
    provenance = snapshot_code(str(exp_dir), str(project), exclude_paths=[str(runs)])
    h_after = compute_code_hash(str(project), exclude_paths=[str(runs)])

    assert not (exp_dir / "code" / "runs").exists()
    assert h_before == h_after == provenance["code_hash"]
    # Without the exclusion the old experiment's jsonl would have leaked into the hash.
    assert compute_code_hash(str(project)) != h_before
