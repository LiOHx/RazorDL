"""fused_linear_tile_size config plumbing and legacy key migration."""

import pytest

from razordl.core.engine.common.flat_config import build_single_model_config_dict


def _model_config(flat_extra):
    cfg = build_single_model_config_dict(
        {"model": "x", **flat_extra},
        data_config={"train_data_path": "./d", "max_length": 8, "sp_size": 1,
                     "dataset_processor_path": ""},
        model_default="x",
        processor_max_length=8,
    )
    return cfg["worker_group_config"]["model_group_config"]["model_config"]


def test_new_key_plumbs_to_model_config():
    assert _model_config({"fused_linear_tile_size": 512})["fused_linear_tile_size"] == 512
    assert _model_config({})["fused_linear_tile_size"] == 2048


def test_legacy_chunk_size_maps_onto_new_key_with_warning():
    with pytest.warns(DeprecationWarning, match="fused_linear_tile_size"):
        mc = _model_config({"chunk_size": 333})
    assert mc["fused_linear_tile_size"] == 333


def test_legacy_chunked_loss_flag_warns_and_is_a_noop():
    with pytest.warns(DeprecationWarning, match="no-op"):
        mc = _model_config({"chunked_loss": True})
    assert mc["fused_linear_tile_size"] == 2048


def test_legacy_keys_also_emit_a_visible_log_warning(caplog):
    """warnings.warn alone is invisible under a real `razordl train` (the
    warning attributes to an imported module and CPython's default filter
    hides it) -- the remap must ALSO land in the log."""
    import logging as _logging
    with caplog.at_level(_logging.WARNING), pytest.warns(DeprecationWarning):
        _model_config({"chunk_size": 333})
    assert any("fused_linear_tile_size" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("bad", [0, -1, "abc", 2048.5])
def test_bad_fused_linear_tile_size_rejected_at_config_time(bad):
    """The operator's <=0 check is the last line of defense; config parsing
    must reject bad values with a clear message BEFORE Ray starts.  (Floats
    are coerced; the adjudicated-high finding was about 0/negative/strings
    reaching the worker and crashing or silently zeroing gradients.)"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises((ValueError, TypeError), match="fused_linear_tile_size|invalid literal"):
            _model_config({"fused_linear_tile_size": bad})
