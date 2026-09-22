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
