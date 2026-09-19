"""The on-policy engine must reject sp_size > 1 before Ray/vLLM startup.

GRPO/OPD rollout and loss never split the sequence, so with sp_size > 1 the
Ulysses patch makes SP ranks run attention over concatenated copies of the
same full sequence — silently wrong on nonzero ranks.  The check lives in
``on_policy_single_model/main.py::_validate_config`` and runs before Ray
starts, so it is testable without a cluster.
"""

import pytest

from razordl.core.engine.on_policy_single_model.main import _validate_config
from razordl.presets.grpo.config import GRPOConfig
from razordl.presets.opd.config import OPDConfig


def _flat(sp_size: int) -> dict:
    return {"model": "dummy-model", "data_path": "./data", "sp_size": sp_size}


@pytest.mark.parametrize("cls", [GRPOConfig, OPDConfig])
def test_rejects_sp_size_above_one(cls):
    config = cls.from_flat_dict(_flat(2))
    assert config.data_config.sp_size == 2
    with pytest.raises(ValueError, match="sp_size=2"):
        _validate_config(config)


@pytest.mark.parametrize("cls", [GRPOConfig, OPDConfig])
def test_accepts_default_sp_size(cls):
    config = cls.from_flat_dict(_flat(1))
    _validate_config(config)  # must not raise
