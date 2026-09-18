"""Policy -> vLLM weight sync happens once per optimizer step, not once per batch.

`ModelGroup.update_step` steps the optimizer when `step % accumulate_grad_steps
== 0`, so with grad accumulation the weights the rollout sees are new only on
the step right after that. Fake engine on a stub ModelGroup; no vLLM here.
"""

from types import SimpleNamespace

import pytest

from razordl.presets.grpo.workgroup import GRPOPolicyModelGroup
from razordl.presets.opd.workgroup import OPDPolicyModelGroup


class _FakeEngine:
    def __init__(self):
        self.calls = []

    def update_weights(self, weights, peft_config=None):
        self.calls.append(list(weights))


def _stub(cls, accumulate_grad_steps):
    mg = cls.__new__(cls)
    mg._vllm_engine = _FakeEngine()
    mg._vllm_synced_step = None
    mg.model_group_config = SimpleNamespace(
        model_config=SimpleNamespace(adapter_config=SimpleNamespace(use_adapter=False)),
        optimizer_config=SimpleNamespace(accumulate_grad_steps=accumulate_grad_steps),
    )
    mg.iter_vllm_weights = lambda lora_only=False: iter(())
    return mg


def _rollout_steps(mg, steps):
    synced = []
    for step in steps:
        before = len(mg._vllm_engine.calls)
        with mg.vllm_rollout_context(step) as engine:
            assert engine is mg._vllm_engine
        if len(mg._vllm_engine.calls) > before:
            synced.append(step)
    return synced


@pytest.mark.parametrize("cls", [GRPOPolicyModelGroup, OPDPolicyModelGroup])
def test_sync_only_after_optimizer_steps(cls):
    mg = _stub(cls, accumulate_grad_steps=8)
    # optimizer steps at 8 and 16 -> new weights for rollouts 9 and 17
    assert _rollout_steps(mg, range(1, 17)) == [1, 9]
    assert len(mg._vllm_engine.calls) == 2


@pytest.mark.parametrize("cls", [GRPOPolicyModelGroup, OPDPolicyModelGroup])
def test_no_accumulation_syncs_every_step(cls):
    mg = _stub(cls, accumulate_grad_steps=1)
    assert _rollout_steps(mg, range(1, 5)) == [1, 2, 3, 4]


@pytest.mark.parametrize("cls", [GRPOPolicyModelGroup, OPDPolicyModelGroup])
def test_resumed_run_syncs_on_its_first_rollout(cls):
    """A checkpoint is loaded after `_init_vllm_engine`, so the first rollout
    of a resumed process must push the restored weights even mid-accumulation."""
    mg = _stub(cls, accumulate_grad_steps=8)
    assert _rollout_steps(mg, range(11, 19)) == [11, 17]
