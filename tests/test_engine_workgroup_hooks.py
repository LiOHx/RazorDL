"""EngineWorkGroup auto-wrapping: hooks run once per outermost update_step."""

import contextlib

from razordl.core.engine.common.workgroup import EngineWorkGroup


class _Recording(EngineWorkGroup):
    def __init__(self):  # no config needed
        self.calls = []

    def _pre_update_step(self, step):
        self.calls.append("pre")

    def _post_update_step(self, input_dict, step):
        self.calls.append("post")
        return {"opt": step}

    def _autocast_context(self):
        return contextlib.nullcontext()


class _Base(_Recording):
    def update_step(self, input_dict, step):
        self.calls.append("base")
        return {"loss": 1.0}


class _Custom(_Base):
    def update_step(self, input_dict, step):
        self.calls.append("custom")
        return super().update_step(input_dict, step)


class _OnPolicyBase(_Recording):
    def _run_update_step(self, input_dict, step):
        self.calls.append("run")
        return {"loss": 2.0}


class _OnPolicyOverride(_OnPolicyBase):
    def update_step(self, input_dict, step):
        self.calls.append("custom")
        return super().update_step(input_dict, step)


def test_super_call_does_not_double_run_hooks():
    wg = _Custom()
    info = wg.update_step({}, 3)
    # Used to be ['pre', 'custom', 'pre', 'base', 'post', 'post'].
    assert wg.calls == ["pre", "custom", "base", "post"]
    assert info == {"loss": 1.0, "opt": 3}


def test_plain_subclass_runs_hooks_once():
    wg = _Base()
    wg.update_step({}, 1)
    assert wg.calls == ["pre", "base", "post"]
    wg.update_step({}, 2)  # depth counter resets between calls
    assert wg.calls == ["pre", "base", "post", "pre", "base", "post"]


def test_on_policy_override_calling_super():
    wg = _OnPolicyOverride()
    info = wg.update_step({}, 5)
    assert wg.calls == ["pre", "custom", "run", "post"]
    assert info == {"loss": 2.0, "opt": 5}


def test_depth_resets_after_an_exception():
    class _Boom(_Recording):
        def update_step(self, input_dict, step):
            raise RuntimeError("boom")

    wg = _Boom()
    try:
        wg.update_step({}, 1)
    except RuntimeError:
        pass
    assert wg._update_step_depth == 0
