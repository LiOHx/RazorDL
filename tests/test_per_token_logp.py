import torch

from razordl.ops.model.per_token_logp import compute_per_token_log_probs


class _Logits:
    def __init__(self, logits):
        self.logits = logits


class _FakeLM(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self._logits = logits

    def forward(self, input_ids, attention_mask):
        return _Logits(self._logits)


def test_temperature_divides_logits_before_log_softmax():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    ids = torch.randint(0, 7, (2, 5))
    model = _FakeLM(logits)

    lp = compute_per_token_log_probs(model, ids, torch.ones_like(ids), temperature=0.5)
    expected = torch.log_softmax(logits[:, :-1] / 0.5, dim=-1).gather(2, ids[:, 1:, None]).squeeze(-1)
    assert torch.allclose(lp, expected)

    # T=1 is the plain log-softmax; tempered values differ, so the rollout
    # temperature is not silently ignored.
    plain = compute_per_token_log_probs(model, ids, torch.ones_like(ids))
    assert not torch.allclose(lp, plain)
