import torch
import torch.nn.functional as F

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


def test_return_shape_and_alignment():
    logits = torch.randn(2, 6, 11)
    ids = torch.randint(0, 11, (2, 6))
    lp = compute_per_token_log_probs(_FakeLM(logits), ids, torch.ones_like(ids))
    assert lp.shape == (2, 5)
    expected = torch.log_softmax(logits[:, :-1], dim=-1).gather(2, ids[:, 1:, None]).squeeze(-1)
    assert torch.allclose(lp, expected)


def test_gradients_match_log_softmax_composition():
    """The fused-CE form must give the same parameter gradients as the old
    log_softmax+gather composition — this pins the rolled-target trick: the
    discarded last column must not leak gradient into the real positions."""
    vocab, B, L = 13, 2, 6
    ids = torch.randint(0, vocab, (B, L))

    def _run(mode):
        torch.manual_seed(1)
        w = torch.nn.Parameter(torch.randn(vocab, 8))
        h = torch.nn.Parameter(torch.randn(B, L, 8))
        logits = h @ w.t()
        if mode == "ref":
            lp = torch.log_softmax(logits[:, :-1], dim=-1).gather(2, ids[:, 1:, None]).squeeze(-1)
        else:
            targets = torch.cat([ids[:, 1:], ids[:, :1]], dim=1)
            lp = -F.cross_entropy(
                logits.reshape(-1, vocab), targets.reshape(-1), reduction="none"
            ).view(B, L)[:, :-1]
        lp.sum().backward()
        return h.grad.clone(), w.grad.clone()

    ref_h, ref_w = _run("ref")
    new_h, new_w = _run("new")
    assert torch.allclose(ref_h, new_h, atol=1e-6)
    assert torch.allclose(ref_w, new_w, atol=1e-6)


def test_no_grad_path_does_not_build_a_graph():
    logits = torch.randn(1, 4, 9, requires_grad=True)
    ids = torch.randint(0, 9, (1, 4))
    lp = compute_per_token_log_probs(_FakeLM(logits), ids, torch.ones_like(ids), no_grad=True)
    assert lp.grad_fn is None
