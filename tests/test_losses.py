"""Loss-function contracts that only show up as wrong training, never as errors."""

import torch

from razordl.presets.dft.workgroup import DistDFTLoss


def test_dft_weight_is_stop_gradient():
    """Low-confidence tokens (ce > 1) must still be pushed towards the target.

    Without detach on the exp(-ce) weight the gradient factor is
    exp(-ce) * (1 - ce), negative for ce > 1: the model would be trained to
    *lower* the target probability of exactly the tokens it gets wrong.
    """
    torch.manual_seed(0)
    logits = torch.randn(4, 8) * 3.0
    labels = torch.tensor([0, 1, 2, 3])
    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    assert (ce > 1).any(), "fixture must contain low-confidence tokens"

    logits_dft = logits.clone().requires_grad_(True)
    DistDFTLoss()(logits_dft, labels).backward()

    logits_ce = logits.clone().requires_grad_(True)
    torch.nn.functional.cross_entropy(logits_ce, labels, reduction="sum").backward()

    # Same direction on the target logit for every token, including ce > 1.
    target_grad_dft = logits_dft.grad[torch.arange(4), labels]
    target_grad_ce = logits_ce.grad[torch.arange(4), labels]
    assert torch.all(target_grad_dft < 0)
    assert torch.all(torch.sign(target_grad_dft) == torch.sign(target_grad_ce))

    # And the magnitude is exactly the detached weight times the CE gradient.
    weight = torch.exp(-ce)
    expected = target_grad_ce * weight / 4
    assert torch.allclose(target_grad_dft, expected, atol=1e-6)


def _two_rank_gather(rank_values):
    """Return an all_gather_object stand-in that reports both ranks' counts."""
    state = {"calls": 0}

    def fake(x, float_mean=False):
        state["calls"] += 1
        return list(rank_values)

    return fake


def test_global_mean_denominator_matches_single_process_gradient(monkeypatch):
    """Per-rank losses averaged over ranks (and grads mean-reduced by FSDP/DDP)
    must equal the single-process global token mean; the old
    ``local_sum / global_count`` gave 1/W of both."""
    import razordl.ops.loss.distributed as dl

    torch.manual_seed(0)
    logits = [torch.randn(3, 8), torch.randn(5, 8)]        # rank 0 has 3 tokens, rank 1 has 5
    labels = [torch.randint(0, 8, (3,)), torch.randint(0, 8, (5,))]
    counts = [3, 5]

    # Reference: one process, one global token mean.
    ref_logits = [l.clone().requires_grad_(True) for l in logits]
    ref_loss = torch.nn.functional.cross_entropy(torch.cat(ref_logits), torch.cat(labels), reduction="mean")
    ref_loss.backward()

    rank_losses, rank_grads = [], []
    for r in range(2):
        monkeypatch.setattr(dl, "all_gather_object", _two_rank_gather(counts))
        lg = logits[r].clone().requires_grad_(True)
        loss = dl.DistCrossEntropyLoss()(lg, labels[r])
        loss.backward()
        rank_losses.append(loss.item())
        rank_grads.append(lg.grad)

    # Logged value: rank mean of per-rank losses.
    assert abs(sum(rank_losses) / 2 - ref_loss.item()) < 1e-6
    # Gradient: the backend averages over W ranks.
    for r in range(2):
        assert torch.allclose(rank_grads[r] / 2, ref_logits[r].grad, atol=1e-6)
