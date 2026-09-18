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
