"""SFT/DFT losses stream through FusedLinearCrossEntropy with identical math.

Each test compares the workgroup's operator path against the criterion's
full-logits forward on the same fake model -- the reduction contract is
``criterion.reduce_per_token_ce(ce_per_token, labels)``.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from razordl.ops.loss.distributed import DistCrossEntropyLoss


class _FakeLM(torch.nn.Module):
    def __init__(self, seed=0, vocab=29, dim=8, softcap=None):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab, bias=False)
        self.config = SimpleNamespace(final_logit_softcapping=softcap)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                logits_to_keep=0, position_ids=None):
        hidden = self.emb(input_ids)
        logits = self.head(hidden)
        if logits_to_keep:
            logits = logits[:, -int(logits_to_keep):]
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,) if output_hidden_states else None,
        )

    def get_output_embeddings(self):
        return self.head


def _stub_workgroup(criterion, tile=3, softcap=None):
    from razordl.presets.sft.workgroup import SFTWorkGroup

    wg = SFTWorkGroup.__new__(SFTWorkGroup)
    wg.criterion = criterion
    wg.fused_linear_tile_size = tile
    return wg


def _batch(seed=0, vocab=29):
    torch.manual_seed(seed)
    ids = torch.randint(0, vocab, (2, 6))
    labels = ids.clone()
    labels[0, :2] = -100  # pad positions
    return ids, labels


def _full_logits_ref(fake, ids, labels, criterion):
    logits = fake(ids).logits
    return criterion(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))


def test_sft_ce_parity_non_shifted():
    fake = _FakeLM(0)
    ids, labels = _batch()
    wg = _stub_workgroup(DistCrossEntropyLoss(ignore_index=-100))

    actual = wg._compute_loss(
        fake, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, labels
    )
    expected = _full_logits_ref(fake, ids, labels, wg.criterion)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_sft_ce_parity_shifted_sp_convention():
    """SP path: labels arrive pre-shifted (split_for_sp convention) and every
    position is an aligned target -- no roll, no trailing slice."""
    fake = _FakeLM(1)
    ids, _ = _batch(seed=1)
    torch.manual_seed(11)
    shifted_labels = torch.randint(0, 29, (2, 6))
    shifted_labels[1, -2:] = -100
    wg = _stub_workgroup(DistCrossEntropyLoss(ignore_index=-100))

    actual = wg._compute_loss(
        fake, {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
        shifted_labels, shifted=True,
    )
    expected = wg.criterion(
        fake(ids).logits.reshape(-1, 29), shifted_labels.reshape(-1)
    )
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_sft_ce_parity_with_softcap():
    fake = _FakeLM(2, softcap=15.0)
    ids, labels = _batch(seed=2)
    wg = _stub_workgroup(DistCrossEntropyLoss(ignore_index=-100))
    actual = wg._compute_loss(
        fake, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, labels
    )
    # reference: softcap applied to the full logits (the criterion itself
    # has no softcap -- it lives in _compute_loss, from model.config)
    logits = fake(ids).logits[:, :-1]
    logits = torch.tanh(logits / 15.0) * 15.0
    ce = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
        ignore_index=-100, reduction="none",
    ).view(labels.size(0), -1)
    expected = wg.criterion.reduce_per_token_ce(ce, labels[:, 1:])
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_dft_weighted_loss_parity():
    from razordl.presets.dft.workgroup import DistDFTLoss

    fake = _FakeLM(3)
    ids, labels = _batch(seed=3)
    wg = _stub_workgroup(DistDFTLoss(ignore_index=-100, mini_scale=0.1))
    actual = wg._compute_loss(
        fake, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, labels
    )
    expected = _full_logits_ref(fake, ids, labels, wg.criterion)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_losses_backward_through_operator():
    """Gradients reach the fake's parameters through the streaming path."""
    fake = _FakeLM(4)
    ids, labels = _batch(seed=4)
    wg = _stub_workgroup(DistCrossEntropyLoss(ignore_index=-100))
    loss = wg._compute_loss(
        fake, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, labels
    )
    loss.backward()
    assert fake.head.weight.grad is not None
    assert fake.emb.weight.grad is not None
    assert torch.isfinite(fake.head.weight.grad).all()
