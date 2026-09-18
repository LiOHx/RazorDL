"""Ulysses sequence-parallel contracts, run on CPU without a process group."""

import torch

from razordl.ops.parallel import sequence_parallel as sp


def _split_on_rank(monkeypatch, rank, size, **batch):
    monkeypatch.setattr(sp, "get_sp_rank", lambda: rank)
    monkeypatch.setattr(sp, "get_sp_world_size", lambda: size)
    return sp.split_for_sp(**batch)


def test_split_labels_keep_every_target_across_chunk_boundaries(monkeypatch):
    torch.manual_seed(0)
    B, S, size = 2, 12, 3
    input_ids = torch.randint(0, 50, (B, S))
    labels = input_ids.clone()
    labels[0, :4] = -100  # left padding on row 0
    mask = (labels != -100).long()

    expected = torch.nn.functional.pad(labels[:, 1:], (0, 1), value=-100)
    parts = [
        _split_on_rank(monkeypatch, r, size, input_ids=input_ids, attention_mask=mask, labels=labels)
        for r in range(size)
    ]
    # Every rank's labels line up with its *unshifted* logits, and the union
    # over ranks is exactly the single-process shifted target: shifting after
    # the split lost one label per boundary (positions 3 and 7 here).
    assert torch.equal(torch.cat([p["labels"] for p in parts], dim=1), expected)
    assert torch.equal(torch.cat([p["input_ids"] for p in parts], dim=1), input_ids)
    for r, p in enumerate(parts):
        assert p["position_ids"][0].tolist() == list(range(r * 4, r * 4 + 4))

    single = _split_on_rank(monkeypatch, 0, 1, input_ids=input_ids, attention_mask=mask, labels=labels)
    assert torch.equal(single["labels"], labels)  # untouched when SP is off
