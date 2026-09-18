"""HF-generate fallback masks: EOS stays in the response mask, pads do not."""

import pytest
import torch

from razordl.ops.model.rollout_utils import hf_generate_masks, remove_left_padding_batch

PAD, EOS, A = 0, 2, 7


def _batch():
    # prompt (P=3, left padded) + generated (L=4)
    prompt_mask = torch.tensor([[0, 1, 1], [1, 1, 1], [0, 0, 1]])
    generated = torch.tensor(
        [
            [PAD, A, A, A, EOS, PAD, PAD],  # EOS at position 1 of the response, pads after
            [A, A, A, A, A, A, A],          # never stopped
            [PAD, PAD, A, EOS, EOS, EOS, EOS],  # pad == eos style: trailing fill is EOS too
        ]
    )
    return generated, prompt_mask


def test_response_mask_keeps_first_eos_and_drops_the_rest():
    generated, prompt_mask = _batch()
    attn, resp = hf_generate_masks(generated, prompt_mask, EOS)
    assert resp.tolist() == [
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
    ]
    assert attn.tolist() == [
        [0, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 0],
    ]


def test_old_pad_rule_dropped_eos_when_pad_equals_eos():
    """Documents the bug: with pad == eos, `generated != pad` has no EOS in it."""
    generated, prompt_mask = _batch()
    old_attn = (generated != EOS).long()
    _, resp = hf_generate_masks(generated, prompt_mask, EOS)
    assert (old_attn[2] * resp[2]).sum() == 0  # third row's only response token is the EOS


def test_eos_may_be_a_list():
    generated, prompt_mask = _batch()
    _, resp_list = hf_generate_masks(generated, prompt_mask, [EOS, 99])
    _, resp_int = hf_generate_masks(generated, prompt_mask, EOS)
    assert torch.equal(resp_list, resp_int)
    _, resp_none = hf_generate_masks(generated, prompt_mask, None)
    assert resp_none[:, 3:].all()  # no EOS known -> whole response counts


def test_shape_mismatch_rejected():
    generated, prompt_mask = _batch()
    with pytest.raises(ValueError):
        hf_generate_masks(generated, prompt_mask[:2], EOS)


def test_remove_left_padding_batch():
    ids = torch.tensor([[PAD, PAD, 5, 6], [5, 6, 7, 8], [PAD, PAD, PAD, PAD]])
    assert remove_left_padding_batch(ids, PAD) == [[5, 6], [5, 6, 7, 8], [PAD] * 4]
