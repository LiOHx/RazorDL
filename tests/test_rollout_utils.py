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


# --- truncate_chat_prompt (needs a real chat template) ---------------------------

QWEN = "/home/lioh_wsl/Models/Qwen3.5-0.8B"


@pytest.fixture(scope="module")
def qwen_tokenizer():
    import os

    if not os.path.isdir(QWEN):
        pytest.skip(f"local tokenizer not found at {QWEN}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(QWEN)


def _generation_suffix(tokenizer, **kw):
    """Whatever the template appends for the assistant turn (Qwen3.5 adds an empty think block)."""
    text = tokenizer.apply_chat_template(_messages(1), tokenize=False, add_generation_prompt=True, **kw)
    return text[text.rfind("<|im_start|>assistant"):]


def _messages(n_words):
    return [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": " ".join(f"word{i}" for i in range(n_words))},
    ]


def test_short_prompt_is_rendered_unchanged(qwen_tokenizer):
    from razordl.ops.model.rollout_utils import truncate_chat_prompt

    msgs = _messages(5)
    direct = qwen_tokenizer.encode(
        qwen_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
        add_special_tokens=False,
    )
    assert truncate_chat_prompt(qwen_tokenizer, msgs, 256) == direct


def test_long_prompt_keeps_the_generation_prompt(qwen_tokenizer):
    from razordl.ops.model.rollout_utils import truncate_chat_prompt

    msgs = _messages(400)
    max_length = 64
    ids = truncate_chat_prompt(qwen_tokenizer, msgs, max_length)
    assert len(ids) <= max_length
    text = qwen_tokenizer.decode(ids)
    assert text.endswith(_generation_suffix(qwen_tokenizer)), text[-60:]
    assert "Answer briefly." in text
    assert "word0 word1" in text  # the head of the user text survives
    # what `encode(truncation=True)` produced: the tail (generation prompt) was cut
    old = qwen_tokenizer.encode(
        qwen_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
        add_special_tokens=False, truncation=True, max_length=max_length,
    )
    assert not qwen_tokenizer.decode(old).endswith(_generation_suffix(qwen_tokenizer))


def test_template_overhead_alone_over_budget_raises(qwen_tokenizer):
    from razordl.ops.model.rollout_utils import truncate_chat_prompt

    with pytest.raises(ValueError):
        truncate_chat_prompt(qwen_tokenizer, _messages(400), 4)


def test_grpo_prompt_never_contains_the_assistant_turn(qwen_tokenizer):
    """`answer` given *and* a trailing assistant message: the latter used to be rendered into the prompt."""
    from types import SimpleNamespace

    from razordl.presets.grpo.dataset import GRPODataset

    stub = SimpleNamespace(processor=qwen_tokenizer, max_length=256)
    item = {
        "messages": [{"role": "user", "content": "2+3?"}, {"role": "assistant", "content": "SECRET"}],
        "answer": "5",
    }
    out = GRPODataset._tokenize_prompt(stub, item)
    text = qwen_tokenizer.decode(out["prompt_ids"])
    assert "SECRET" not in text
    assert out["answer"] == "5"
    assert text.endswith(_generation_suffix(qwen_tokenizer))
    old_format = GRPODataset._tokenize_prompt(stub, {"messages": item["messages"]})
    assert old_format["answer"] == "SECRET"
