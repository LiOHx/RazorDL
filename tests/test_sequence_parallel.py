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


# ---------------------------------------------------------------------------
# Two-thread emulation of an SP group (gloo has no all_to_all and one GPU
# cannot host two NCCL ranks, so the collectives are faked with a barrier).
# ---------------------------------------------------------------------------
import copy
import threading

import pytest


class _FakeDist:
    """Stand-in for torch.distributed: each thread is one rank."""

    def __init__(self, world_size):
        self.world_size = world_size
        self._local = threading.local()
        self._barrier = threading.Barrier(world_size, timeout=60)
        self._slots = {}

    def set_rank(self, rank):
        self._local.rank = rank

    def get_rank(self, group=None):
        return self._local.rank

    def get_world_size(self, group=None):
        return self.world_size

    def _exchange(self, payload):
        self._slots[self.get_rank()] = payload
        self._barrier.wait()
        everyone = [self._slots[r] for r in range(self.world_size)]
        self._barrier.wait()  # slots may be reused after everyone has read
        return everyone

    def all_to_all(self, out_list, in_list, group=None):
        me = self.get_rank()
        for src, chunks in enumerate(self._exchange(in_list)):
            out_list[src].copy_(chunks[me])

    def all_gather(self, out_list, tensor, group=None):
        for src, t in enumerate(self._exchange(tensor)):
            out_list[src].copy_(t)

    def reduce_scatter(self, output, in_list, group=None, op=None):
        me = self.get_rank()
        output.copy_(sum(chunks[me] for chunks in self._exchange(in_list)))


def _run_sp_ranks(monkeypatch, model, batch, world_size=2):
    """Run ``model`` under SP on ``world_size`` threads; return per-rank outputs."""
    fake = _FakeDist(world_size)
    monkeypatch.setattr(sp, "dist", fake)
    labels = batch["labels"]
    n_valid = int((labels[:, 1:] != -100).sum())
    results, errors = {}, {}

    def worker(rank):
        try:
            fake.set_rank(rank)
            local = copy.deepcopy(model)
            sp.apply_ulysses_sp(local, sp_group=object())
            inputs = sp.split_for_sp(batch["input_ids"], batch["attention_mask"], labels)
            local_labels = inputs.pop("labels")
            logits = local(**inputs).logits
            # A5 convention: local_sum / (global_count / W) so the DP
            # mean-reduce of the gradients equals the global token mean.
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), local_labels.reshape(-1),
                ignore_index=-100, reduction="sum",
            ) / (n_valid / world_size)
            loss.backward()
            results[rank] = {
                "logits": logits.detach(),
                "grads": {n: p.grad.detach().clone() for n, p in local.named_parameters() if p.grad is not None},
            }
        except BaseException as exc:  # noqa: BLE001 - surfaced in the main thread
            errors[rank] = exc
            fake._barrier.abort()

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[min(errors)]
    return [results[r] for r in range(world_size)]


def _reference(model, batch):
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    labels = batch["labels"]
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1),
        ignore_index=-100, reduction="mean",
    )
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return logits.detach(), grads


def _left_padded_batch(vocab, B=2, S=16, pad=5):
    torch.manual_seed(1)
    input_ids = torch.randint(3, vocab, (B, S))
    mask = torch.ones(B, S, dtype=torch.long)
    input_ids[0, :pad] = 0
    mask[0, :pad] = 0
    labels = input_ids.masked_fill(mask == 0, -100)
    # The first real token is predicted from the last *pad* position, whose
    # hidden state is implementation-defined (HF eager lets a fully masked
    # row attend uniformly; SP lets it attend to itself).  Real SFT data
    # masks the prompt anyway, so leave that target out of the comparison.
    labels[0, pad] = -100
    return {"input_ids": input_ids, "attention_mask": mask, "labels": labels}


def _tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=128, tie_word_embeddings=False, attn_implementation="eager",
    )
    return Qwen3ForCausalLM(cfg).float().eval()


def _assert_sp_matches_reference(monkeypatch, model, batch):
    ref_logits, ref_grads = _reference(copy.deepcopy(model), batch)
    ranks = _run_sp_ranks(monkeypatch, model, batch)

    sp_logits = torch.cat([r["logits"] for r in ranks], dim=1)
    valid = batch["attention_mask"].bool()
    assert torch.isfinite(sp_logits).all()
    assert torch.allclose(sp_logits[valid], ref_logits[valid], atol=1e-4), (
        (sp_logits[valid] - ref_logits[valid]).abs().max()
    )
    # What FSDP's mean-reduce would see: the rank-average of the grads.
    for name, ref in ref_grads.items():
        got = sum(r["grads"][name] for r in ranks) / len(ranks)
        assert torch.allclose(got, ref, atol=1e-4), (name, (got - ref).abs().max())


def test_ulysses_sp_matches_unsplit_forward_with_left_padding(monkeypatch):
    model = _tiny_qwen3()
    batch = _left_padded_batch(model.config.vocab_size)
    _assert_sp_matches_reference(monkeypatch, model, batch)


def test_ulysses_sp_matches_unsplit_forward_without_padding(monkeypatch):
    model = _tiny_qwen3()
    batch = _left_padded_batch(model.config.vocab_size, pad=0)
    _assert_sp_matches_reference(monkeypatch, model, batch)


def test_ulysses_sp_chunked_fallback_with_left_padding(monkeypatch):
    """Same check when SDPA rejects the mask and the online-softmax path runs."""
    def _no_sdpa(*args, **kwargs):
        raise RuntimeError("no kernel")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", _no_sdpa)
    real_chunked = sp._chunked_causal_attention
    monkeypatch.setattr(
        sp, "_chunked_causal_attention",
        lambda q, k, v, scaling, chunk_size=2048, key_valid=None:
            real_chunked(q, k, v, scaling, chunk_size=4, key_valid=key_valid),
    )
    model = _tiny_qwen3()
    batch = _left_padded_batch(model.config.vocab_size)
    _assert_sp_matches_reference(monkeypatch, model, batch)


def _tiny_qwen3_5(full_attention_interval=4, num_hidden_layers=4):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=num_hidden_layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=128,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4,
        full_attention_interval=full_attention_interval,
        tie_word_embeddings=False, attn_implementation="eager",
    )
    return Qwen3_5ForCausalLM(cfg).float().eval()


def test_ulysses_sp_gated_attention_matches_unsplit_forward(monkeypatch):
    """Qwen3.5 attention: the output gate is split per head, not in flat halves."""
    model = _tiny_qwen3_5(full_attention_interval=1, num_hidden_layers=2)
    assert set(model.config.layer_types) == {"full_attention"}
    batch = _left_padded_batch(model.config.vocab_size)
    _assert_sp_matches_reference(monkeypatch, model, batch)


@pytest.mark.parametrize("pad", [0, 5])
def test_ulysses_sp_gated_delta_net_matches_unsplit_forward(monkeypatch, pad):
    """Qwen3.5: the GatedDeltaNet layers are gathered, the attention one split."""
    model = _tiny_qwen3_5()
    assert model.config.layer_types.count("linear_attention") == 3
    batch = _left_padded_batch(model.config.vocab_size, pad=pad)
    _assert_sp_matches_reference(monkeypatch, model, batch)


def test_ulysses_sp_refuses_unpatched_linear_attention(monkeypatch):
    model = _tiny_qwen3_5()
    fake = _FakeDist(2)
    fake.set_rank(0)
    monkeypatch.setattr(sp, "dist", fake)
    monkeypatch.setattr(sp, "_is_linear_attention", lambda m: False)
    with pytest.raises(RuntimeError, match="linear_attention"):
        sp.apply_ulysses_sp(model, sp_group=object())


def test_padded_attention_routes_large_s_to_chunked(monkeypatch):
    """Past _SDP_MASK_MAX_SEQ the padded path must not materialize the SxS mask."""
    from razordl.ops.parallel import sequence_parallel as sp

    B, H, S, D = 1, 2, 32, 8
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    key_valid = torch.tensor([[0] * 5 + [1] * (S - 5)], dtype=torch.bool)  # 5 left-pad tokens

    orig_chunked = sp._chunked_causal_attention
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return orig_chunked(*args, **kwargs)

    monkeypatch.setattr(sp, "_chunked_causal_attention", spy)
    monkeypatch.setattr(sp, "_SDP_MASK_MAX_SEQ", 16)

    out_chunked, _ = sp._padded_causal_attention(q, k, v, 1.0, key_valid, chunk_size=8)
    assert len(calls) == 1, "S=32 > threshold=16 should route to the chunked kernel"

    calls.clear()
    monkeypatch.setattr(sp, "_SDP_MASK_MAX_SEQ", 10**9)
    out_sdpa, _ = sp._padded_causal_attention(q, k, v, 1.0, key_valid, chunk_size=8)
    assert calls == [], "S=32 < huge threshold should use the SDPA path"

    # The two paths agree on valid queries.  Fully-pad queries (positions
    # 0..4) differ by design: SDPA re-enables the diagonal, chunked returns 0.
    assert torch.allclose(out_chunked[:, 5:], out_sdpa[:, 5:], atol=1e-4)
