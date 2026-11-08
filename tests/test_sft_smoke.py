"""Real (non-mocked) single-process SFT training smoke tests.

These exercise the actual model/dataset/workgroup/trainer path: tokenizer
loading, forward/backward, fused linear CE, optimizer step, checkpointing,
resume, init_from, and experiment metadata.  They run on CPU by forcing the
hardware dispatcher, so they are safe in the integration CI job.
"""

import json
import math
import uuid

import pytest
import torch.distributed as dist
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from razordl.core.engine.single_model.trainer import Trainer
from razordl.ops.hardware import device as hw_device
from razordl.presets.sft import SFTConfig, SFTCollator, SFTDataset, SFTWorkGroup


pytestmark = pytest.mark.integration


def _write_tiny_model_and_tokenizer(root):
    model_dir = root / "model"
    tokenizer_dir = root / "tokenizer"
    model_dir.mkdir()
    tokenizer_dir.mkdir()

    vocab = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
    for word in ["user", "assistant", "hello", "world", "what", "is", "2", "+", "4", ":", "\n"]:
        vocab[word] = len(vocab)

    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        bos_token="[BOS]",
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    tokenizer.save_pretrained(tokenizer_dir)

    config = Qwen3Config(
        vocab_size=len(vocab),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=3,
    )
    Qwen3ForCausalLM(config).save_pretrained(model_dir)
    return model_dir, tokenizer_dir


def _write_data(data_dir, rows=None):
    data_dir.mkdir()
    rows = rows or [
        ("what is 2 + 2", "4"),
        ("hello world", "hello"),
        ("what is 2 + 2", "4"),
        ("hello world", "hello"),
    ]
    with open(data_dir / "train.jsonl", "w") as f:
        for question, answer in rows:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ]
                    }
                )
                + "\n"
            )


def _prepare_runtime(monkeypatch):
    """Force CPU and remove the two Ray-only seams from the trainer path."""
    monkeypatch.setattr(hw_device, "get_available_device", lambda: "cpu")
    monkeypatch.setattr(hw_device, "get_device_count", lambda: 0)

    import razordl.core.base.trainer as trainer_module

    monkeypatch.setattr(
        trainer_module, "prepare_data_loader", lambda loader, **kwargs: loader
    )

    import ray.train

    monkeypatch.setattr(ray.train, "report", lambda *args, **kwargs: None)


def _flat_config(model_dir, tokenizer_dir, data_dir, outputs_dir, **overrides):
    flat = {
        "model": str(model_dir),
        "data_path": str(data_dir),
        "outputs_dir": str(outputs_dir),
        "parallel_backend": "ddp",
        "precision": "fp32",
        "use_adapter": False,
        "batch_size": 1,
        "grad_accum": 1,
        "num_epochs": 1,
        "max_length": 32,
        "num_workers": 0,
        "save_steps": 0,
        "save_ckpt_steps": 2,
        "log_steps": 1,
        "fused_linear_tile_size": 8,
        "dataset_processor_path": str(tokenizer_dir),
        "seed": 42,
    }
    flat.update(overrides)
    return flat


def _run_training(flat, exp_dir):
    config = SFTConfig.from_flat_dict(flat)
    config.trainer_config.output_dir = str(exp_dir)

    if dist.is_initialized():
        dist.destroy_process_group()
    init_file = exp_dir.parent / f"dist_init_{uuid.uuid4().hex}"
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=0, world_size=1
    )
    try:
        Trainer(config, SFTWorkGroup, SFTDataset, SFTCollator).run_training_loop()
    finally:
        dist.destroy_process_group()
    return config


def _step_info(exp_dir):
    return [
        json.loads(line)
        for line in (exp_dir / "step_info.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_sft_trains_and_writes_checkpoint(tmp_path, monkeypatch):
    model_dir, tokenizer_dir = _write_tiny_model_and_tokenizer(tmp_path)
    data_dir = tmp_path / "data"
    _write_data(data_dir)
    _prepare_runtime(monkeypatch)

    exp_dir = tmp_path / "exp"
    _run_training(
        _flat_config(
            model_dir, tokenizer_dir, data_dir, tmp_path / "outputs", streaming=True
        ),
        exp_dir,
    )

    assert (exp_dir / "checkpoint_000002").is_dir()
    assert (exp_dir / "checkpoint_000004").is_dir()
    assert (exp_dir / "checkpoint_info.json").is_file()

    model_files = list(
        (exp_dir / "checkpoint_000002" / "workgroup" / "model_group").glob("*.safetensors")
    )
    assert model_files, "full checkpoint should contain model weights"
    assert (
        exp_dir / "checkpoint_000002" / "workgroup" / "model_group" / "optimizer.pt"
    ).is_file()

    info = json.loads((exp_dir / "checkpoint_info.json").read_text())
    assert info["kind"] == "model_only"

    step_info = _step_info(exp_dir)
    assert len(step_info) == 4
    assert all(math.isfinite(row["loss"]) for row in step_info)


def test_resume_from_checkpoint_continues(tmp_path, monkeypatch):
    model_dir, tokenizer_dir = _write_tiny_model_and_tokenizer(tmp_path)
    data_dir = tmp_path / "data"
    _write_data(data_dir, rows=[("what is 2 + 2", "4"), ("hello world", "hello")])
    _prepare_runtime(monkeypatch)

    exp_dir = tmp_path / "exp"
    flat = _flat_config(model_dir, tokenizer_dir, data_dir, tmp_path / "outputs")
    _run_training(flat, exp_dir)

    assert [row["step"] for row in _step_info(exp_dir)] == [1, 2]
    assert (exp_dir / "checkpoint_000002").is_dir()

    resume_flat = dict(flat)
    resume_flat.update(
        {
            "resume_mode": "manual",
            "resume_from": str(exp_dir),
            "num_epochs": 2,
        }
    )
    _run_training(resume_flat, exp_dir)

    assert [row["step"] for row in _step_info(exp_dir)] == [1, 2, 3, 4]
    assert (exp_dir / "checkpoint_000004").is_dir()


def test_init_from_fork_starts_at_step_zero(tmp_path, monkeypatch):
    model_dir, tokenizer_dir = _write_tiny_model_and_tokenizer(tmp_path)
    data_dir = tmp_path / "data"
    _write_data(data_dir, rows=[("what is 2 + 2", "4"), ("hello world", "hello")])
    _prepare_runtime(monkeypatch)

    source_dir = tmp_path / "source"
    flat = _flat_config(
        model_dir, tokenizer_dir, data_dir, tmp_path / "outputs", save_ckpt_steps=1
    )
    _run_training(flat, source_dir)
    assert (source_dir / "checkpoint_000001").is_dir()

    fork_dir = tmp_path / "fork"
    fork_flat = dict(flat)
    fork_flat.update(
        {
            "init_from": str(source_dir / "checkpoint_000001"),
            "outputs_dir": str(tmp_path / "fork_outputs"),
        }
    )
    _run_training(fork_flat, fork_dir)

    assert [row["step"] for row in _step_info(fork_dir)] == [1, 2]
    assert (fork_dir / "checkpoint_000002").is_dir()
