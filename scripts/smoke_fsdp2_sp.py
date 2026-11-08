#!/usr/bin/env python3
"""Manual 2-GPU FSDP2 + Ulysses sequence-parallel smoke test.

Usage:
    python scripts/smoke_fsdp2_sp.py

Requires at least 2 CUDA GPUs.  The script builds a tiny Qwen3 checkpoint and
tokenizer in a temporary directory, launches two workers with
``torch.distributed.run``, trains for 2 steps with ``parallel_backend: fsdp2``
and ``sp_size: 2``, and verifies that both checkpoints and model weights were
written.

This is deliberately NOT part of the CPU CI job; FSDP2 needs CUDA.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _build_assets(root: Path) -> tuple[Path, Path, Path]:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    model_dir = root / "model"
    tokenizer_dir = root / "tokenizer"
    data_dir = root / "data"
    model_dir.mkdir()
    tokenizer_dir.mkdir()
    data_dir.mkdir()

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

    with open(data_dir / "train.jsonl", "w") as f:
        for question, answer in [
            ("what is 2 + 2", "4"),
            ("hello world", "hello"),
        ]:
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

    return model_dir, tokenizer_dir, data_dir


def _run_worker(asset_dir: Path) -> None:
    import torch
    import torch.distributed as dist

    from razordl.core.engine.single_model.trainer import Trainer
    from razordl.presets.sft import SFTConfig, SFTCollator, SFTDataset, SFTWorkGroup

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    # Ray is not involved in this direct-trainer smoke test.
    import razordl.core.base.trainer as trainer_module

    trainer_module.prepare_data_loader = lambda loader, **kwargs: loader

    import ray.train

    ray.train.report = lambda *args, **kwargs: None

    model_dir = asset_dir / "model"
    tokenizer_dir = asset_dir / "tokenizer"
    data_dir = asset_dir / "data"
    exp_dir = asset_dir / "exp"

    flat = {
        "model": str(model_dir),
        "data_path": str(data_dir),
        "outputs_dir": str(asset_dir / "outputs"),
        "parallel_backend": "fsdp2",
        "sp_size": 2,
        "precision": "bf16",
        "use_adapter": False,
        "batch_size": 1,
        "grad_accum": 1,
        "num_epochs": 1,
        "max_length": 16,
        "num_workers": 0,
        "lazy_tokenize": True,
        "save_steps": 0,
        "save_ckpt_steps": 1,
        "log_steps": 1,
        "fused_linear_tile_size": 8,
        "dataset_processor_path": str(tokenizer_dir),
        "seed": 42,
    }
    config = SFTConfig.from_flat_dict(flat)
    config.trainer_config.output_dir = str(exp_dir)

    try:
        trainer = Trainer(config, SFTWorkGroup, SFTDataset, SFTCollator)
        trainer.run_training_loop()
        dist.barrier()

        if dist.get_rank() == 0:
            for step in (1, 2):
                ckpt = exp_dir / f"checkpoint_{step:06d}"
                assert ckpt.is_dir(), f"missing {ckpt}"
                weights = list((ckpt / "workgroup" / "model_group").glob("*.safetensors"))
                assert weights, f"missing model weights in {ckpt}"
            assert (exp_dir / "checkpoint_info.json").is_file()
            rows = [
                json.loads(line)
                for line in (exp_dir / "step_info.jsonl").read_text().splitlines()
                if line.strip()
            ]
            assert len(rows) == 2
            assert all(math.isfinite(row["loss"]) for row in rows)
            print(f"[SMOKE] FSDP2 + SP passed: {exp_dir}")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("asset_dir", nargs="?")
    args = parser.parse_args()

    if args.worker:
        if not args.asset_dir:
            raise SystemExit("--worker requires asset_dir")
        _run_worker(Path(args.asset_dir))
        return

    import torch

    if torch.cuda.device_count() < 2:
        raise SystemExit(
            f"requires >= 2 CUDA GPUs, found {torch.cuda.device_count()}"
        )

    with tempfile.TemporaryDirectory(prefix="razordl-fsdp2-sp-") as tmp:
        root = Path(tmp)
        _build_assets(root)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                __file__,
                "--worker",
                str(root),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
