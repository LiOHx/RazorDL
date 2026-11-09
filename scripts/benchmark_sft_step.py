#!/usr/bin/env python3
"""Measure end-to-end SFT step time and peak GPU memory.

Manual probe.  It copies the first N rows of a JSONL dataset into a temp dir,
runs the real SFT trainer for N steps, and reports the per-step times from
``step_info.jsonl`` plus peak CUDA memory.

Usage:
    .venv/bin/python scripts/benchmark_sft_step.py \
        --model /path/to/Qwen3-0.6B \
        --data datasets/demo_chat \
        --steps 5
"""

import argparse
import json
import os
import tempfile
import uuid
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist

from razordl.core.engine.single_model.trainer import Trainer
from razordl.presets.sft import SFTConfig, SFTCollator, SFTDataset, SFTWorkGroup


def _prepare_data(source: str, steps: int) -> Path:
    src = Path(source)
    data_file = src if src.is_file() else src / "train.jsonl"
    if not data_file.is_file():
        raise FileNotFoundError(f"no train.jsonl under {source}")
    lines = [line for line in data_file.read_text().splitlines() if line.strip()][:steps]
    tmp = Path(tempfile.mkdtemp(prefix="razordl-sft-bench-"))
    data_dir = tmp / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("\n".join(lines) + "\n")
    return data_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--data", default="datasets/demo_chat")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--backend", default="ddp", choices=["ddp", "fsdp2"])
    parser.add_argument("--fused-linear-tile-size", type=int, default=128)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    data_dir = _prepare_data(args.data, args.steps)
    tmp = data_dir.parent
    exp_dir = tmp / "exp"

    flat = {
        "model": args.model,
        "data_path": str(data_dir),
        "outputs_dir": str(tmp / "outputs"),
        "batch_size": args.batch_size,
        "grad_accum": 1,
        "num_epochs": 1,
        "max_length": args.max_length,
        "num_workers": 0,
        "use_adapter": False,
        "precision": args.precision,
        "parallel_backend": args.backend,
        "enable_gradient_checkpointing": args.gradient_checkpointing,
        "save_steps": 0,
        "save_ckpt_steps": 0,
        "log_steps": 1,
        "fused_linear_tile_size": args.fused_linear_tile_size,
        "streaming": True,
        "cache_tokenized": False,
        "dataset_processor_path": args.model,
        "seed": 42,
    }
    config = SFTConfig.from_flat_dict(flat)
    config.trainer_config.output_dir = str(exp_dir)

    # Direct trainer: skip Ray's data-loader wrapper and final report.
    import razordl.core.base.trainer as trainer_module

    trainer_module.prepare_data_loader = lambda loader, **kwargs: loader
    import ray.train

    ray.train.report = lambda *args, **kwargs: None

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_file = tmp / f"dist_init_{uuid.uuid4().hex}"
    dist.init_process_group(
        backend, init_method=f"file://{init_file}", rank=0, world_size=1
    )
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        trainer = Trainer(config, SFTWorkGroup, SFTDataset, SFTCollator)
        trainer.run_training_loop()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        peak_gb = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available()
            else float("nan")
        )
        dist.destroy_process_group()

    step_info = [
        json.loads(line)
        for line in (exp_dir / "step_info.jsonl").read_text().splitlines()
        if line.strip()
    ]
    step_times = [row["step_time"] for row in step_info]
    steady = step_times[1:] if len(step_times) > 1 else step_times
    print(
        f"model={args.model} backend={args.backend} precision={args.precision} "
        f"batch={args.batch_size} max_length={args.max_length} steps={len(step_info)}"
    )
    print("step_times:", step_times)
    print(f"steady_mean={sum(steady) / len(steady):.3f}s peak_gpu={peak_gb:.2f}GB")
    assert len(step_info) == args.steps
    assert all("loss" in row for row in step_info)
    print("SFT STEP BENCH OK")


if __name__ == "__main__":
    main()
