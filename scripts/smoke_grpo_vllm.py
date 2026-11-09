#!/usr/bin/env python3
"""Full GRPO training smoke test with vLLM rollout + weight sync.

Manual test; run it inside the vLLM environment (`.venv-vllm`):

    .venv-vllm/bin/python scripts/smoke_grpo_vllm.py \
        --model /path/to/Qwen3-0.6B

It builds a tiny 2-row GSM8K-style dataset, runs 2 real GRPO steps with a
frozen reference model and vLLM rollout, and checks that losses/metrics and
the final model checkpoint are produced.

This is a pipeline smoke test, not a convergence test: with a 16-token
completion cap the reward may be 0 for every sample.
"""

import argparse
import json
import os
import tempfile
import uuid
from pathlib import Path

os.environ.setdefault("RAZORDL_VLLM_EXECUTOR_BACKEND", "mp")
os.environ.setdefault("RAZORDL_VLLM_ENFORCE_EAGER", "1")
os.environ.setdefault("RAZORDL_VLLM_SLEEP_MODE", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARN")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch.distributed as dist

from razordl.core.engine.on_policy_single_model.trainer import Trainer
from razordl.presets.grpo import GRPOConfig, GRPOCollator, GRPODataset, GRPOWorkGroup


def _write_data(data_dir: Path):
    rows = [
        ("What is 2 + 2?", "4"),
        ("What is 3 + 3?", "6"),
    ]
    with open(data_dir / "train.jsonl", "w") as f:
        for question, answer in rows:
            f.write(
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": question}],
                        "answer": answer,
                    }
                )
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-completion-length", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--save-ckpt-steps", type=int, default=0)
    args = parser.parse_args()

    os.environ["RAZORDL_VLLM_GPU_MEMORY_UTILIZATION"] = str(
        args.gpu_memory_utilization
    )

    tmp = Path(tempfile.mkdtemp(prefix="razordl-grpo-vllm-smoke-"))
    data_dir = tmp / "data"
    data_dir.mkdir()
    _write_data(data_dir)

    flat = {
        "model": args.model,
        "data_path": str(data_dir),
        "outputs_dir": str(tmp / "outputs"),
        "max_length": args.max_length,
        "max_completion_length": args.max_completion_length,
        "group_size": args.group_size,
        "batch_size": args.batch_size,
        "grad_accum": 1,
        "num_epochs": args.num_epochs,
        "num_workers": 0,
        "use_adapter": False,
        "precision": args.precision,
        "parallel_backend": "ddp",
        "enable_gradient_checkpointing": True,
        "save_steps": 0,
        "save_ckpt_steps": args.save_ckpt_steps,
        "log_steps": 1,
        "fused_linear_tile_size": 64,
        "dataset_processor_path": args.model,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "kl_coef": 0.001,
        "clip_eps": 0.2,
        "loss_micro_batch_size": 1,
        "seed": 42,
    }
    config = GRPOConfig.from_flat_dict(flat)
    exp_dir = tmp / "exp"
    config.trainer_config.output_dir = str(exp_dir)

    # Direct trainer: skip Ray's data-loader wrapper and final report.
    import razordl.core.base.trainer as trainer_module

    trainer_module.prepare_data_loader = lambda loader, **kwargs: loader
    import ray.train

    ray.train.report = lambda *args, **kwargs: None

    init_file = tmp / f"dist_init_{uuid.uuid4().hex}"
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=0, world_size=1
    )
    try:
        trainer = Trainer(config, GRPOWorkGroup, GRPODataset, GRPOCollator)
        trainer.run_training_loop()
    finally:
        dist.destroy_process_group()

    step_info = [
        json.loads(line)
        for line in (exp_dir / "step_info.jsonl").read_text().splitlines()
        if line.strip()
    ]
    print("EXP:", exp_dir)
    print("STEPS:", len(step_info), step_info)
    assert step_info, "training produced no step metrics"
    assert all("loss" in row and "reward" in row for row in step_info)
    assert (exp_dir / "checkpoint_info.json").is_file()
    print("GRPO VLLM SMOKE OK")


if __name__ == "__main__":
    main()
