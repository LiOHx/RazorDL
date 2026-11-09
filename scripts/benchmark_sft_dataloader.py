#!/usr/bin/env python3
"""Benchmark SFT data-loading modes and DataLoader worker counts.

Manual probe, not a CI test.  Uses a deterministic fake tokenizer so the
measurement isolates dataset mode / worker overhead from real tokenizer cost.

Usage:
    python scripts/benchmark_sft_dataloader.py
    python scripts/benchmark_sft_dataloader.py --samples 5000 --workers 0,2,4
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from torch.utils.data import DataLoader

import razordl.presets.sft.dataset as dataset_module
from razordl.presets.sft import SFTConfig, SFTCollator, SFTDataset


class _FakeTokenizer:
    chat_template = "{{ messages }}"
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "left"

    def apply_chat_template(
        self, messages, tokenize=False, tools=None, add_generation_prompt=False
    ):
        return " ".join(message["content"] for message in messages)

    def encode(self, text, add_special_tokens=False):
        return [2 + (hash(word) % 1000) for word in text.split()]


def _write_data(data_dir: Path, samples: int):
    data_dir.mkdir(parents=True)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    with open(data_dir / "train.jsonl", "w") as f:
        for i in range(samples):
            question = " ".join(words[(i + j) % len(words)] for j in range(8))
            answer = " ".join(words[(i + j + 3) % len(words)] for j in range(6))
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


def _make_dataset(root: Path, mode: str, samples: int, max_length: int):
    data_dir = root / f"data_{mode}"
    _write_data(data_dir, samples)
    flat = {
        "model": "dummy-model",
        "data_path": str(data_dir),
        "outputs_dir": str(root / "outputs"),
        "max_length": max_length,
        "dataset_processor_path": "dummy-tokenizer",
        "streaming": mode == "streaming",
        "lazy_tokenize": mode == "lazy",
        "cache_tokenized": False,
        "batch_size": 8,
        "num_workers": 0,
    }
    config = SFTConfig.from_flat_dict(flat)
    return SFTDataset(config)


def _benchmark(dataset, batch_size, workers, max_batches, warmup_batches, repeats):
    collator = SFTCollator(dataset.dataset_processor, sp_size=1)
    timed_batches = max(0, max_batches - warmup_batches)
    times = []
    for _ in range(repeats):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=collator,
            persistent_workers=workers > 0,
        )
        iterator = iter(loader)
        for _ in range(warmup_batches):
            next(iterator, None)
        start = time.perf_counter()
        for batch_idx, _batch in enumerate(iterator):
            if batch_idx + 1 >= timed_batches:
                break
        times.append(time.perf_counter() - start)
    return statistics.median(times), timed_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--workers", default="0,2,4")
    parser.add_argument("--modes", default="offline,lazy,streaming")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    # Avoid downloading a tokenizer; the fake one is deterministic.
    dataset_module.AutoTokenizer.from_pretrained = lambda *a, **k: _FakeTokenizer()

    workers_list = [int(w) for w in args.workers.split(",") if w.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    with tempfile.TemporaryDirectory(prefix="razordl-dataloader-bench-") as tmp:
        root = Path(tmp)
        print(
            f"samples={args.samples} batch_size={args.batch_size} "
            f"max_length={args.max_length} max_batches={args.max_batches} "
            f"repeats={args.repeats}"
        )
        print(f"{'mode':<12} {'workers':>8} {'time (s)':>10} {'samples/s':>12}")
        print("-" * 48)
        for mode in modes:
            dataset = _make_dataset(root, mode, args.samples, args.max_length)
            for workers in workers_list:
                elapsed, timed_batches = _benchmark(
                    dataset,
                    args.batch_size,
                    workers,
                    args.max_batches,
                    args.warmup_batches,
                    args.repeats,
                )
                loaded = min(timed_batches * args.batch_size, len(dataset))
                print(
                    f"{mode:<12} {workers:>8} {elapsed:>10.4f} "
                    f"{loaded / elapsed:>12.1f}"
                )


if __name__ == "__main__":
    main()
