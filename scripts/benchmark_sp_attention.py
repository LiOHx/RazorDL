#!/usr/bin/env python3
"""Benchmark the Ulysses SP chunked-attention fallback.

Manual probe for `_chunked_causal_attention`: larger chunks are faster but use
more memory (scores are chunk_size x chunk_size per head).  The SDPA/xformers
fast paths are not exercised here; this measures the always-available fallback.

Usage:
    python scripts/benchmark_sp_attention.py
    python scripts/benchmark_sp_attention.py --seq-lens 2048,4096 --chunks 512,1024
"""

import argparse
import statistics
import time

import torch

from razordl.ops.parallel.sequence_parallel import _chunked_causal_attention


def _measure(q, k, v, chunk_size, device, repeats):
    times = []
    peaks = []
    scale = 1.0 / (q.shape[-1] ** 0.5)
    with torch.no_grad():
        # Warm up CUDA kernels/allocator before the first timed run.
        _chunked_causal_attention(q, k, v, scale, chunk_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            _chunked_causal_attention(q, k, v, scale, chunk_size)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            if device.type == "cuda":
                peaks.append(torch.cuda.max_memory_allocated() / 1e6)
    return statistics.median(times), max(peaks) if peaks else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seq-lens", default="2048,4096,8192")
    parser.add_argument("--chunks", default="512,1024,2048,4096")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for seq_len in [int(s) for s in args.seq_lens.split(",") if s.strip()]:
        q = torch.randn(
            args.batch, args.heads, seq_len, args.head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        print(
            f"device={device} batch={args.batch} heads={args.heads} "
            f"head_dim={args.head_dim} seq_len={seq_len} dtype={dtype}"
        )
        print(f"{'chunk':>8} {'time (s)':>10} {'peak MB':>10}")
        print("-" * 32)
        for chunk in [int(c) for c in args.chunks.split(",") if c.strip()]:
            elapsed, peak = _measure(q, k, v, chunk, device, args.repeats)
            print(f"{chunk:>8} {elapsed:>10.4f} {peak:>10.1f}")
        print()


if __name__ == "__main__":
    main()
