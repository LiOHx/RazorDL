#!/usr/bin/env python3
"""Benchmark FusedLinearCrossEntropy against materialized logits.

This is a manual performance probe, not a CI test.  It reports wall time and
peak CUDA memory for the full forward+backward of a per-token CE loss.

Usage:
    python scripts/benchmark_fused_linear_ce.py
    python scripts/benchmark_fused_linear_ce.py --device cpu --seq-len 512
    python scripts/benchmark_fused_linear_ce.py --tiles 256,512,1024,2048
"""

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from razordl.ops.loss.fused_linear_ce import fused_linear_cross_entropy


def _benchmark(fn, device, repeats):
    times = []
    peaks = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        loss = fn()
        loss.sum().backward()
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
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tiles", default="256,512,1024,2048,4096")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    torch.manual_seed(0)
    hidden = torch.randn(
        args.batch, args.seq_len, args.hidden, device=device, requires_grad=True
    )
    weight = torch.randn(
        args.vocab, args.hidden, device=device, requires_grad=True
    )
    target = torch.randint(
        0, args.vocab, (args.batch, args.seq_len), device=device
    )

    def materialized():
        hidden.grad = None
        weight.grad = None
        logits = hidden @ weight.T
        return F.cross_entropy(
            logits.reshape(-1, args.vocab), target.reshape(-1), reduction="none"
        )

    print(
        f"device={device} batch={args.batch} seq_len={args.seq_len} "
        f"hidden={args.hidden} vocab={args.vocab} repeats={args.repeats}"
    )
    print(f"{'method':<24} {'time (s)':>10} {'peak MB':>10}")
    print("-" * 48)

    elapsed, peak = _benchmark(materialized, device, args.repeats)
    print(f"{'materialized logits':<24} {elapsed:>10.4f} {peak:>10.1f}")

    for tile in [int(t) for t in args.tiles.split(",") if t.strip()]:
        def fused(tile=tile):
            hidden.grad = None
            weight.grad = None
            return fused_linear_cross_entropy(hidden, weight, target, tile)

        elapsed, peak = _benchmark(fused, device, args.repeats)
        print(f"{'fused tile=' + str(tile):<24} {elapsed:>10.4f} {peak:>10.1f}")


if __name__ == "__main__":
    main()
