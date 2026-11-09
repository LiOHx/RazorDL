# Performance notes

These are manual probes, not CI benchmarks.  Numbers are hardware-specific;
measure on your own model, sequence length, and GPU.

## Fused linear CE tile size

`FusedLinearCrossEntropy` streams the lm_head in tiles.  The tile size trades
extra recomputation for peak memory; it is not "bigger is always better".

Run:

```bash
python scripts/benchmark_fused_linear_ce.py
# or a smaller CPU probe:
python scripts/benchmark_fused_linear_ce.py --device cpu --seq-len 512
```

Sample result (RTX 2080 Ti, torch 2.6, fp32, batch 1, seq 2048, hidden 1024,
vocab 32000, 2 repeats):

| method | time (s) | peak MB |
|---|---:|---:|
| materialized logits | 0.0662 | 944.0 |
| fused tile=256 | 0.1559 | 629.7 |
| fused tile=512 | 0.0452 | 696.5 |
| fused tile=1024 | 0.0440 | 961.8 |
| fused tile=2048 | 0.0481 | 1345.6 |
| fused tile=4096 | 0.0479 | 1345.6 |

Takeaways from this run:

- Small tiles cut peak memory but can be much slower (tile=256 here).
- At tile=512/1024 the fused path was faster than materialized logits on this
  card; tile=1024 had higher peak memory than materialized in this specific
  configuration.
- The optimum depends on `vocab`, `seq_len`, dtype, and GPU.  Sweep
  `fused_linear_tile_size` on the real workload; the default 2048 is a
  reasonable starting point, not a universal best.

## Ulysses SP chunked-attention fallback

`_chunked_causal_attention` is the always-available SP attention fallback
(used when SDPA efficient kernels / xformers cannot handle the shape).  Larger
chunks are faster but the score buffer grows as `chunk² × heads`.

Run:

```bash
python scripts/benchmark_sp_attention.py
```

Sample result (RTX 2080 Ti, fp16, batch 1, heads 8, head_dim 64, 3 repeats):

| seq_len | chunk 512 | chunk 1024 | chunk 2048 |
|---:|---:|---:|---:|
| 2048 | 0.0064 s / 54.7 MB | 0.0062 s / 159.7 MB | 0.0130 s / 436.7 MB |
| 4096 | 0.0159 s / 63.1 MB | 0.0133 s / 168.1 MB | 0.0216 s / 579.3 MB |
| 8192 | 0.0438 s / 79.9 MB | 0.0336 s / 184.9 MB | 0.0417 s / 596.1 MB |

On this box chunk=1024 was the best time/memory point; chunk=512 traded speed
for lower memory, and chunk=2048 used substantially more memory.  The SDPA /
xformers fast paths bypass this fallback entirely, so benchmark the real model
before tuning.

## SFT DataLoader modes and worker count

`scripts/benchmark_sft_dataloader.py` compares offline / lazy / streaming
datasets and DataLoader worker counts.  It uses a deterministic fake tokenizer
to isolate dataset/worker overhead from real tokenizer cost.

```bash
python scripts/benchmark_sft_dataloader.py --samples 2000 --max-batches 50
```

Sample result (fake tokenizer, 2000 samples, batch 8, 48 timed batches,
2 repeats):

| mode | workers 0 | workers 2 | workers 4 |
|---|---:|---:|---:|
| offline | 215k samples/s | 5.9k | 6.0k |
| lazy | 99k samples/s | 5.5k | 6.1k |
| streaming | 57k samples/s | 5.7k | 5.8k |

Takeaway: with a trivial tokenizer, worker process IPC dominates and 0 workers
wins.  Workers only pay off when per-sample work (real tokenization, video
decode, augmentation) is expensive enough to exceed that overhead.  Measure
with the real tokenizer/data before changing `num_workers`.

## Future probes

- End-to-end step time and peak memory for a real SFT/GRPO config.
