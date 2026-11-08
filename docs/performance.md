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

## Future probes

- Ulysses SP attention chunk size (`_chunked_causal_attention`) for long
  sequences.
- DataLoader worker count vs tokenization/cache hit behavior.
- End-to-end step time and peak memory for a real SFT/GRPO config.
