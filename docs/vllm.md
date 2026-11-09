# vLLM environment (GRPO / OPD)

vLLM is **not** part of the project `.venv`.  It pins its own torch /
transformers versions and is CUDA-wheel sensitive, so use a separate
environment.

## Tested baseline

This is the environment the vLLM rollout smoke was verified with:

| Component | Tested version |
|---|---|
| Python | 3.11.15 |
| vLLM | 0.32.0+cu129 |
| torch | 2.13.0+cu129 |
| transformers | 5.19.0 (vLLM requires `>=5.16.1,<5.20`) |
| CUDA wheels | cu129; driver 12.7 worked via minor-version compatibility |
| GPU | RTX 2080 Ti (sm_75) |
| RazorDL extras | core + `grpo` / `opd` |
| Verified models | Qwen3.5-0.8B and Qwen3.5-4B (`Qwen3_5ForConditionalGeneration`) |
| Verified paths | rollout generation + full GRPO/OPD training with Qwen3-0.6B |
| Smoke scripts | `smoke_vllm_rollout.py`, `smoke_grpo_vllm.py`, `smoke_opd_vllm.py` |

`vllm_rollout.py` targets the vLLM V1 API (the module docstring mentions
vLLM >= 0.16).  **Only 0.32.0 is verified**; older V1 versions may work but are
not covered by CI or the smoke script.

## Project-local environment (tested)

A project-local `.venv-vllm` was built and verified in this repository:

| Package | Version |
|---|---|
| Python | 3.11.15 |
| vLLM | 0.32.0+cu129 (GitHub release wheel) |
| torch | 2.13.0+cu129 |
| torchvision / torchaudio / torchcodec | 0.28.0+cu129 / 2.11.0+cu129 / 0.16.0+cu129 |
| transformers | 5.19.0 |
| numpy | 2.3.5 |
| fsspec | 2025.3.0 |
| cuda-python / cuda-bindings | 12.9.10 / 12.9.10 |

`uv pip check` is clean and the full test suite passes in that environment
(211 passed).  Use it with:

```bash
source .venv-vllm/bin/activate
python scripts/smoke_vllm_rollout.py --model /path/to/Qwen3.5-0.8B
```

The environment directory contains uv's own `.gitignore`, so it is not
committed.

> **Important:** do not install plain `vllm==0.32.0` from PyPI on this driver.
> PyPI resolves torch `+cu130`, which fails on the installed 12.7 driver.
> Use the vLLM `+cu129` wheel from the GitHub release and install torch /
> torchvision / torchaudio / torchcodec from the PyTorch `cu129` index.

## Install

Create a separate environment and install RazorDL first, then vLLM (vLLM's
pins must win):

```bash
uv venv --python 3.11 /path/to/razordl-vllm
uv pip install --python /path/to/razordl-vllm/bin/python -e ".[grpo]"
uv pip install --python /path/to/razordl-vllm/bin/python "vllm==0.32.0"
uv pip check --python /path/to/razordl-vllm/bin/python
```

If you only want to run the rollout smoke against a checkout without
installing RazorDL into that environment:

```bash
PYTHONPATH=. /path/to/razordl-vllm/bin/python \
    scripts/smoke_vllm_rollout.py \
    --model /path/to/Qwen3.5-0.8B
```

## Smoke test

```bash
PYTHONPATH=. /path/to/razordl-vllm/bin/python \
    scripts/smoke_vllm_rollout.py \
    --model /path/to/Qwen3.5-0.8B
```

Qwen3.5-4B also works on the same 22 GB card:

```bash
PYTHONPATH=. /path/to/razordl-vllm/bin/python \
    scripts/smoke_vllm_rollout.py \
    --model /path/to/Qwen3.5-4B \
    --gpu-memory-utilization 0.5 \
    --max-model-len 256
```

At `gpu_memory_utilization=0.5` the 4B model leaves only ~0.69 GiB for KV
cache (6 concurrent 256-token requests); the 0.8B model leaves ~6.8 GiB.
Raise the utilization or lower `max_model_len` if you need more concurrency.

Expected output ends with:

```text
response_token_ids: [[...]]
response_mask: [[1, 1, ...]]
VLLM SMOKE OK
```

## Full GRPO / OPD training smoke

Both presets were verified end-to-end with Qwen3-0.6B (text-only) on one
RTX 2080 Ti:

```bash
# GRPO: rollout + frozen reference + vLLM weight sync + optimizer steps
.venv-vllm/bin/python scripts/smoke_grpo_vllm.py \
    --model /path/to/Qwen3-0.6B

# OPD: student rollout + teacher log-probs + chunked PPO loss
.venv-vllm/bin/python scripts/smoke_opd_vllm.py \
    --model /path/to/Qwen3-0.6B
```

Both produced 2 step records with finite losses/grad norms and a final
`checkpoint_info.json`.  The scripts force `mp` executor + eager mode; the
production Ray path still uses `external_launcher`.

Notes from the smoke runs:

- GRPO used `fp16` and passed.
- OPD `fp16` hit a GradScaler overflow (`grad_norm=nan`, step skipped) on the
  second step; the smoke script defaults to `fp32`, which passed cleanly.
- Qwen3.5-0.8B full GRPO **OOMs at vLLM init** on a 22 GB card because the
  multimodal model + policy + frozen reference + vLLM exceed memory.  Its
  rollout-only path is verified; use a text-only model or more VRAM for full
  training.

## Direct-smoke overrides

The GRPO/OPD presets read these environment variables before constructing the
vLLM engine:

| Variable | Default | Purpose |
|---|---|---|
| `RAZORDL_VLLM_EXECUTOR_BACKEND` | `external_launcher` | use `mp` for direct single-process smoke |
| `RAZORDL_VLLM_ENFORCE_EAGER` | `0` | `1` avoids compile/CUDAGraph warmup |
| `RAZORDL_VLLM_GPU_MEMORY_UTILIZATION` | `0.25` | vLLM KV-cache budget |
| `RAZORDL_VLLM_SLEEP_MODE` | `1` | `0` disables sleep mode for direct smoke |

## Known limitations

- **LoRA weight sync is not implemented for vLLM V1.**  `update_weights()`
  raises `NotImplementedError` when `peft_config` is passed; use
  `use_adapter: false` for GRPO/OPD with vLLM.
- Full training with **Qwen3.5 multimodal** models needs more VRAM than the
  22 GB test card for policy + reference + vLLM.
- On sm_75 (RTX 2080 Ti), vLLM disables FlashInfer sampler/kernels and falls
  back to slower paths.  This is expected.
- Do not install vLLM into the project `.venv`: it upgrades torch to 2.13 and
  changes the tested SFT/GRPO baseline.
