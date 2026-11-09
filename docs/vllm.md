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
| Verified path | `GRPOVLLMRollout` generation (`scripts/smoke_vllm_rollout.py`) |

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

## Known limitations

- **LoRA weight sync is not implemented for vLLM V1.**  `update_weights()`
  raises `NotImplementedError` when `peft_config` is passed; use
  `use_adapter: false` for GRPO/OPD with vLLM.
- **Full GRPO/OPD training with vLLM weight sync is not verified.**  The
  rollout generation path is verified; the preset's full training path uses
  `external_launcher`, `enable_sleep_mode=True`, and Ray, and still needs a
  dedicated smoke run.
- On sm_75 (RTX 2080 Ti), vLLM disables FlashInfer sampler/kernels and falls
  back to slower paths.  This is expected.
- Do not install vLLM into the project `.venv`: it upgrades torch to 2.13 and
  changes the tested SFT/GRPO baseline.
