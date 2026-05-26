# GRPO Preset

Group Relative Policy Optimization (GRPO) for math reasoning with vLLM-based rollout.

## Overview

This preset trains a causal LM to solve math problems using GRPO:

- **vLLM rollout**: Fast generation of multiple completions per prompt
- **Group-normalized advantage**: Rewards are normalized within each group of completions
- **k1 KL penalty**: Prevents the policy from diverging too far from the reference model
- **No critic network**: Uses group mean/std instead of a separate value model

## Requirements

Install the GRPO extra dependencies:

```bash
pip install -e ".[grpo]"
```

vLLM is required for fast rollout but must be installed separately due to CUDA version sensitivity:

```bash
pip install vllm>=0.16.0
```

If vLLM is not available, the preset falls back to HF `generate()` (much slower).

## Dataset

GRPO uses the **GSM8K** dataset (included in `razordl/datasets/gsm8k/`).

When you run `razordl init --preset grpo`, the CLI automatically copies the
GSM8K dataset from the shared `razordl/datasets/` directory into your project
`data/` folder. The preset declares this via `DATASET = "gsm8k"` in
`__init__.py`.

If you want to use your own data, modify `data_path` in `config.yaml` or replace
the contents of the `data/` directory.

## Data Format

Each line in `train.jsonl` is a JSON object with OpenAI chat-completion messages.

**Recommended format** (answer as a separate field):

```json
{"messages": [{"role": "user", "content": "What is 15 + 27?"}], "answer": "42"}
```

**Legacy format** (answer in the last assistant message, backward-compatible):

```json
{"messages": [{"role": "user", "content": "What is 15 + 27?"}, {"role": "assistant", "content": "42"}]}
```

The system prompt is prepended automatically by the collator:

```
You are a helpful math assistant. You MUST follow this format:
1. Think step by step inside <think>...</think> tags.
2. After </think>, output ONLY the answer using the format: \boxed{answer}.
```

## Quick Start

```bash
# 1. Create a project
razordl init --project_name my_grpo --preset grpo --mode simple

# 2. Edit config.yaml (set model, data_path, etc.)
# 3. Run training
bash run.sh
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 2 | Prompts per engine step |
| `group_size` | 8 | Completions per prompt (GRPO group size) |
| `grad_accum` | 8 | Engine steps per optimizer update |
| `lr` | 1e-6 | Learning rate (conservative, stable for 0.6B) |
| `kl_coef` | 0.001 | KL penalty coefficient |
| `max_completion_length` | 2048 | Max generated tokens per completion |
| `temperature` | 0.9 | Sampling temperature |
| `top_p` | 0.9 | Nucleus sampling p |
| `clip_eps` | 0.2 | PPO clipping epsilon |

## Reward Structure

Three-level reward to avoid binary collapse:

- **0.0**: No `</think>` tag (wrong format)
- **0.5**: Correct format, wrong answer
- **1.5**: Correct format, correct answer

Answer extraction supports `\boxed{...}`, `#### ...`, or bare trailing numbers. Float comparison handles `5`, `5.0`, and `5.` as equal.

## Known Limitations

- **LoRA + vLLM**: LoRA weight sync is not yet implemented for vLLM V1. Use `use_adapter: false` (default) for full-model training.
- **Single GPU memory**: 0.6B model + reference model + vLLM fits in ~45GB GPU with `gpu_memory_utilization=0.25`.

## Evaluation

After training, evaluate a checkpoint with an independent vLLM run:

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

checkpoint = "./output/checkpoint_003040/workgroup/policy_model_group"
llm = LLM(model=checkpoint, dtype="bfloat16", trust_remote_code=True)
# ... generate and compute reward
```

## References

- DeepSeekMath: [arxiv:2402.03300](https://arxiv.org/abs/2402.03300)
- veRL / TRL GRPO implementations
