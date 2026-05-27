# RazorDL

A flexible distributed deep learning training framework for LLMs and multimodal models.

RazorDL is designed for researchers and engineers who want **full control over training logic** without writing boilerplate code for distributed training, checkpointing, LoRA, FSDP2, and sequence parallelism.

Compared to `transformers.Trainer` and `verl`, RazorDL lets you customize the training step with minimal code while the framework handles all infrastructure.

---

## Features

- **Minimal boilerplate** -- customize only what matters (`update_step`, `build_model`, `Dataset`)
- **Distributed training** -- built on Ray Train + PyTorch FSDP2
- **Sequence Parallelism** -- Ulysses-style SP for long-context training (Qwen2/3/3.5)
- **Chunked Loss** -- avoid OOM from giant logits tensor (vocab 248K × seq 65K)
- **Dataset Cache** -- auto-cache tokenized data to disk for instant reload
- **LoRA / PEFT** -- compatible with vLLM key-cleaned adapters
- **Activation Offloading** -- async CPU offloading with gradient checkpointing
- **Experiment management** -- auto-created timestamped experiment dirs with code snapshots, provenance hashes, and pip freeze; `resume_mode: auto` detects code/config changes via hash comparison
- **Deterministic training** -- centralized `set_seed` for reproducible weight initialization and per-step seeding
- **Distributed metrics** -- `Reducible` / `DistStats` aggregate cross-rank min/max/std/n exactly (no more "mean of per-rank max" artifacts)
- **On-Policy Distillation (PG)** -- student samples rollouts via vLLM, teacher provides per-token logπ as PPO advantage (no reference-KL penalty)
- **Three-tier user architecture** -- simple (black-box), custom (white-box), full (independent project)
- **CLI tooling** -- `razordl init`, `razordl train`, `razordl diff` for comparing experiments, and `razordl ckpt info <dir>` for inspecting checkpoint metadata

---

## Installation

```bash
git clone https://github.com/LiOHx/RazorDL.git
cd RazorDL
pip install -e .
```

For SFT training dependencies:
```bash
pip install -e ".[sft]"
```

For GRPO (reinforcement learning with vLLM rollout) dependencies:
```bash
pip install -e ".[grpo]"
```

For OPD (on-policy distillation with vLLM rollout) dependencies:
```bash
pip install -e ".[opd]"
```

For video-embedding (contrastive learning) dependencies:
```bash
pip install -e ".[video_embedding]"
```

For xformers attention optimization:
```bash
pip install -e ".[xformers]"
```

Install everything:
```bash
pip install -e ".[all]"
```

---

## Quick Start (Beginner)

For users who just want to fine-tune a model with their own data.

### 1. Generate a simple project

```bash
razordl init my_sft_project --preset sft
cd my_sft_project
```

This creates a minimal project:
```
my_sft_project/
├── config.yaml      # training hyperparameters
├── run.sh           # launch script
└── data/            # sample data (3–5 records) for quick validation
```

Each preset ships with a tiny sample dataset under `data/` so you can run a smoke test immediately.  Replace it with your own data when ready.

### 2. Prepare your data (optional for first run)

Place your data in `data/` as `train.jsonl` or `train.json`:

```jsonl
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
```

The built-in dataset auto-detects the file and uses the OpenAI chat-completion format.

### 3. Edit `config.yaml`

```yaml
model: Qwen/Qwen3-4B-Instruct
data_path: ./data
output_dir: ./output
max_length: 4096
lr: 5.0e-5
lora_r: 8
num_epochs: 3
```

### 4. Launch training

```bash
razordl train
# or
bash run.sh
```

---

## Custom Training (Intermediate)

For users who want to customize the loss, model loading, or data format.

### 1. Generate a custom project

```bash
razordl init my_custom_project --preset sft --mode custom
cd my_custom_project
```

This creates a full project:
```
my_custom_project/
├── config.yaml      # training hyperparameters
├── run.sh           # launch script
└── src/
    ├── dataset.py   # your data loading logic
    ├── workgroup.py # custom loss / training logic (optional)
    └── main.py      # entry point (no need to edit)
```

### 2. Custom dataset

Edit `dataset.py` to match your data format:

```python
from razordl.presets.sft import SFTDataset

class MyDataset(SFTDataset):
    def load_data(self):
        import json
        with open("data/train.jsonl") as f:
            return [json.loads(line) for line in f]

    def format_item(self, item):
        return {
            "messages": [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]},
            ],
        }
```

### 3. Custom loss

Edit `workgroup.py` to override `_compute_loss()`:

```python
from razordl.presets.sft import SFTWorkGroup

class MyWorkGroup(SFTWorkGroup):
    def _compute_loss(self, model, input_dict, labels):
        # Custom loss logic
        output = model(**input_dict)
        logits = output.logits
        # e.g. add an auxiliary loss term
        return my_custom_loss(logits, labels)
```

The base class handles sequence parallelism, gradient backward, and distributed averaging automatically. You only write the loss.

### 4. Custom model loading

If you need a custom model (e.g. vision-language), subclass `SFTModelGroup`:

```python
from razordl.presets.sft import SFTModelGroup

class MyModelGroup(SFTModelGroup):
    def build_model(self):
        # Load your custom model
        return MyVisionLanguageModel(...)
```

Then wire it in `main.py` or `workgroup.py`.

---

## Architecture

RazorDL uses a layered abstraction with three user tiers:

**Tier 1 — Simple (black-box)**
```bash
razordl init my_project --preset sft          # only config.yaml + run.sh
```
Edit `config.yaml`, put data in `data/`, run `razordl train`.

**Tier 2 — Custom (white-box)**
```bash
razordl init my_project --preset sft --mode custom
```
Full project with `dataset.py` and `workgroup.py`.  Override `_compute_loss()`
to customize the loss while the framework handles distributed training,
checkpointing, and sequence parallelism automatically.

**Tier 3 — Full (independent project)**
```bash
razordl init my_project --preset sft --mode full
```
Completely self-contained project (~5K lines) with all framework code
inlined.  Modify any layer — `src/engine/main.py`, `src/base/trainer.py`,
`src/ops/parallel/fsdp2.py`, etc.  No dependency on the razordl package.

```
Internal layers (copied into full-mode projects)
  Base layer    -- config, trainer, workgroup abstractions
  Engine layer  -- Ray init, training loop, data loading
  Ops layer     -- FSDP2, LoRA, sequence parallelism, distributed helpers

Preset layer (razordl/presets/)
  SFT preset           -- standard supervised fine-tuning
  DFT preset           -- confidence-weighted fine-tuning
  GRPO preset          -- Group Relative Policy Optimization (RL with vLLM rollout)
  OPD preset           -- PG On-Policy Distillation (student rollout + teacher logπ)
  video_embedding      -- video-text contrastive learning (multimodal)
  (future: DPO, ...)
```

---

## Full Independent Project (Advanced)

For users who want to own every line of training infrastructure — modify Ray initialization, FSDP2 strategy, training loop, or any layer directly.

### 1. Generate a fully independent project

```bash
razordl init my_full_project --preset sft --mode full
cd my_full_project
```

This creates a **self-contained project** with no dependency on the razordl package:

```
my_full_project/
├── src/
│   ├── base/          # core abstractions (from razordl/core/base/), including checkpoint_info
│   ├── engine/        # training engine (from razordl/core/engine/<variant>/)
│   │   └── common/    # shared FSDP2/LoRA/resume/optimizer lifecycle
│   ├── ops/           # FSDP2, LoRA, SP, distributed helpers, loss utilities
│   ├── config.py      # SFTConfig
│   ├── dataset.py     # complete Dataset + Collator
│   ├── workgroup.py   # complete ModelGroup + WorkGroup
│   └── main.py        # Ray entry point
├── config.yaml
├── requirements.txt
└── run.sh
```

### 2. Modify any layer

All code is in your project. Edit anything — `src/engine/main.py`, `src/base/trainer.py`, `src/ops/parallel/fsdp2.py`, etc. No need to fork the razordl repository.

```bash
pip install -r requirements.txt
bash run.sh
```

### How it works

razordl uses AST-based dependency analysis to discover exactly which modules your preset needs, copies them, and rewrites all `from razordl.xxx` imports to local imports. The result is a minimal, clean project (~5K lines) containing only the code you need — no examples, no tests, no unrelated presets.

---

## Project Structure

```
RazorDL/
├── razordl/
│   ├── core/              # framework internals
│   │   ├── base/          # config, trainer, workgroup abstractions, checkpoint_info
│   │   ├── engine/        # training engines
│   │   │   ├── common/    # shared FSDP2/LoRA/resume/optimizer lifecycle (single source of truth)
│   │   │   ├── single_model/        # supervised single-model engine (SFT/DFT/video_embedding)
│   │   │   └── on_policy_single_model/  # vLLM rollout + GRPO/PPO
│   │   └── export/        # full-mode project export (AST-based dependency walker)
│   ├── ops/               # FSDP2, SP, LoRA, HF builders, distributed helpers, loss utilities
│   ├── presets/           # pre-built configs + default templates for common tasks
│   │   ├── sft/           # supervised fine-tuning
│   │   ├── dft/           # confidence-weighted fine-tuning
│   │   ├── grpo/          # Group Relative Policy Optimization (math RL)
│   │   ├── opd/           # PG On-Policy Distillation (student rollout + teacher logπ)
│   │   ├── video_embedding/  # video-text contrastive learning
│   │   └── _template/     # boilerplate for new presets
│   └── cli/               # command line tools (init, train, ckpt)
```

---

## License

MIT
