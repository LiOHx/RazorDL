# Testing workflows

Smoke tests for each preset. Loaded on demand via `@docs/testing.md` from the root `CLAUDE.md`.

After any smoke test, inspect the resulting checkpoint with:

```bash
razordl ckpt info <output_dir>/checkpoint_NNNNNN
```

Pretty-prints `checkpoint_info.json` (or reports legacy format for old `complete`-only checkpoints). Use it to verify topology, metrics, file sizes, and provenance.

---

## SFT preset

```bash
# 1. Create temp project
python -c "
from razordl.cli.init import handle_init
from argparse import Namespace
handle_init(Namespace(project_name='test_sft', preset='sft', path='/tmp', mode='simple'))
"

# 2. Point to a local model
cd /tmp/test_sft
# Edit config.yaml: model: /path/to/Qwen3.5-4B
# Edit run.sh: CUDA_VISIBLE_DEVICES=2,3

# 3. Run
bash run.sh
```

---

## GRPO preset

Requires extra deps:

```bash
source .venv/bin/activate
uv pip install vllm
```

Then the same flow as SFT but with `--preset grpo`:

```bash
python -c "
from razordl.cli.init import handle_init
from argparse import Namespace
handle_init(Namespace(project_name='test_grpo', preset='grpo', path='/tmp', mode='simple'))
"
cd /tmp/test_grpo
# Edit config.yaml: model: /path/to/Qwen3-0.6B
# Edit run.sh: CUDA_VISIBLE_DEVICES=2,3
bash run.sh
```

**Key checks during test:**

- vLLM engine starts, generates completions
- `[GRPO sample]` logs show responses with rewards (0.0 / 0.5 / 1.5)
- `reward_mean` increases from near 0 toward > 0
- Loss decreases and `clip_fraction` stays near 0

---

## OPD preset

Requires extra deps:

```bash
source .venv/bin/activate
uv pip install vllm
```

Then the same flow as SFT but with `--preset opd`:

```bash
python -c "
from razordl.cli.init import handle_init
from argparse import Namespace
handle_init(Namespace(project_name='test_opd', preset='opd', path='/tmp', mode='simple'))
"
cd /tmp/test_opd
# Edit config.yaml: model: /path/to/Qwen3.5-0.8B, teacher_model: /path/to/Qwen3.5-4B
# Edit run.sh: CUDA_VISIBLE_DEVICES=2,3
bash run.sh
```

**Key checks during test:**

- vLLM engine starts, generates completions
- `[OPD sample]` logs show student responses with teacher log-prob advantage
- `loss` decreases and `distill_loss` tracks the KL between student and teacher
- Only the policy model (student) appears in checkpoint dirs; teacher is frozen and not persisted

---

## video_embedding preset

Requires extra deps not in core:

```bash
source .venv/bin/activate
uv pip install decord opencv-python qwen_vl_utils
uv pip install torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then the same flow as SFT but with `--preset video_embedding`.

**Key checks during test:**

- `num_gpus` in logs matches `CUDA_VISIBLE_DEVICES`
- Model loads, FSDP2 wraps, dataset loads
- First training step completes without `TensorDict` / `batch_size` mismatch
- Checkpoint saves correctly
