# Testing workflows

Smoke tests for each preset. Loaded on demand via `@docs/testing.md` from the root `CLAUDE.md`.

Automated smoke tests:

```bash
# CPU integration job: tiny Qwen3 model, real forward/backward/checkpoint
pytest -m integration

# Manual 2-GPU job: FSDP2 + Ulysses SP, tiny model, 2 training steps
python scripts/smoke_fsdp2_sp.py
```

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

### Experiment-management checks (SFT, single GPU, ~35 s each)

Run the SFT flow above once with `save_steps: 5`, `save_ckpt_steps: 10`, `grad_accum: 1`, `num_epochs: 4` (20 steps on the bundled 5-sample `data/`). Then:

- **Fork:** copy `config.yaml` + `data/` to a new dir, set `init_from: <exp>/checkpoint_000010`. Log must contain `[EXP] New experiment (init_from fork)`, `[INIT_FROM] Loading weights from`, `[RESUME] Preloading LoRA adapter` and `[TRAINER] Training starting from scratch`. A bad path raises `FileNotFoundError` on the driver before Ray starts.
- **Copy recovery:** `cp -r <exp>/code /tmp/x && cp -r data /tmp/x && cd /tmp/x && razordl train ...` → `[EXP] Using pre-set output_dir` and `[RESUME] Found checkpoint at step 20`; nothing new is created under `/tmp/x`.
- **Custom outputs dir:** `outputs_dir: ./runs`, run once, confirm `runs/<exp>/code/runs` does not exist; delete `runs/<exp>/checkpoint_info.json` (simulates an interrupted run) and run again → `[EXP] Auto-resuming from ... (code + razordl hash match)`.
- **Backend switch:** `parallel_backend: ddp` + `resume_mode: manual` + `resume_from: <exp>` → `ValueError: [RESUME] parallel_backend mismatch`, zero training steps.

---

## vLLM rollout smoke (manual)

The project `.venv` does not install vLLM (CUDA-version sensitive).  Use a
separate environment that has vLLM, and point `PYTHONPATH` at the repo if
RazorDL is not installed there:

```bash
PYTHONPATH=. /path/to/vllm-env/bin/python scripts/smoke_vllm_rollout.py \
    --model /path/to/Qwen3.5-0.8B
```

Verified with vLLM 0.32.0 + Qwen3.5-0.8B on an RTX 2080 Ti (single GPU):
the wrapper loads the model, generates one completion, and returns matching
token/mask lists.  This covers the rollout generation path, not full GRPO/OPD
training or weight sync.

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
