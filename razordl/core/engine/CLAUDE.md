# Engine layer — shared lifecycle + variants

Loaded when working in `razordl/core/engine/`.

## Layer responsibilities (top-down)

| Layer | Responsibility | Example |
|---|---|---|
| `core/base/` | Abstract contracts, generic serialization | `BaseTrainer`, `checkpoint_info` |
| `core/engine/common/` | Shared lifecycle for ALL variants | `FSDPModelGroup`, `EngineWorkGroup`, `EngineTrainer`, `flat_config` |
| `core/engine/<variant>/` | Paradigm-specific thin orchestration | `single_model.WorkGroup`, `on_policy_single_model.WorkGroup` |
| `presets/<name>/` | Task-specific data / model / loss / reward | `SFTWorkGroup`, `GRPOWorkGroup` |

## Hard rules

- **IMPORTANT — Engine is pure framework, NEVER training logic.** Concrete forward passes, loss functions (CE, contrastive, ...), SP split logic, chunked-loss monkey-patching, backward calls belong in **presets** (task-specific) or **ops** (stateless utilities). If you write `torch.nn.functional.cross_entropy` (or any concrete loss) in an engine file, it is in the wrong layer. We deleted `engine/single_model/lm_workgroup.py` for exactly this violation — its logic moved to `presets/sft/workgroup.py::SFTWorkGroup`. Before adding code here, ask: *"Is this specific training behavior, or generic infrastructure?"* Specific → preset.
- **IMPORTANT — Shared engine logic has ONE home: `engine/common/`.** Never duplicate FSDP2 / LoRA / resume / optimizer / save-load / grad-clip / offload / seeding across variants. If a variant needs to differ, expose a hook in `engine/common/`, do not fork.
- **IMPORTANT — Variants must NOT cross-import.** `on_policy_single_model` must never `import` from `single_model` (and vice versa). Shared pieces → `engine/common/` or `ops/`. Cross-imports drag unrelated files into full-mode export and cause hard-to-debug file-collision / missing-code bugs.
- **Presets inherit engine variant classes**, never `engine/common/*` or `base/*` directly.

## Auto-wrapping (`engine/common/`)

`FSDPModelGroup.__init_subclass__` auto-wraps `build_model()` with `_post_build_model()` (adapter, resume, FSDP2/SP).

`EngineWorkGroup.__init_subclass__` auto-wraps `update_step()` (or `_run_update_step()` for on-policy) with `_pre_update_step()` (seeding, offload load) and `_post_update_step()` (optimizer step, grad clip, offload).

→ **Presets must NOT** perform optimizer step, gradient clipping, parameter offloading, checkpoint save/load, or shared seeding. Those are engine-wrapped.

LM-style presets (SFT/DFT) define their own `update_step` and loss (SP split, next-token CE, chunked loss) in the preset layer. On-policy presets should prefer the `rollout() / compute_reward() / compute_advantage() / compute_loss()` hooks; override `_run_update_step()` only when the task truly needs a different backward schedule (e.g. chunked GRPO loss).

## Frozen ModelGroups

`FSDPModelGroup.save_model_and_processor()` and `save_checkpoint()` early-return when `is_trainable=False`. Reference / teacher models therefore do NOT appear in `checkpoint_*/` subdirs — only `policy_model_group/` does. New on-policy presets must reuse the engine's `reference_model_group` slot for any frozen counterpart (teacher, critic, etc.), not introduce a parallel `model_group_config`.

## Flat config (YAML → nested)

**IMPORTANT — `engine/common/flat_config.py::build_single_model_config_dict` is the ONLY home for the shared trainer / model / optimizer / LoRA / offload / `ray_kwargs` / `log_steps * grad_accum` / `compute_checksums` mapping.** Each preset's `from_flat_dict(d)` MUST delegate to it. Presets only declare:

- their own `data_config` dict (task-specific keys like GRPO's `group_size` / `kl_coef`, video_embedding's `nframes`)
- defaults: `model_default`, `lr_default`, `grad_accum_default`, `log_steps_default`, `task_type_default`

Adding a new public flat key = edit `flat_config.py` once.

## Experiment management

Users configure `outputs_dir` (parent) + `resume_mode: auto|manual`. `output_dir` is set by the framework at experiment creation and must NOT appear in user configs.

- `resume_mode: auto` — scan `outputs/` for the latest experiment, compare `code_hash` + razordl version against `provenance.json`; hash match + incomplete → auto-resume; hash mismatch → new experiment with warning.
- `resume_mode: manual` — always new experiment unless `resume_from` points to an existing one.
- `init_from: <checkpoint-path>` — fork a new experiment from a checkpoint: weights load, optimizer + step counter start fresh, always new experiment dir.
- **Copy recovery:** copying `code/` to another location works because `config.yaml`'s `output_dir` already points to the original experiment dir → `razordl train` resumes from the original checkpoints.

`razordl diff` compares experiment code / config (current vs latest, current vs specified, A vs B). Output shows file changes (A/M/D), YAML config key diffs, and git provenance.
