# Presets — task-specific training recipes

Loaded when working in `razordl/presets/`.

## Directory contract

Every preset directory `razordl/presets/<name>/` must contain:

```
<name>/
  __init__.py          # exports canonical classes + CLI CamelCase aliases
  config.py            # canonical config class (e.g. SFTConfig, GRPOConfig)
  workgroup.py         # canonical WorkGroup + preset-specific model/loss/reward
  _export.py           # AST-based code generation (export_workgroup, export_dataset)
  _export_full.py      # one-line re-export of the engine's export profile
  default_config.yaml  # complete template with ALL params, [常用] / [不常用] sections
  requirements.txt     # auto-synced from pyproject.toml — DO NOT edit manually
```

`dataset.py` is optional if the preset reuses another's dataset (e.g. DFT reuses SFT).

## Naming rules (enforced by CLI auto-discovery)

- CLI converts preset name with `CamelCase = "".join(p.capitalize() for p in preset.split("_"))`.
- Each preset module exposes CLI aliases: `{CamelCase}Config`, `{CamelCase}WorkGroup`, `{CamelCase}Dataset`, `{CamelCase}Collator`.
- Canonical class names may preserve acronyms for readability (`SFTConfig`, `GRPOWorkGroup`); `__init__.py` provides aliases (`SftConfig`, `GrpoWorkGroup`) for the CLI.
- Preset module path is always `razordl.presets.{preset}` (e.g. `razordl.presets.dft`).

## CLI auto-discovery

`cli/__init__.py` scans `razordl/presets/` at startup. Directories starting with `_` are excluded. `--preset` choices and error messages are generated dynamically. **No hardcoded preset names anywhere in CLI code.**

## Adding a new preset

Copy `presets/_template/` and edit files marked `[改]`. The CLI auto-discovers it. **No changes to `cli/__init__.py`, `cli/init.py`, or `cli/train.py` are needed.** See `presets/_template/README.md` for step-by-step instructions.

## Flat config delegation

**`from_flat_dict(d)` MUST delegate to `razordl.core.engine.common.flat_config.build_single_model_config_dict()`** for the shared trainer / model / optimizer / LoRA / offload / `ray_kwargs` mapping. Presets only declare:

- their own `data_config` dict (task-specific keys)
- defaults: `model_default`, `lr_default`, `grad_accum_default`, `log_steps_default`, `task_type_default`

**Never copy the trainer / model / optimizer mapping into a preset.** Adding a new shared flat key = edit `engine/common/flat_config.py` once.

## CLI integration

The CLI auto-discovers presets via the rules in `@razordl/cli/CLAUDE.md`. As long as your preset follows the directory contract + naming rules above, no CLI changes are needed. From a preset author's perspective the contract is: expose `{CamelCase}Config` / `WorkGroup` / `Dataset` / `Collator` aliases in `__init__.py`, and the CLI will find them.

## Model profile enforcement

Every preset's `build_model()` MUST go through `razordl.ops.model.huggingface.enforce_model_profile(model_path)`. LM-style presets get this automatically through `build_causal_lm`; multimodal presets call it explicitly. See `@razordl/ops/model/profiles/CLAUDE.md` for adding support for a new model family.

## Generated code self-containment

In `custom` and `full` modes, generated `workgroup.py` / `dataset.py` must be readable end-to-end. **No jumping to parent classes for behavior.** AST extraction inlines what's needed.

## Docstrings in SFT source files

Docstrings in `presets/sft/` describe SFT specifically. When other presets reuse SFT classes via AST export (DFT, OPD), they must strip/replace SFT-specific docstring text in the generated output. **Never modify SFT source files directly** to accommodate other presets.

## requirements.txt is auto-generated

Top comment says "Auto-generated from pyproject.toml — Do not edit manually". To add a preset-specific dep:

1. Add it to `pyproject.toml` → `[project.optional-dependencies].<preset>` (extras name MUST match preset name exactly).
2. Commit — the pre-commit hook (`scripts/sync_requirements.py`) syncs `presets/<preset>/requirements.txt`.
