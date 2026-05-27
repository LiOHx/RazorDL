# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository. This is the **root** contract; subdirectory `CLAUDE.md` files extend it for their subtree.

## How to maintain CLAUDE.md (META — read before editing ANY CLAUDE.md)

This project uses **nested CLAUDE.md files**: a thin root + one per subtree + on-demand long docs in `docs/`. Read these rules before touching any of them.

**Loading model.**

- This root file is loaded **every** conversation.
- Subdirectory `CLAUDE.md` (e.g. `razordl/core/engine/CLAUDE.md`) is loaded **only** when Claude touches a file under that subtree.
- `@docs/*.md` references are loaded **only** when explicitly needed.

**Where a new rule belongs (apply this scope test in order):**

1. Does it constrain ≥ 2 sibling subtrees, or describe global identity / environment? → **this root file**.
2. Does it only apply to ONE subtree? → **that subtree's `CLAUDE.md`**.
3. Is it a long workflow / runbook (smoke tests, env troubleshooting)? → **`docs/<name>.md`**, then `@import` from the relevant `CLAUDE.md`.

**Loading-path sanity check (MANDATORY before saving any rule).** For every new or moved rule, answer both:

1. **Which file paths does this rule actually constrain?** (i.e. where does the buggy code get written if someone violates the rule?)
2. **When Claude edits one of those paths, will it load this rule?** (Root is always loaded; a subtree `CLAUDE.md` is loaded only when Claude touches files under that subtree.)

If the answer to (2) is "no", the rule is in the wrong place — move it. Common trap: a rule "about how presets are dispatched" placed in `presets/CLAUDE.md`, but the *code* implementing dispatch lives in `cli/init.py` and `cli/train.py`, so Claude editing CLI never sees the rule.

**Writing a rule.**

- **One sentence, one rule.** Must answer: **"what specific bug or test failure appears if this rule is deleted?"** If you cannot answer, do not add it.
- **Born from a real mistake, not imagination.** Add a rule the moment you catch Claude (or a human) making the mistake the rule would prevent — ideally on the **second** occurrence. Speculative rules ("in case someone tries to…", "we might want to…") inflate every conversation's context for problems that may never occur. CLAUDE.md is a record of actual past failures, not a defensive moat.
- **Never write a naked "Never X" — always pair the prohibition with the correct alternative** (e.g. "Never cross-import variants → promote shared code to `engine/common/`"). A bare Never traps Claude the moment it believes it must X; the alternative gives it an exit.
- **Use imperative voice; do not restate the README.** Imperatives ("Use `uv pip`") are stronger than passives ("you should use uv pip"). Claude already has the codebase — it doesn't need your value-prop or onboarding narrative, only the constraints README cannot express.
- Mark cross-cutting hard rules with **IMPORTANT**.

**Editing a rule.** Never duplicate the same rule in two `CLAUDE.md` files — promote to the closest common ancestor. When you change code that a rule references, update the rule in the **same commit**.

**Creating a new `CLAUDE.md`.** Only create one when the directory has ≥ 3 unique rules that don't apply outside its subtree. Filename must be exactly `CLAUDE.md` (case-sensitive). Place it at the directory root.

**Deleting a rule.** Delete (don't "soften") when: (a) the constraint is now enforced by code or tests, (b) architecture changed and it no longer applies, or (c) violating it produces no observable effect.

**Periodic audit (every ~3 months, or after any major refactor).** Open every `CLAUDE.md` and ask three questions per rule: (a) is it still true? (b) is it a rule, or a story? (c) is it specific enough to enforce / verifiable in code? Delete on any "no". **Rules referencing paths, scripts, or APIs that no longer exist are the highest-risk category** — they actively mislead Claude into producing wrong code, because Claude trusts the rule over the codebase. Remove them even if they read as "useful in spirit".

**Size budgets (hard limits).** Root ≤ 120 lines. Each nested `CLAUDE.md` ≤ 80 lines. `docs/*.md` may be longer (loaded on demand).

## Project identity

RazorDL is a distributed LLM / multimodal training framework. `razordl init --preset <name> --mode <mode>` generates a project in one of three tiers:

| Mode | Generated output | Runtime dep |
|------|------------------|-------------|
| `simple` | `config.yaml` + `run.sh` + `data/` | needs `razordl` installed |
| `custom` | adds editable `src/{dataset,workgroup,main}.py` | needs `razordl` installed |
| `full`   | self-contained `src/` with all framework code inlined | **zero** dep on `razordl` |

`run.sh` is **always** generated in code by `init.py` — never copied from a template.

## Directory map (with subtree contracts)

```
razordl/
  core/
    base/                       → @razordl/core/base/CLAUDE.md
    engine/
      common/                   shared lifecycle (FSDP2, LoRA, resume, optimizer)
      single_model/             supervised engine (SFT / DFT / video_embedding)
      on_policy_single_model/   RL engine (GRPO / OPD) — vLLM rollout
                                → @razordl/core/engine/CLAUDE.md
    export/                     → @razordl/core/export/CLAUDE.md
  ops/                          razordl-independent utilities
    hardware/                   → @razordl/ops/hardware/CLAUDE.md
    model/profiles/             → @razordl/ops/model/profiles/CLAUDE.md
    {distributed,parallel,model,loss,multimodal,snapshot}/  stateless utils
  presets/                      → @razordl/presets/CLAUDE.md
    sft/ dft/ grpo/             (grpo:  @razordl/presets/grpo/CLAUDE.md)
    opd/ video_embedding/       (opd:   @razordl/presets/opd/CLAUDE.md)
    _template/                  copy this when adding a new preset
  cli/                          → @razordl/cli/CLAUDE.md  (init, train, ckpt, diff)
```

## Cross-cutting hard rules

- **IMPORTANT — Dependency direction is a tree, not a graph.** Vertical inheritance is allowed (`base → engine/common → engine variant → preset`); cross-branch imports are **forbidden** — `on_policy_single_model` must never `import` from `single_model`, and vice versa. Presets under different engines must not cross-import either. Shared code goes to `engine/common/` or `ops/`. Violations silently corrupt full-mode export and cause file collision / missing-code bugs.
- **IMPORTANT — No hardcoded preset names anywhere in CLI code.** Use `os.listdir("razordl/presets")` + the CamelCase convention (`"".join(p.capitalize() for p in preset.split("_"))`). Directories starting with `_` are excluded.
- **IMPORTANT — Never import preset packages at module level in CLI / init code.** Triggers heavy deps (tensordict, vllm). Use `importlib.util.spec_from_file_location` to load `_export.py` by file path; use `importlib.import_module(f"razordl.presets.{preset}")` at call time inside `train.py`.
- **Architecture hierarchy is absolute** (rules live in `@razordl/core/engine/CLAUDE.md`): presets inherit engine variant classes, never `engine/common/*` or `base/*` directly. Engine bugs are fixed in `engine/common/` — single source of truth for FSDP / LoRA / resume / optimizer / grad-clip / offload / seeding.

## Development environment

- **uv-managed venv at `.venv/`.** Activate with `source .venv/bin/activate`. Install with `uv pip install <pkg>` (not bare `pip`). Python: `.venv/bin/python`.
- **PyTorch minimum: `>=2.6.0`** (FSDP2 API stabilized). Enforced in `pyproject.toml`.
- **Dependency single source of truth:** `pyproject.toml` → `[project.dependencies]` (shared core) + `[project.optional-dependencies]` (per-preset extras, name MUST match preset). `scripts/sync_requirements.py` (pre-commit hook) regenerates `presets/*/requirements.txt`. **Never hand-edit those files.**
- **flash-attn ABI / Ray dashboard / extras install:** see `@docs/env-troubleshooting.md`.

## Testing

Smoke-test recipes for SFT / GRPO / OPD / video_embedding live in `@docs/testing.md`. Inspect any checkpoint with `razordl ckpt info <dir>` (pretty-prints `checkpoint_info.json`; accepts legacy `complete`-only checkpoints too).

## Git workflow

- Branch: `master`, push to `origin/master`.
- Commit message: `type: short description`.
- Append `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` to every commit.
