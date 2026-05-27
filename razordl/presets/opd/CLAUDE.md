# OPD preset — PG On-Policy Distillation

Loaded when working in `razordl/presets/opd/`.

## Task

Student samples rollouts via vLLM; teacher provides per-token logπ as PPO advantage (no reference-KL penalty).

## Hard rules

- **Inherits `on_policy_single_model.WorkGroup`.** Prefer overriding `rollout() / compute_reward() / compute_advantage() / compute_loss()`.
- **IMPORTANT — `data_config.teacher_model` is REQUIRED.** Student and teacher MUST share the same tokenizer (vocab + special-token ids). `OPDWorkGroup.__init__` enforces this with `_validate_tokenizer_compat()` — fail loudly here rather than emit silently-broken distillation.
- **Teacher path lives on `data_config`, NOT a second `model_group_config`.** `OPDWorkGroup.__init__` deep-copies the policy config and overrides `model_path` / `processor_path` / `is_trainable=False` for the teacher.
- **Teacher reuses engine's `reference_model_group` slot.** Frozen, never persisted to checkpoint subdirs — only the student (policy) appears.
- **Smoke-test signal:** `[OPD sample]` logs show student responses with teacher log-prob advantage; `loss` decreases; `distill_loss` tracks student-teacher KL. See `@docs/testing.md`.
