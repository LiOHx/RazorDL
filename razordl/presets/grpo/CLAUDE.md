# GRPO preset — Group Relative Policy Optimization

Loaded when working in `razordl/presets/grpo/`.

## Task

Math RL. vLLM rollout → reward (0.0 / 0.5 / 1.5) → group-relative advantage → PPO loss with chunked backward.

## Hard rules

- **Inherits `on_policy_single_model.WorkGroup`.** Override `rollout() / compute_reward() / compute_advantage() / compute_loss()` for task logic. Override `_run_update_step()` ONLY because GRPO needs a chunked backward schedule (per-group sub-batches for memory).
- **IMPORTANT — Reward / advantage / KL metrics use `DistStats.from_tensor`,** NEVER pre-aggregated scalars. See `@razordl/core/base/CLAUDE.md` for why pre-aggregation corrupts `min`/`max`/`std`.
- **Reference model occupies engine's `reference_model_group` slot.** Frozen (`is_trainable=False`), not persisted to checkpoint subdirs.
- **Smoke-test signal:** `reward_mean` should trend > 0; loss decreases; `clip_fraction` stays near 0. See `@docs/testing.md`.
