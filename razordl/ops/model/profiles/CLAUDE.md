# Model profiles — HF model family support registry

Loaded when working in `razordl/ops/model/profiles/`.

## What it does

Looks up a checkpoint's `cfg.model_type` and either returns a (possibly-adjusted) `PreTrainedConfig` for `from_pretrained(config=...)` or raises `UnsupportedModelError` BEFORE the deep stack crashes on an unsupported architecture.

## Adding a new model family

1. Create `razordl/ops/model/profiles/<model_type>.py` with a `@register`-decorated `ModelProfile` subclass.
2. Add an **absolute-import line** in `profiles/__init__.py`.

**IMPORTANT — Use absolute imports, NOT relative.** The full-mode export AST walker breaks on relative imports inside `profiles/__init__.py`.

## Hard rules

- **IMPORTANT — Every preset's `build_model()` MUST go through `enforce_model_profile(model_path)`** (defined in `razordl/ops/model/huggingface.py`). This prevents deep-stack crashes from unsupported models. LM-style presets get it automatically through `build_causal_lm`; multimodal presets (e.g. `video_embedding`) call it explicitly.
- **Profiles are stateless and idempotent.** They only handle HF-load-time concerns (config preparation, hard precondition checks). **Never put training logic in a profile.**
