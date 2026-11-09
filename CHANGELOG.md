# Changelog

All notable changes to RazorDL are documented here.  The project is currently
pre-1.0; APIs may change between minor releases.

## [0.1.1] - 2026-11-09

Documentation-only follow-up to the initial alpha release.

### Changed

- Renamed every `CLAUDE.md` to `AGENTS.md` (root + 9 nested subtree files) and
  updated all references in docs, code comments, and tests.
- Clarified README vLLM setup: `.[grpo]` / `.[opd]` do not install vLLM.
- Fixed the custom-loss example signature (`shifted` keyword) and documented
  the distributed-token-normalization contract.
- Documented the vLLM LoRA weight-sync limitation and corrected the
  full-project line count (~8K).

## [0.1.0] - 2026-11-09

Initial alpha release.

### Added

- GitHub Actions CI: lint + unit tests on Python 3.10/3.11/3.12, plus a
  separate integration job.
- `LICENSE` (MIT), `CONTRIBUTING.md`, and development extras in
  `pyproject.toml`.
- `ruff` and `pytest` configuration; pre-commit hook for `ruff`.
- Tiny-model end-to-end SFT integration tests (train, resume, `init_from`).
- Manual 2-GPU FSDP2 + Ulysses SP smoke script.
- `cache_tokenized` option to disable the SFT tokenized dataset cache.
- `streaming: true` JSONL dataset option (byte-offset index, tokenize on
  demand) for SFT/DFT.
- Multi-node Ray documentation and cluster-GPU worker planning.
- Fused linear CE tile-size benchmark script and performance notes.
- Ulysses SP chunked-attention benchmark script.
- SFT DataLoader mode/worker benchmark script.
- End-to-end SFT step-time/peak-memory benchmark script.
- Manual vLLM rollout smoke script (`scripts/smoke_vllm_rollout.py`).
- Full GRPO/OPD vLLM training smoke scripts (`smoke_grpo_vllm.py`,
  `smoke_opd_vllm.py`); both verified with Qwen3-0.6B.
- Project-local `.venv-vllm` (vLLM 0.32.0+cu129 / torch 2.13.0+cu129)
  verified with Qwen3.5-0.8B/4B; full test suite passes there.
- Env overrides for direct vLLM smoke (`RAZORDL_VLLM_*`).

### Fixed

- `dataset_processor_path` is now honored by SFT/DFT/GRPO/OPD/video_embedding
  datasets and folded into the canonical `processor_config.processor_path`.
- The SFT tokenized cache key now includes the resolved tokenizer path.
- Multi-node Ray jobs request the cluster GPU count instead of only the
  driver's local GPU count.
- `scripts/sync_requirements.py` works on Python 3.10 via a `tomli` fallback.
- Pin `fsspec[http]<=2025.3.0` for datasets 4.x compatibility; `uv pip check`
  is clean again.

### Removed

- `optimizer_path` / `scheduler_path` from default configs; setting either now
  raises a clear `NotImplementedError` instead of being silently ignored.
- `adapter_name` from the public adapter config (PEFT always uses `"default"`).
- `worker_group_name` from default configs (field kept as reserved).

### Documentation

- README development section, CI/Python badges, and license link.
- `docs/testing.md` automated smoke-test instructions.
- `docs/improvement-plan.md` with the prioritized roadmap.
- `docs/multi-node.md`.
- `docs/vllm.md` with the tested vLLM 0.32 / torch 2.13 / Python 3.11
  environment and the rollout smoke instructions (Qwen3.5-0.8B and 4B).
