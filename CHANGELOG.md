# Changelog

All notable changes to RazorDL are documented here.  The project is currently
pre-1.0; APIs may change between minor releases.

## [Unreleased]

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
- Manual vLLM rollout smoke script (`scripts/smoke_vllm_rollout.py`).

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

## [0.1.0]

Initial alpha release.
