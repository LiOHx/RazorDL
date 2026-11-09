# RazorDL improvement plan

Status: P0–P3 mostly implemented (remaining: end-to-end performance probe and
a release tag); P2 still has the optional vLLM smoke.

Baseline (verified 2026-11-09):

- `pytest -q` → 186 passed, 1 skipped (optional `PIL` missing)
- CLI discovery works: `dft, grpo, opd, sft, video_embedding`
- No CI workflow, no `LICENSE` file, no dev extras
- Distributed/GPU tests are mocked or single-process; no real multi-GPU test
- Several public config keys are documented but unused (see P1)

After P0–P3 changes:

- `pytest` → 210 passed, 1 skipped (includes 3 integration smoke tests)
- `pytest -m "not integration"` → 207 passed, 1 skipped
- `pytest -m integration` → 3 passed
- CPU-only (`CUDA_VISIBLE_DEVICES="" pytest`) → 204 passed, 7 skipped
- `ruff check .` and `pre-commit run --all-files` pass
- `scripts/smoke_fsdp2_sp.py` is syntax-checked but not executed on this box
  (1 CUDA GPU; it requires 2)

Priorities:

- **P0** — safety net: make the repo testable and enforceable in CI
- **P1** — correctness/consistency: stop silently ignoring documented config
- **P2** — confidence: real integration tests for the paths that break in production
- **P3** — scale/polish: performance, docs, release hygiene

Each task should keep `tests/test_full_export.py` green (full-mode export is a
load-bearing feature).

---

## P0 — Safety net (highest value, low risk)

- [x] **Add `LICENSE`** (MIT, copyright LiOHx). `pyproject.toml` and README
      already declare MIT; the repository currently has no license text.
- [x] **Add dev dependencies** in `pyproject.toml`:
      `[project.optional-dependencies].dev = ["pytest>=8", "pytest-timeout>=2",
      "ruff>=0.6", "pre-commit>=3"]`.
- [x] **Add pytest config** in `pyproject.toml`:
      `[tool.pytest.ini_options] testpaths = ["tests"]`, `addopts = "-q"`.
- [x] **Add ruff config** in `pyproject.toml` (`target-version = "py310"`,
      conservative rule set `E4/E7/E9/F`; `E501`, `E402`, `E741` ignored until
      the existing codebase is cleaned up; `I` deferred).
- [x] **Add `.github/workflows/ci.yml`**:
      - matrix: Python 3.10 / 3.11 / 3.12
      - install CPU torch first (`--index-url https://download.pytorch.org/whl/cpu`)
      - `pip install -e ".[dev]"`
      - `ruff check .`
      - `pytest`
      - GPU-only tests skip automatically (`torch.cuda.is_available()` guards).
- [x] **Add pre-commit hooks** for `ruff check` (and optionally `ruff format`);
      keep the existing `sync-requirements` hook.
- [x] **Add README badges** (CI, Python versions) and a short “Development”
      section with `pip install -e ".[dev]"`, `pytest`, `ruff check .`.

Acceptance:

- CI is green on a fresh PR.
- `pytest -q` and `ruff check .` both pass locally from a clean `.venv`.
- No source behavior changes.

---

## P1 — Config correctness (stop silently ignoring documented keys)

- [x] **Fix `dataset_processor_path`** (real bug):
      - `SFTDataset.__init__` always loaded the tokenizer from
        `model_config.model_path`; the documented `dataset_processor_path`
        (and `processor_config.processor_path`) was ignored on the dataset side.
      - `flat_config.py` now folds `dataset_processor_path` into
        `processor_config.processor_path`; `resolve_processor_path()` in
        `ops/model/huggingface.py` is the single resolver used by SFT, DFT
        (via SFTDataset), GRPO, OPD, and video_embedding.
      - `SFTDataset._cache_hash()` now hashes the resolved tokenizer path.
      - Tests: `tests/test_processor_path.py` and `tests/test_sft_dataset.py`.
- [x] **Decide `optimizer_path` / `scheduler_path`**:
      - Removed from all default configs; setting either now raises a clear
        `NotImplementedError` in `flat_config.py` instead of being ignored.
- [x] **Decide `adapter_name`**:
      - Removed from `BaseAdapterConfig` and `flat_config.py`; PEFT already
        hard-codes `"default"`, so the key never did anything.
- [x] **Audit remaining public keys**:
      - `worker_group_name` removed from default configs; field kept with a
        “reserved” comment for future multi-model work.
      - `micro_batch_size_per_gpu` remains because video_embedding reads it.
- [x] **Add default-config parse coverage** for every preset
      (`tests/test_default_configs.py`), including presets whose `__init__.py`
      needs optional heavy dependencies.
- [ ] **Full static config-consumption audit**: every key in a preset's
      `default_config.yaml` must be consumed by `flat_config.py` or the
      preset, or be explicitly marked deprecated. Regression tests for the
      known dead keys are in place, but the full audit is still TODO.

Acceptance:

- No documented config key is silently ignored.
- A test covers the `dataset_processor_path` path end to end (dataset + model
  group + cache hash).
- `pytest tests/test_full_export.py` stays green.

---

## P2 — Integration confidence (real paths, not mocks)

- [x] **Tiny-model SFT smoke test**:
      - `tests/test_sft_smoke.py` builds a tiny Qwen3 checkpoint/tokenizer in
        `tmp_path`, runs the real trainer for 4 steps with
        `parallel_backend: ddp` on CPU, and asserts checkpoints,
        `checkpoint_info.json`, model weights, optimizer state, and finite
        loss values.
      - Marked `@pytest.mark.integration`; the main CI job runs
        `pytest -m "not integration"`, and a separate integration job runs
        `pytest -m integration`.
- [x] **Multi-GPU smoke script** (`scripts/smoke_fsdp2_sp.py`):
      - Self-contained: builds a tiny model/tokenizer/data, launches
        `torch.distributed.run --nproc_per_node=2`, trains 2 steps with
        `parallel_backend: fsdp2` + `sp_size: 2`, and verifies checkpoints.
      - Documented in `docs/testing.md`; not run in GitHub-hosted CI (needs
        2 CUDA GPUs).  The current dev box has 1 GPU, so this script is
        syntax-checked but not executed here.
- [ ] **GRPO/OPD vLLM smoke test** (optional, manual):
      - Tiny policy model + `SKIP_VLLM=1` HF fallback first, then a vLLM run.
      - Assert reward/loss metrics are produced and only the policy
        checkpoint is written.
- [x] **Experiment-management integration test**:
      - `tests/test_sft_smoke.py::test_resume_from_checkpoint_continues` runs
        a real 2-step job, resumes from `checkpoint_000002`, and asserts the
        step log continues at 3–4.
      - `test_init_from_fork_starts_at_step_zero` forks weights from
        `checkpoint_000001` into a new experiment and asserts the step counter
        restarts at 1.

Acceptance:

- At least one real (non-mocked) single-process training run is automated.
- Multi-GPU smoke script is documented and reproducible on a 2-GPU machine.

---

## P3 — Scale, performance, release polish

- [x] **Streaming JSONL dataset** for SFT:
      `streaming: true` builds a byte-offset index and tokenizes one line per
      `__getitem__`, so the raw dataset is not held in RAM.  It remains
      map-style (shuffling/resume work); a full `IterableDataset` path and
      non-JSONL streaming are future work.  Default stays `streaming: false`.
- [x] **Dataset cache policy**:
      `cache_tokenized: false` disables both reading and writing the
      tokenized cache; documented in the SFT/DFT default configs.
- [x] **Multi-node Ray docs and worker plan**:
      `docs/multi-node.md` covers `ray_kwargs`, shared filesystem
      assumptions, code availability, NCCL, and checkpoint visibility.
      `_ray_worker_plan()` now requests `ray.cluster_resources()["GPU"]`
      workers instead of only the driver's local count.
- [x] **Add `CONTRIBUTING.md` / `CHANGELOG.md`**:
      release tagging is still pending; `docs/release.md` records the
      pre-tag checks (2-GPU and vLLM verification are the current blockers).
- [x] **Fused CE tile-size benchmark**:
      `scripts/benchmark_fused_linear_ce.py` and `docs/performance.md`
      (sample table from an RTX 2080 Ti).
- [x] **SP chunked-attention benchmark**:
      `scripts/benchmark_sp_attention.py` and the sample table in
      `docs/performance.md`.
- [x] **SFT DataLoader mode/worker benchmark**:
      `scripts/benchmark_sft_dataloader.py` and the sample table in
      `docs/performance.md`.
- [ ] **End-to-end step-time probe** for a real SFT/GRPO config.

---

## Suggested execution order

1. P0 in one PR (`chore: add license, dev extras, ruff, CI`).
2. P1 in one or two PRs (`fix: honor dataset_processor_path`,
   `chore: remove unimplemented optimizer/scheduler config`).
3. P2 tiny-model smoke test, then the multi-GPU script.
4. P3 as needed.

## Out of scope for now

- vLLM V1 LoRA weight sync (known limitation; needs a separate design)
- On-policy `sp_size > 1` (engine intentionally rejects it)
- DPO preset
- Full data-encryption feature (discussed separately; not part of this plan)
