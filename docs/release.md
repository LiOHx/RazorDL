# Release checklist

Use this before creating a version tag.  The project is pre-1.0; a tag is a
public promise that the listed checks passed on the stated hardware.

## Before tagging

- [x] `master`/`main` is clean and matches `origin`.
- [x] Local CI-equivalent checks pass (GitHub Actions result not checked; no
      `gh` client on the dev box).
- [x] Integration tests pass (`pytest -m integration`).
- [x] `pytest -m "not integration"` and `pytest` pass locally.
- [x] `ruff check .` and `pre-commit run --all-files` pass.
- [x] Full-mode export test passes in `.venv-vllm` (PIL installed there).
- [ ] 2-GPU smoke test passes on a real 2-GPU host:
      `python scripts/smoke_fsdp2_sp.py` — **accepted exception for the
      v0.1.0 alpha tag; no 2-GPU host available.**
- [x] vLLM rollout smoke passes (`scripts/smoke_vllm_rollout.py`; verified
      with vLLM 0.32 + Qwen3.5-0.8B on a single RTX 2080 Ti).
- [x] If releasing GRPO/OPD, run a full preset with vLLM rollout + weight
      sync and check reward/loss/checkpoint behavior (Qwen3-0.6B; GRPO fp16,
      OPD fp32).
- [x] Check the known limitations in the README / preset docs are still
      accurate.
- [x] Move `CHANGELOG.md` entries from `[Unreleased]` to the release version
      with the date (`0.1.0`, 2026-11-09).
- [x] Version is already `0.1.0` in `pyproject.toml` and
      `razordl/__init__.py`.
- [x] Tag and push:

```bash
git tag -a v0.1.0 -m "RazorDL v0.1.0 (alpha)"
git push origin v0.1.0
```

- [ ] Create the GitHub release from the tag and paste the changelog section.

## Current status (v0.1.0 alpha)

- CI, unit tests, integration smoke tests, and single-GPU FSDP2/SP code paths
  are covered.
- The 2-GPU `scripts/smoke_fsdp2_sp.py` has not been run on this dev box
  (1 GPU only).  This is an accepted exception for the v0.1.0 alpha tag.
- vLLM rollout generation and full GRPO/OPD training with weight sync are
  verified with Qwen3-0.6B in a separate vLLM 0.32 environment.  Qwen3.5
  multimodal full training needs more VRAM than the 22 GB test card.
- `video_embedding` needs optional `PIL`/`decord`/`torchvision`; the full
  export test skips when those are absent.

Treat v0.1.0 as an alpha tag: APIs may still change, and multi-node / vLLM
paths are documented but not CI-enforced.

## v0.1.1 (2026-11-09)

Documentation-only follow-up: `AGENTS.md` rename + README clarifications.
No runtime code changes; v0.1.0 remains the initial alpha release.
