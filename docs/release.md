# Release checklist

Use this before creating a version tag.  The project is pre-1.0; a tag is a
public promise that the listed checks passed on the stated hardware.

## Before tagging

- [ ] `master`/`main` is clean and matches `origin`.
- [ ] CI is green (lint + unit tests on Python 3.10/3.11/3.12).
- [ ] Integration job is green (`pytest -m integration`).
- [ ] `pytest -m "not integration"` and `pytest` pass locally.
- [ ] `ruff check .` and `pre-commit run --all-files` pass.
- [ ] Full-mode export test passes (`tests/test_full_export.py`); install the
      preset's optional deps if the test skips.
- [ ] 2-GPU smoke test passes on a real 2-GPU host:
      `python scripts/smoke_fsdp2_sp.py`.
- [x] vLLM rollout smoke passes (`scripts/smoke_vllm_rollout.py`; verified
      with vLLM 0.32 + Qwen3.5-0.8B on a single RTX 2080 Ti).
- [ ] If releasing GRPO/OPD, run a full preset with vLLM rollout + weight
      sync and check reward/loss/checkpoint behavior.
- [ ] Check the known limitations in the README / preset docs are still
      accurate.
- [ ] Move `CHANGELOG.md` entries from `[Unreleased]` to the release version
      with the date.
- [ ] Bump `version` in `pyproject.toml` and `razordl/__init__.py` together.
- [ ] Tag and push:

```bash
git tag -a vX.Y.Z -m "RazorDL vX.Y.Z"
git push origin vX.Y.Z
```

- [ ] Create the GitHub release from the tag and paste the changelog section.

## Current status (v0.1.0 candidate)

- CI, unit tests, integration smoke tests, and single-GPU FSDP2/SP code paths
  are covered.
- The 2-GPU `scripts/smoke_fsdp2_sp.py` has not been run on this dev box
  (1 GPU only).
- vLLM rollout generation is verified in a separate vLLM 0.32 environment;
  full GRPO/OPD training with vLLM weight sync is still unverified.
- `video_embedding` needs optional `PIL`/`decord`/`torchvision`; the full
  export test skips when those are absent.

Treat v0.1.0 as an alpha tag: APIs may still change, and multi-node / vLLM
paths are documented but not CI-enforced.
