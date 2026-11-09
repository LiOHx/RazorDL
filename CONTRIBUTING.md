# Contributing to RazorDL

Thanks for helping improve RazorDL.  This project is a distributed training
framework; changes that look local can affect full-mode export, checkpoint
compatibility, or multi-GPU behavior.

## Setup

```bash
git clone https://github.com/LiOHx/RazorDL.git
cd RazorDL
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

Python 3.10+ is supported.  PyTorch >= 2.6 is required for the FSDP2 API.

## Checks before opening a PR

```bash
ruff check .
pytest -m "not integration"
pytest -m integration   # tiny-model end-to-end SFT smoke tests
```

`pytest` with no marker runs both suites.  GPU-only tests skip automatically
when no CUDA device is present.  The 2-GPU FSDP2 + SP script is manual:

```bash
python scripts/smoke_fsdp2_sp.py
```

## Architecture rules

Read the nearest `AGENTS.md` before editing a subtree.  The hard rules that
matter most:

- Dependency direction is a tree: `base -> engine/common -> engine variant ->
  preset`.  Never cross-import engine variants or presets from different
  engines; put shared code in `engine/common/` or `ops/`.
- Engine code is framework infrastructure, not training logic.  Concrete
  losses, forward passes, rollout/reward logic, and SP split logic belong in
  presets or `ops/`.
- Presets inherit engine variant classes, never `base/` or `engine/common/`
  directly.
- Public flat-config keys are mapped in exactly one place:
  `engine/common/flat_config.py::build_single_model_config_dict`.  Do not copy
  the trainer/model/optimizer mapping into a preset.
- CLI code must not hardcode preset names or import preset packages at module
  level.
- Keep `tests/test_full_export.py` green; full-mode export is a load-bearing
  feature.

## Adding a preset

Copy `razordl/presets/_template/` and follow the comments.  The CLI discovers
new presets by scanning `razordl/presets/`; no CLI changes are required.
Update `pyproject.toml` optional dependencies if the preset needs new packages,
then run `python scripts/sync_requirements.py` (the pre-commit hook does this
automatically when `pyproject.toml` changes).

## Tests

- Unit tests live in `tests/` and should not download models or require a GPU
  unless guarded by `pytest.mark.skipif`.
- Integration tests are marked `@pytest.mark.integration`; they may build tiny
  local models but must not require network access.
- Add a regression test for every bug fix, especially around resume,
  checkpointing, config plumbing, and distributed reductions.

## Commit messages

Use `type: short description` (for example `fix: honor dataset_processor_path`).
Keep changes scoped; do not mix unrelated refactors with behavior fixes.
