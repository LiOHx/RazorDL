# Base layer — abstract interfaces

Loaded when working in `razordl/core/base/`.

## What lives here

Stable contracts only: `BaseTrainer`, `BaseWorkGroup`, `BaseModelGroup`, `BaseConfig`, the `checkpoint_info` schema, and the `Reducible` metrics ABC. **Never put concrete training logic here** — that belongs in engine variants or presets.

## Hard rules

- **IMPORTANT — `set_seed` is the ONLY seeding entry point.** `BaseTrainer.set_seed(seed)` seeds Python/numpy/torch/CUDA. It is called before model init (for deterministic LoRA/weight init) and before every training step via `_pre_update_step`. Never add ad-hoc `random.seed` / `torch.manual_seed` calls elsewhere — import and call `set_seed`.
- **IMPORTANT — Auto-wrap uses a depth counter** (`AutoSetModelGroupNameWorkGroup` in `workgroup.py`). Every subclass `__init__` is wrapped, but `__post_init__` fires only when the **outermost** wrapper returns (depth → 0). A naive one-shot guard would fire before `self.model_group` is created in subclasses like `SFTWorkGroup` (which create it AFTER `super().__init__()` returns), and `auto_set_model_group_name` would silently walk an empty set.

## Reducible metrics (`metrics.py`)

`step_info` returned from `update_step` is gathered cross-rank by `_summarize_step_info` in `trainer.py`. Plain scalars get mean-reduced — fine.

**Distribution-shaped fields (where min/max/std are meaningful) MUST NOT be pre-aggregated on a rank.** "Mean of per-rank max" is meaningless and produces values outside the true sample set (e.g. GRPO `reward.max=1.0` when rewards are `{0.0, 0.5, 1.5}`).

Use `DistStats.from_tensor(t)`: it emits `{sum, sum_sq, n, min, max}` and unfolds to `{mean, std, min, max, n}` after merge (`std` is population std). Adding a new distribution kind (histogram, ratio, weighted mean) = subclass `Reducible`; the aggregator is not touched.

`_summarize_step_info` handles a single `Reducible` leaf AND a list-of-same-type `Reducible` (the post-`all_gather_object` shape).

**IMPORTANT — Gathering happens EXACTLY ONCE in `BaseTrainer.run_training_loop`.** Do NOT add `all_gather_object` in `EngineTrainer.update_step` — `DistStats.n` would inflate by `world_size`.

## Checkpoint metadata (`checkpoint_info.py`)

`checkpoint_info.json` (rich metadata) replaces the legacy single-integer `complete` file. Schema: `completed_step`, `timestamp`, `elapsed_seconds`, `topology` (`world_size` / `sp_size`), `seed`, `provenance` (`framework_version`, `git_commit`, `config_hash`), `metrics` (latest aggregated `step_info`), `files` (size + optional SHA256), `resumed_from`, `kind` (`checkpoint` or `model_only`).

- `is_complete()` accepts BOTH `checkpoint_info.json` AND legacy `complete` for backwards compatibility.
- `_check_topology_compat()` **warns**, does not raise, on `world_size` / `sp_size` mismatch on resume.
- `trainer_config.compute_checksums: bool = False` — opt-in SHA256 over model files at save time; file sizes are always recorded.
- `framework_version` is `"razordl"` in this repo. Full-mode exports rewrite the brand constant to `None`, so generated projects log `"unknown"` and never advertise the upstream package name.
