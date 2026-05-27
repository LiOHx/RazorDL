# Hardware detection — extendable device backends

Loaded when working in `razordl/ops/hardware/`.

## Files

- `cuda.py` — CUDA-specific: version, driver compatibility, recommended PyTorch index
- `device.py` — dispatches to backends; `get_available_device()` returns `"cuda"` / `"mps"` / `"cpu"`
- `check_device_compatibility()` — called at training startup; raises `RuntimeError` with install guidance

## Adding a new backend (ROCm / XPU / …)

1. Create a new file (e.g. `rocm.py`) following the `cuda.py` shape.
2. Add **one line** in `device.py` to dispatch to it.

## Hard rule

- **IMPORTANT — Use `importlib.import_module` for lazy backend loading.** Top-level imports cause circular imports during package init.
