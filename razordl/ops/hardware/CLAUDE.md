# Hardware detection — extendable device backends

Loaded when working in `razordl/ops/hardware/`.

## Files

- `cuda.py` — CUDA-specific: version, driver compatibility, recommended PyTorch index, **runtime capability probes** (compute capability, native bf16, flash-attn 2, memory)
- `device.py` — dispatches to backends; `get_available_device()` returns `"cuda"` / `"mps"` / `"cpu"`; `get_torch_device()` / `get_device_id()` are the ONLY device-namespace helpers — `ops/parallel/*` and `ops/distributed/*` import them from here instead of keeping local copies (three verbatim copies once drifted, one frozen at import time)
- `precision.py` — backend-**independent** policy on top of the probes: `resolve_precision()`, `to_torch_dtype()`, `needs_grad_scaler()`, `needs_fp32_master_weights()`
- `check_device_compatibility()` — called at training startup; raises `RuntimeError` with install guidance

## Adding a new backend (ROCm / XPU / …)

1. Create a new file (e.g. `rocm.py`) following the `cuda.py` shape.
2. Add **one line** per dispatch function in `device.py`.
3. Never touch `precision.py` — it is policy, not capability.

## Hard rules

- **IMPORTANT — Use `importlib.import_module` for lazy backend loading.** Top-level imports cause circular imports during package init.
- **IMPORTANT — Capability criteria are compute capability, never `torch.cuda.is_bf16_supported()`.** That call defaults to `including_emulation=True` and returns True on Turing (sm_75), where bf16 is emulated in software, so it cannot answer "is bf16 native here?" — use `get_device_capability()[0] >= 8`.
- **Emulated bf16 is a supported target, not a fallback to route around.** `resolve_precision("auto")` returns bf16 on every CUDA GPU; on sm_75 the emulation measured faster and lighter than the only correct fp16 path (fp32 masters + autocast) — numbers in `precision.py`. Change that policy only against a fresh measurement.
