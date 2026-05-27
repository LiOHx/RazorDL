# Environment troubleshooting

Long-form fixes for environment issues. Loaded on demand via `@docs/env-troubleshooting.md`.

## Adding dependencies (uv)

Always use `uv pip` inside the venv:

```bash
source .venv/bin/activate
uv pip install torchvision --index-url https://download.pytorch.org/whl/cu126
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
```

## flash-attn ABI mismatch

`flash-attn` must be compiled against the installed PyTorch version. If you see `undefined symbol: _ZN3c104cuda...`, the ABI is mismatched. Fix:

```bash
uv pip uninstall flash-attn
uv pip install "torch==2.6.0+cu126" --index-url https://download.pytorch.org/whl/cu126
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
```

## Ray dashboard spam

`pyproject.toml` declares dashboard deps (`aiohttp-cors`, `grpcio`, `opencensus`, `opentelemetry-*`, `prometheus-client`) alongside `ray[train]`. Without them, Ray starts the dashboard in minimal mode, the State API is unavailable, and `PlacementGroupCleaner` spams `Failed to query Ray Train Controller actor state` every second.

If all deps are installed, do **not** set `include_dashboard: false` — that also disables the State API and triggers the same spam.
