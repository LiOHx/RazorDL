"""Structured metadata for training checkpoints.

Each checkpoint directory contains a ``checkpoint_info.json`` describing its
completed step, topology, provenance, metrics and the model/optimizer files it
holds.  The file replaces the older single-integer ``complete`` marker while
still recognizing legacy checkpoints for backwards compatibility.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from typing import Any

from razordl.core.base import logging

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
INFO_FILENAME = "checkpoint_info.json"
LEGACY_MARKER = "complete"

_MODEL_FILE_SUFFIXES = (".safetensors",)
_EXTRA_FILES = ("optimizer.pt", "scheduler.pt")

# Brand identity for provenance lookup.  Full-mode export rewrites this to
# ``None`` so generated projects don't carry the upstream package name.
_FRAMEWORK_PACKAGE_NAME = "razordl"


def get_framework_version() -> str:
    if _FRAMEWORK_PACKAGE_NAME is None:
        return "unknown"
    try:
        from importlib.metadata import version
        return version(_FRAMEWORK_PACKAGE_NAME)
    except Exception:
        try:
            import importlib
            mod = importlib.import_module(_FRAMEWORK_PACKAGE_NAME)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return "unknown"


def get_git_commit() -> str | None:
    """Return the current project's git commit SHA, or ``None`` on any failure."""
    try:
        framework_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        repo_root = os.path.dirname(framework_root)
        result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except Exception:
        return None


def compute_config_hash(config) -> str:
    try:
        payload = json.dumps(config.to_dict(), sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        return "sha256:unknown"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_files(ckpt_dir: str, compute_checksums: bool = False) -> dict[str, dict[str, Any]]:
    """Walk *ckpt_dir* and return ``{relpath: {size, sha256?}}`` for model/optimizer files.

    Skips nested ``checkpoint_*`` directories so that an outer save (e.g. final
    ``save_model_and_processor(output_dir)``) doesn't pick up files belonging
    to per-step checkpoints saved earlier.
    """
    result: dict[str, dict[str, Any]] = {}
    for root, dirs, files in os.walk(ckpt_dir):
        dirs[:] = [d for d in dirs if not (d.startswith("checkpoint_") or d.endswith(".tmp"))]
        for filename in files:
            if filename == INFO_FILENAME or filename == INFO_FILENAME + ".tmp":
                continue
            if not (filename.endswith(_MODEL_FILE_SUFFIXES) or filename in _EXTRA_FILES):
                continue
            full_path = os.path.join(root, filename)
            relpath = os.path.relpath(full_path, ckpt_dir)
            try:
                size = os.path.getsize(full_path)
            except OSError:
                continue
            entry: dict[str, Any] = {"size": size}
            if compute_checksums:
                try:
                    entry["sha256"] = _sha256_file(full_path)
                except OSError:
                    pass
            result[relpath] = entry
    return result


def _get_topology(config) -> dict[str, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    sp_size = 1
    data_cfg = getattr(config, "data_config", None)
    if data_cfg is not None:
        sp_size = getattr(data_cfg, "sp_size", 1) or 1
    if sp_size == 1:
        wg_cfg = getattr(config, "worker_group_config", None)
        if wg_cfg is not None:
            mg_cfg = getattr(wg_cfg, "model_group_config", None)
            if mg_cfg is not None:
                model_cfg = getattr(mg_cfg, "model_config", None)
                if model_cfg is not None:
                    sp_size = getattr(model_cfg, "sp_size", 1) or 1
    return {"world_size": world_size, "sp_size": sp_size}


def build_info(
    *,
    completed_step: int,
    config,
    elapsed_seconds: float,
    last_step_info: dict | None,
    resumed_from: str | None,
    ckpt_dir: str,
    kind: str,
    compute_checksums: bool = False,
) -> dict[str, Any]:
    """Assemble the ``checkpoint_info.json`` payload."""
    metrics: dict[str, Any] = {}
    if last_step_info:
        for k, v in last_step_info.items():
            if k == "step":
                continue
            metrics[k] = v

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "completed_step": int(completed_step),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(float(elapsed_seconds), 2),
        "topology": _get_topology(config),
        "seed": int(getattr(getattr(config, "trainer_config", None), "seed", 0) or 0),
        "provenance": {
            "framework_version": get_framework_version(),
            "git_commit": get_git_commit(),
            "config_hash": compute_config_hash(config),
        },
        "metrics": metrics,
        "files": scan_files(ckpt_dir, compute_checksums=compute_checksums),
        "resumed_from": resumed_from,
    }


def write_info(ckpt_dir: str, info: dict[str, Any]) -> None:
    """Write *info* to ``<ckpt_dir>/checkpoint_info.json`` atomically (tmp+rename)."""
    final_path = os.path.join(ckpt_dir, INFO_FILENAME)
    tmp_path = final_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, final_path)


def read_info(ckpt_dir: str) -> dict[str, Any] | None:
    path = os.path.join(ckpt_dir, INFO_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[checkpoint_info] Failed to read {path}: {e}")
        return None


def _has_required_content(ckpt_dir: str) -> bool:
    has_model = False
    has_optimizer = False
    for _root, dirs, files in os.walk(ckpt_dir):
        dirs[:] = [d for d in dirs if not (d.startswith("checkpoint_") or d.endswith(".tmp"))]
        if "model.safetensors" in files or "adapter_model.safetensors" in files:
            has_model = True
        if "optimizer.pt" in files:
            has_optimizer = True
    return has_model and has_optimizer


def is_complete(ckpt_dir: str, require_marker: bool = True) -> bool:
    """Return True iff *ckpt_dir* contains a finished checkpoint.

    Accepts either the new ``checkpoint_info.json`` or the legacy ``complete``
    marker.  Content check (``has_model AND has_optimizer``) always runs.
    """
    if require_marker:
        new_marker = os.path.isfile(os.path.join(ckpt_dir, INFO_FILENAME))
        old_marker = os.path.isfile(os.path.join(ckpt_dir, LEGACY_MARKER))
        if not (new_marker or old_marker):
            return False
    return _has_required_content(ckpt_dir)
