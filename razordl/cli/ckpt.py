"""CLI subcommand: ``razordl ckpt`` — inspect training checkpoints.

Currently exposes ``info <dir>``.  Designed to grow ``list`` / ``verify`` /
``best`` sub-actions without restructuring.
"""
from __future__ import annotations

import json
import os
import sys


def _info(checkpoint_dir: str) -> int:
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: not a directory: {checkpoint_dir}", file=sys.stderr)
        return 2

    from razordl.core.base import checkpoint_info as ckpt_info

    info = ckpt_info.read_info(checkpoint_dir)
    if info is not None:
        print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
        return 0

    legacy_marker = os.path.join(checkpoint_dir, ckpt_info.LEGACY_MARKER)
    if os.path.isfile(legacy_marker):
        try:
            with open(legacy_marker, "r", encoding="utf-8") as f:
                step = f.read().strip()
        except OSError:
            step = "unknown"
        print(json.dumps({
            "schema_version": 0,
            "kind": "legacy",
            "completed_step": step,
            "note": "Legacy checkpoint (pre-checkpoint_info.json). Only step number is available.",
        }, indent=2, ensure_ascii=False))
        return 0

    print(
        f"No checkpoint metadata found at {checkpoint_dir}. "
        f"Looked for {ckpt_info.INFO_FILENAME} and {ckpt_info.LEGACY_MARKER}.",
        file=sys.stderr,
    )
    return 1


def handle_ckpt(args) -> None:
    action = getattr(args, "ckpt_action", None)
    if action == "info":
        sys.exit(_info(args.checkpoint_dir))
    print(f"Unknown ckpt action: {action!r}", file=sys.stderr)
    sys.exit(2)
