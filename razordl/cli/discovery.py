"""Preset discovery shared by the CLI entry points (no heavy imports)."""

import os


def available_presets() -> list[str]:
    """Preset names found by scanning razordl/presets/ (``_*`` excluded)."""
    presets_dir = os.path.join(os.path.dirname(__file__), "..", "presets")
    if not os.path.isdir(presets_dir):
        raise RuntimeError(f"razordl presets directory not found: {presets_dir}")
    return sorted(
        d for d in os.listdir(presets_dir)
        if os.path.isdir(os.path.join(presets_dir, d)) and not d.startswith("_")
    )


def default_preset(presets: list[str] | None = None) -> str:
    presets = presets if presets is not None else available_presets()
    return "sft" if "sft" in presets else presets[0]
