#!/usr/bin/env python3
"""Sync preset requirements.txt files from pyproject.toml.

Core dependencies ([project.dependencies]) are shared by all presets.
Preset-specific extras ([project.optional-dependencies].{preset}) are merged
per-preset.  New presets only need a new extra in pyproject.toml.

Run manually:  python scripts/sync_requirements.py
Or let the pre-commit hook run it automatically when pyproject.toml changes.
"""

import os
import sys

PYPROJECT = "pyproject.toml"
PRESETS_DIR = "razordl/presets"


def _parse_dep(dep: str) -> tuple[str, str]:
    """Parse 'torch>=2.4.0' -> ('torch', '>=2.4.0')."""
    dep = dep.strip()
    # Find first operator char
    for i, ch in enumerate(dep):
        if ch in "<>=!~":
            return dep[:i], dep[i:]
    return dep, ""


def _extract_deps(pyproject_path: str) -> tuple[list[str], dict[str, list[str]]]:
    """Parse [project.dependencies] and [project.optional-dependencies] from pyproject.toml."""
    import tomllib

    with open(pyproject_path, "rb") as f:
        cfg = tomllib.load(f)

    core = cfg.get("project", {}).get("dependencies", [])
    extras = cfg.get("project", {}).get("optional-dependencies", {})
    return core, extras


def _merge_deps(core: list[str], extra: list[str]) -> list[str]:
    """Merge core + extra deps.  Extra overrides core on version conflict."""
    merged: dict[str, str] = {}
    for dep in core + extra:
        dep = dep.strip()
        if not dep or dep.startswith("#"):
            continue
        pkg, spec = _parse_dep(dep)
        merged[pkg.lower()] = dep  # extra comes after core, so it wins
    return list(merged.values())


def _write_preset_reqs(preset_dir: str, deps: list[str]) -> bool:
    """Write requirements.txt for a preset. Returns True if file changed."""
    req_path = os.path.join(preset_dir, "requirements.txt")

    lines = [
        "# Auto-generated from pyproject.toml",
        "# Do not edit manually. Run: python scripts/sync_requirements.py",
        "",
    ]
    for dep in deps:
        lines.append(dep)
    content = "\n".join(lines) + "\n"

    # Check if content changed
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            if f.read() == content:
                return False

    with open(req_path, "w", encoding="utf-8") as f:
        f.write(content)
    # Auto-stage so pre-commit doesn't block on the modification
    os.system(f"git add {req_path}")
    return True


def main() -> int:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    if not os.path.exists(PYPROJECT):
        print(f"Error: {PYPROJECT} not found in {project_root}")
        return 1

    core_deps, extras = _extract_deps(PYPROJECT)

    if not os.path.isdir(PRESETS_DIR):
        print(f"Warning: {PRESETS_DIR} not found")
        return 0

    changed = False
    for preset in sorted(os.listdir(PRESETS_DIR)):
        preset_path = os.path.join(PRESETS_DIR, preset)
        if not os.path.isdir(preset_path):
            continue
        # Only sync if the preset already has a requirements.txt
        if not os.path.exists(os.path.join(preset_path, "requirements.txt")):
            continue

        preset_extra = extras.get(preset, [])
        merged = _merge_deps(core_deps, preset_extra)
        if _write_preset_reqs(preset_path, merged):
            print(f"Updated {preset_path}/requirements.txt")
            changed = True

    if changed:
        print("Requirements synced. Re-run git commit to complete.")
    else:
        print("All requirements.txt files are already up to date.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
