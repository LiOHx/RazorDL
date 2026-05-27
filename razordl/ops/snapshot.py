"""Experiment snapshot utilities: code hash, directory snapshot, experiment scanning."""

import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime


EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "env_backups", "outputs", "data", ".eggs", "build", "dist"}
EXCLUDE_SUFFIXES = {".pyc", ".egg-info", ".safetensors", ".pt", ".pth", ".log"}
INCLUDE_SUFFIXES = {".py", ".yaml", ".yml", ".sh", ".json", ".jsonl", ".md"}

# Config keys set by the framework (not user-configurable).  Stripped when
# hashing and diffing YAML files so snapshot-injected values like output_dir
# do not cause false differences.
INTERNAL_CONFIG_KEYS = {"output_dir", "resume_checkpoint_dir"}

TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def _should_include(filepath: str) -> bool:
    """Check if a file should be included in the code snapshot/hash."""
    name = os.path.basename(filepath)
    if name.startswith("."):
        return False
    _, ext = os.path.splitext(name)
    if ext in EXCLUDE_SUFFIXES:
        return False
    if ext in INCLUDE_SUFFIXES:
        return True
    return False


def compute_code_hash(project_dir: str) -> str:
    """Compute a combined SHA256 hash of all relevant project files."""
    h = hashlib.sha256()
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, project_dir)
            if not _should_include(fpath):
                continue
            try:
                with open(fpath, "rb") as f:
                    h.update(rel.encode("utf-8") + b"\x00")
                    h.update(f.read())
            except OSError:
                continue
    return h.hexdigest()


def compute_config_hash(config_path: str) -> str:
    """Compute SHA256 of a single config file."""
    h = hashlib.sha256()
    with open(config_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _git_info(project_dir: str) -> dict:
    """Get git commit and dirty status for the project directory."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", project_dir, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": None, "git_dirty": False}

    try:
        subprocess.check_call(
            ["git", "-C", project_dir, "diff-index", "--quiet", "HEAD", "--"],
            stderr=subprocess.DEVNULL,
        )
        dirty = False
    except subprocess.CalledProcessError:
        dirty = True

    return {"git_commit": commit, "git_dirty": dirty}


def _razordl_git_info() -> dict:
    """Get version identity for the installed razordl package.

    Tries git first (always current during development).  Falls back to
    ``importlib.metadata.version`` when installed without git (e.g. from
    a tarball or sdist).
    """
    razordl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(razordl_dir)
    try:
        commit = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        try:
            subprocess.check_call(
                ["git", "-C", repo_root, "diff-index", "--quiet", "HEAD", "--"],
                stderr=subprocess.DEVNULL,
            )
            dirty = False
        except subprocess.CalledProcessError:
            dirty = True

        return {"razordl_git_commit": commit, "razordl_git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback: installed version (pip install, no git available)
    try:
        from importlib.metadata import version
        return {"razordl_version": version("razordl")}
    except Exception:
        return {}


def _pip_freeze() -> str:
    """Capture pip freeze output for the current environment."""
    try:
        return subprocess.check_output(
            ["pip", "freeze", "--exclude-editable"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def snapshot_code(exp_dir: str, project_dir: str):
    """Copy project code into ``exp_dir/code/`` for reproducibility.

    After copying, the config.yaml in the snapshot has its ``output_dir``
    field hard-coded to the experiment directory absolute path, so the
    snapshot can be copied elsewhere and still point back to the original
    experiment's checkpoints.
    """
    code_dir = os.path.join(exp_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        rel_root = os.path.relpath(root, project_dir)
        target_dir = os.path.join(code_dir, rel_root) if rel_root != "." else code_dir
        os.makedirs(target_dir, exist_ok=True)

        for fname in files:
            fpath = os.path.join(root, fname)
            if not _should_include(fpath):
                continue
            shutil.copy2(fpath, os.path.join(target_dir, fname))

    # Fix config.yaml in snapshot: set output_dir to absolute path
    snapshot_config = os.path.join(code_dir, "config.yaml")
    if os.path.exists(snapshot_config):
        import yaml

        with open(snapshot_config) as f:
            cfg = yaml.safe_load(f) or {}
        cfg["output_dir"] = os.path.abspath(exp_dir)
        with open(snapshot_config, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Write provenance
    config_path = os.path.join(project_dir, "config.yaml")
    provenance = {
        "code_hash": compute_code_hash(project_dir),
        "config_hash": compute_config_hash(config_path) if os.path.exists(config_path) else None,
        "timestamp": datetime.now().isoformat(),
        **_git_info(project_dir),
        **_razordl_git_info(),
    }
    with open(os.path.join(exp_dir, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    # Write pip freeze for full environment reproducibility
    freeze = _pip_freeze()
    if freeze:
        with open(os.path.join(exp_dir, "pip_freeze.txt"), "w") as f:
            f.write(freeze)

    return provenance


def scan_experiments(outputs_dir: str) -> list[str]:
    """List all experiment directories, sorted oldest-first."""
    if not os.path.isdir(outputs_dir):
        return []
    exps = []
    for name in os.listdir(outputs_dir):
        if not TIMESTAMP_RE.match(name):
            continue
        path = os.path.join(outputs_dir, name)
        if os.path.isdir(path):
            exps.append(path)
    exps.sort()
    return exps


def get_latest_experiment(outputs_dir: str) -> str | None:
    """Return the path of the most recent experiment, or None."""
    exps = scan_experiments(outputs_dir)
    return exps[-1] if exps else None


def is_experiment_completed(exp_dir: str) -> bool:
    """Check whether an experiment has finished training.

    The end-of-training ``save_model_and_processor`` writes
    ``checkpoint_info.json`` (kind: model_only) to the experiment root.
    This is atomic — only written after all model files are in place.
    """
    info_path = os.path.join(exp_dir, "checkpoint_info.json")
    if not os.path.exists(info_path):
        return False
    try:
        with open(info_path) as f:
            info = json.load(f)
        return info.get("kind") == "model_only"
    except (json.JSONDecodeError, KeyError):
        return False


def get_provenance(exp_dir: str) -> dict | None:
    """Read ``provenance.json`` from an experiment directory."""
    path = os.path.join(exp_dir, "provenance.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---- diff utilities ---------------------------------------------------------

def _resolve_diff_root(base_path: str) -> str:
    """Given a path, return the code/ dir if it's an experiment, or the path itself."""
    code_dir = os.path.join(base_path, "code")
    if os.path.isdir(code_dir):
        return code_dir
    return base_path


def build_file_tree(root_dir: str) -> dict:
    """Build a ``{relative_path: sha256_hex}`` dict for files under *root_dir*.

    For YAML files, internal framework keys (``INTERNAL_CONFIG_KEYS``) are
    stripped before hashing so that snapshot-injected values like
    ``output_dir`` do not cause false differences.
    """
    tree = {}
    if not os.path.isdir(root_dir):
        return tree
    for dirpath, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for fname in sorted(files):
            if fname.startswith("."):
                continue
            _, ext = os.path.splitext(fname)
            if ext not in INCLUDE_SUFFIXES:
                continue
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, root_dir)
            h = hashlib.sha256()
            try:
                if ext in (".yaml", ".yml"):
                    # Strip internal keys before hashing
                    import yaml as _yaml
                    with open(fpath, "rb") as f:
                        raw = f.read()
                    try:
                        data = _yaml.safe_load(raw) or {}
                        for k in INTERNAL_CONFIG_KEYS:
                            data.pop(k, None)
                        normalized = _yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=True)
                        h.update(normalized.encode("utf-8"))
                    except Exception:
                        h.update(raw)
                else:
                    with open(fpath, "rb") as f:
                        h.update(f.read())
            except OSError:
                continue
            tree[rel] = h.hexdigest()
    return tree


def diff_yaml_values(left_path: str, right_path: str) -> dict:
    """Return ``{key: (left_val, right_val)}`` for keys that differ between two YAML files."""
    import yaml

    diffs = {}
    try:
        with open(left_path) as f:
            left = yaml.safe_load(f) or {}
    except Exception:
        left = {}
    try:
        with open(right_path) as f:
            right = yaml.safe_load(f) or {}
    except Exception:
        right = {}
    all_keys = set(left.keys()) | set(right.keys())
    for key in sorted(all_keys):
        if key in INTERNAL_CONFIG_KEYS:
            continue
        lv = left.get(key)
        rv = right.get(key)
        if lv != rv:
            diffs[key] = (lv, rv)
    return diffs


def diff_experiments(left_dir: str, right_dir: str, left_label: str, right_label: str) -> str:
    """Compare two experiment code snapshots and return a formatted report string.

    Each *dir* may be a project directory or an experiment directory (with a
    ``code/`` subdirectory).  Labels are used in the report header.
    """
    left_dir = _resolve_diff_root(left_dir)
    right_dir = _resolve_diff_root(right_dir)

    if not os.path.isdir(left_dir):
        return f"Error: not a directory: {left_dir}"
    if not os.path.isdir(right_dir):
        return f"Error: not a directory: {right_dir}"

    left_tree = build_file_tree(left_dir)
    right_tree = build_file_tree(right_dir)
    left_files = set(left_tree.keys())
    right_files = set(right_tree.keys())

    added = sorted(right_files - left_files)
    deleted = sorted(left_files - right_files)
    common = sorted(left_files & right_files)
    modified = [f for f in common if left_tree[f] != right_tree[f]]
    unchanged = [f for f in common if left_tree[f] == right_tree[f]]

    lines = [f"{left_label}  ←→  {right_label}", ""]

    if added or deleted or modified:
        lines.append("文件变更:")
        for f in added:
            lines.append(f"  A  {f}")
        for f in deleted:
            lines.append(f"  D  {f}")
        for f in modified:
            lines.append(f"  M  {f}")
        if unchanged:
            lines.append(f"  ({len(unchanged)} files unchanged)")
        lines.append("")
    else:
        lines.append("文件: 无变更")
        lines.append("")

    # Config YAML diff
    for yaml_name in ("config.yaml", "default_config.yaml"):
        left_yaml = os.path.join(left_dir, yaml_name)
        right_yaml = os.path.join(right_dir, yaml_name)
        if os.path.exists(left_yaml) and os.path.exists(right_yaml):
            yaml_diffs = diff_yaml_values(left_yaml, right_yaml)
            if yaml_diffs:
                lines.append(f"配置变更 ({yaml_name}):")
                max_key_len = max(len(k) for k in yaml_diffs)
                for key, (lv, rv) in yaml_diffs.items():
                    lines.append(f"  {key:<{max_key_len}}  {lv}  →  {rv}")
                lines.append("")

    # Provenance
    left_prov = get_provenance(os.path.dirname(left_dir)) if left_dir.endswith("code") else None
    right_prov = get_provenance(os.path.dirname(right_dir)) if right_dir.endswith("code") else None

    if left_prov or right_prov:
        lines.append("版本:")
        for label, prov in [(left_label, left_prov), (right_label, right_prov)]:
            if not prov:
                lines.append(f"  {label}:  unknown")
                continue
            parts = []
            if prov.get("git_commit"):
                parts.append(f"project={prov['git_commit']}{' (dirty)' if prov.get('git_dirty') else ''}")
            raz_id = prov.get("razordl_git_commit") or prov.get("razordl_version")
            if raz_id:
                parts.append(f"razordl={raz_id}{' (dirty)' if prov.get('razordl_git_dirty') else ''}")
            lines.append(f"  {label}:  {', '.join(parts) if parts else 'unknown'}")

    return "\n".join(lines)
