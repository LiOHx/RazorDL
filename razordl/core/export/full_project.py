import ast
import importlib.util
import os
import re


def _find_razordl_imports(source: str) -> set[str]:
    tree = ast.parse(source)
    deps = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("razordl"):
                deps.add(node.module)
                # `from razordl.x import y` may pull in submodule razordl.x.y.
                # Record the candidate; resolver returns None if it's just an attribute.
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    deps.add(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("razordl"):
                    deps.add(alias.name)
    return deps


def _resolve_module_file(module_name: str, razordl_root: str) -> str | None:
    rel = module_name.replace("razordl.", "").replace(".", os.sep)
    pkg_dir = os.path.join(razordl_root, "razordl")
    pkg_init = os.path.join(pkg_dir, rel, "__init__.py")
    if os.path.exists(pkg_init):
        return pkg_init
    mod_file = os.path.join(pkg_dir, rel + ".py")
    if os.path.exists(mod_file):
        return mod_file
    return None


def _collect_dependencies(entry_files: list[str], razordl_root: str) -> set[str]:
    to_process = list(entry_files)
    processed = set()
    all_files = set()
    while to_process:
        file_path = to_process.pop()
        if file_path in processed:
            continue
        processed.add(file_path)
        all_files.add(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception:
            continue
        for module_name in _find_razordl_imports(source):
            dep_file = _resolve_module_file(module_name, razordl_root)
            if dep_file and dep_file not in processed:
                to_process.append(dep_file)
    return all_files


def _find_import_line_ranges(source: str) -> list[tuple[int, int]]:
    tree = ast.parse(source)
    ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            ranges.append((node.lineno, node.end_lineno))
    return ranges


def _rewrite_single_import_line(line: str, engine_name: str, preset_name: str) -> str:
    line = re.sub(
        rf"from\s+razordl\.core\.engine\.{engine_name}\.(\w[\w.]*)\s+import",
        r"from engine.\1 import",
        line,
    )
    line = re.sub(
        rf"from\s+razordl\.core\.engine\.{engine_name}\s+import",
        r"from engine import",
        line,
    )
    line = re.sub(
        r"from\s+razordl\.core\.engine\.common(?:\.(\w[\w.]*))?\s+import",
        lambda m: f"from engine.common.{m.group(1)} import" if m.group(1) else "from engine.common import",
        line,
    )
    line = re.sub(
        r"from\s+razordl\.core\.base(?:\.(\w[\w.]*))?\s+import",
        lambda m: f"from base.{m.group(1)} import" if m.group(1) else "from base import",
        line,
    )
    line = re.sub(
        r"from\s+razordl\.ops\.(\w+(?:\.\w+)*)\s+import",
        r"from ops.\1 import",
        line,
    )
    line = re.sub(
        rf"from\s+razordl\.presets\.{preset_name}\.config\s+import",
        r"from config import",
        line,
    )
    line = re.sub(
        rf"from\s+razordl\.presets\.{preset_name}\.(\w[\w.]*)\s+import",
        r"from \1 import",
        line,
    )
    line = re.sub(r"import\s+razordl\.(\w[\w.]*)", r"import \1", line)
    return line


def rewrite_imports(source: str, engine_name: str, preset_name: str) -> str:
    lines = source.splitlines()
    for start, end in _find_import_line_ranges(source):
        for i in range(start - 1, end):
            lines[i] = _rewrite_single_import_line(lines[i], engine_name, preset_name)
    return "\n".join(lines)


def _neutralize_brand(dst_rel: str, source: str) -> str:
    """Strip upstream-package brand strings from exported source files.

    The only currently-tracked brand carrier is ``base/checkpoint_info.py``
    where ``_FRAMEWORK_PACKAGE_NAME`` is used to look up the framework version
    for provenance.  Set it to ``None`` in generated projects so they don't
    advertise ``razordl`` in ``checkpoint_info.json``.
    """
    if dst_rel.endswith("base/checkpoint_info.py"):
        source = source.replace(
            '_FRAMEWORK_PACKAGE_NAME = "razordl"',
            "_FRAMEWORK_PACKAGE_NAME = None",
        )
    return source



def _map_dest_path(src_rel: str, engine_name: str, preset_name: str) -> str | None:
    if src_rel.startswith("razordl/"):
        src_rel = src_rel[len("razordl/") :]
    if src_rel.startswith(f"core/engine/{engine_name}/"):
        return "src/engine/" + src_rel[len(f"core/engine/{engine_name}/") :]
    if src_rel.startswith("core/engine/common/"):
        return "src/engine/common/" + src_rel[len("core/engine/common/") :]
    if src_rel.startswith("core/base/"):
        return "src/base/" + src_rel[len("core/base/") :]
    if src_rel.startswith("ops/"):
        return "src/" + src_rel
    if src_rel.startswith("presets/") and src_rel.endswith("/config.py"):
        return "src/config.py"
    preset_prefix = f"presets/{preset_name}/"
    if src_rel.startswith(preset_prefix):
        leaf = src_rel[len(preset_prefix) :]
        if leaf in {"workgroup.py", "dataset.py", "__init__.py", "_export.py", "_export_full.py"}:
            return None
        return "src/" + leaf
    if src_rel.startswith("presets/"):
        return None
    return src_rel


def export_full_project_for_engine(
    preset_pkg_dir: str,
    project_dir: str,
    razordl_root: str,
    *,
    engine_name: str,
    engine_files: tuple[str, ...],
) -> list[str]:
    created = []
    preset_name = os.path.basename(preset_pkg_dir)
    config_class = f"{preset_name.upper()}Config"

    entry_files = [
        os.path.join(razordl_root, "razordl", "core", "engine", engine_name, file_name)
        for file_name in engine_files
    ]
    entry_files.extend(
        [
            os.path.join(razordl_root, "razordl", "core", "base", "trainer.py"),
            os.path.join(razordl_root, "razordl", "core", "base", "workgroup.py"),
            os.path.join(razordl_root, "razordl", "core", "base", "config.py"),
            os.path.join(razordl_root, "razordl", "core", "base", "logging.py"),
            os.path.join(razordl_root, "razordl", "core", "base", "dataloader.py"),
            os.path.join(preset_pkg_dir, "config.py"),
            os.path.join(preset_pkg_dir, "workgroup.py"),
            os.path.join(preset_pkg_dir, "dataset.py"),
        ]
    )
    entry_files = [p for p in entry_files if os.path.exists(p)]

    for src_path in sorted(_collect_dependencies(entry_files, razordl_root)):
        src_rel = os.path.relpath(src_path, razordl_root)
        dst_rel = _map_dest_path(src_rel, engine_name, preset_name)
        if not dst_rel:
            continue
        dst_path = os.path.join(project_dir, dst_rel)
        os.makedirs(os.path.dirname(dst_path) if os.path.dirname(dst_rel) else project_dir, exist_ok=True)
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        source = rewrite_imports(source, engine_name, preset_name)
        source = _neutralize_brand(dst_rel, source)
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(source)
        created.append(dst_rel)

    export_path = os.path.join(preset_pkg_dir, "_export.py")
    spec = importlib.util.spec_from_file_location("_export", export_path)
    export_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_mod)

    dataset_code = rewrite_imports(export_mod.export_dataset(preset_pkg_dir), engine_name, preset_name)
    workgroup_code = rewrite_imports(export_mod.export_workgroup(preset_pkg_dir), engine_name, preset_name)

    src_dir = os.path.join(project_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    with open(os.path.join(src_dir, "dataset.py"), "w") as f:
        f.write(dataset_code)
    created.append("src/dataset.py")

    with open(os.path.join(src_dir, "workgroup.py"), "w") as f:
        f.write(workgroup_code)
    created.append("src/workgroup.py")

    main_code = f"""import os
import sys
import yaml

from config import {config_class}
from dataset import Dataset, Collator
from workgroup import WorkGroup
from engine.main import main
from ops.snapshot import diff_experiments, get_latest_experiment, scan_experiments

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "diff":
        left_arg = sys.argv[2] if len(sys.argv) > 2 else None
        right_arg = sys.argv[3] if len(sys.argv) > 3 else None

        if left_arg is None and right_arg is None:
            outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
            latest = get_latest_experiment(outputs_dir)
            if latest is None:
                print("No experiments found under outputs/")
                sys.exit(1)
            left_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            right_dir = latest
            left_label = "当前项目"
            right_label = os.path.basename(latest)
        elif left_arg is not None and right_arg is None:
            left_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            right_dir = os.path.abspath(left_arg)
            left_label = "当前项目"
            right_label = os.path.basename(left_arg.rstrip("/"))
        else:
            left_dir = os.path.abspath(left_arg)
            right_dir = os.path.abspath(right_arg)
            left_label = os.path.basename(left_arg.rstrip("/"))
            right_label = os.path.basename(right_arg.rstrip("/"))

        print(diff_experiments(left_dir, right_dir, left_label, right_label))
    else:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        with open(config_path, "r") as f:
            flat_config = yaml.safe_load(f)

        config = {config_class}.from_flat_dict(flat_config)
        main(config, WorkGroup, Dataset, Collator)
"""
    with open(os.path.join(src_dir, "main.py"), "w") as f:
        f.write(main_code)
    created.append("src/main.py")

    preset_reqs_path = os.path.join(preset_pkg_dir, "requirements.txt")
    if os.path.exists(preset_reqs_path):
        with open(preset_reqs_path, "r", encoding="utf-8") as f:
            reqs = f.read()
    else:
        reqs = "torch>=2.0\ntransformers>=4.36\ntensordict\nray[train]>=2.9\nomegaconf\nsafetensors\npeft\ntqdm\n"
    with open(os.path.join(project_dir, "requirements.txt"), "w") as f:
        f.write(reqs)
    created.append("requirements.txt")

    return created
