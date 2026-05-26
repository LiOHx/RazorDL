import os
import shutil

from razordl.core.base import logging

logger = logging.getLogger(__name__)


def _load_export_module(preset_pkg_dir: str):
    """Load the preset's _export.py via file path (avoids triggering heavy deps)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_export", os.path.join(preset_pkg_dir, "_export.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_preset_engine(preset_pkg_dir: str) -> str:
    """Read the ENGINE variable from a preset's __init__.py via AST.

    Avoids importing the heavy preset module just to know which engine it uses.
    Defaults to ``single_model`` if the preset doesn't declare an engine.
    """
    import ast
    init_path = os.path.join(preset_pkg_dir, "__init__.py")
    if not os.path.exists(init_path):
        return "single_model"
    with open(init_path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ENGINE":
                    if isinstance(node.value, ast.Constant):
                        return node.value.value
    return "single_model"


def _read_preset_dataset(preset_pkg_dir: str) -> str | None:
    """Read the DATASET variable from a preset's __init__.py via AST.

    Returns the dataset name (e.g. "gsm8k") if declared, otherwise None.
    """
    import ast
    init_path = os.path.join(preset_pkg_dir, "__init__.py")
    if not os.path.exists(init_path):
        return None
    with open(init_path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DATASET":
                    if isinstance(node.value, ast.Constant):
                        return node.value.value
    return None


def _generate_full_workgroup(preset_pkg_dir: str) -> str:
    """Generate a complete, self-contained workgroup.py for custom mode."""
    mod = _load_export_module(preset_pkg_dir)
    return mod.export_workgroup(preset_pkg_dir)


def _generate_full_dataset(preset_pkg_dir: str) -> str:
    """Generate a complete, self-contained dataset.py for custom mode."""
    mod = _load_export_module(preset_pkg_dir)
    return mod.export_dataset(preset_pkg_dir)


def _copy_preset_data(preset_pkg_dir: str, project_dir: str, razordl_root: str):
    """Copy dataset into the generated project.

    Priority:
    1. Shared datasets/ dir declared by the preset (DATASET variable).
    2. Preset-local data/ dir (backward compat).
    3. Create an empty data/ dir if neither exists.
    """
    project_data_dir = os.path.join(project_dir, "data")
    os.makedirs(project_data_dir, exist_ok=True)

    # 1. Try shared datasets/ dir (declared by DATASET in __init__.py)
    dataset_name = _read_preset_dataset(preset_pkg_dir)
    if dataset_name:
        shared_data_dir = os.path.join(razordl_root, "datasets", dataset_name)
        if os.path.isdir(shared_data_dir):
            shutil.copytree(shared_data_dir, project_data_dir, dirs_exist_ok=True)
            logger.info(f"Copied dataset '{dataset_name}' to {project_data_dir}")
            return
        else:
            logger.warning(
                f"Preset declares DATASET='{dataset_name}' but "
                f"{shared_data_dir} does not exist. "
                f"Please prepare the dataset or set data_path manually."
            )

    # 2. Fallback: preset-local data/ dir
    preset_data_dir = os.path.join(preset_pkg_dir, "data")
    if os.path.isdir(preset_data_dir):
        shutil.copytree(preset_data_dir, project_data_dir, dirs_exist_ok=True)
        logger.info(f"Copied sample data from preset to {project_data_dir}")


def _generate_run_sh(use_razordl_cli: bool, preset: str = "sft") -> str:
    """Generate run.sh content."""
    if use_razordl_cli:
        return f"""#!/bin/bash
# Launch {preset.upper()} training on GPUs 0-3
# Adjust CUDA_VISIBLE_DEVICES as needed

CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    razordl train --config config.yaml --preset {preset}
"""
    return f"""#!/bin/bash
# Launch {preset.upper()} training on GPUs 0-3
# Adjust CUDA_VISIBLE_DEVICES as needed

CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    python src/main.py
"""


def handle_init(args):
    preset = args.preset
    mode = getattr(args, "mode", "simple")
    project_name = args.project_name
    base_path = os.path.abspath(args.path)
    project_dir = os.path.join(base_path, project_name)

    if os.path.exists(project_dir):
        logger.error(f"Directory already exists: {project_dir}")
        raise SystemExit(1)

    # Locate the preset source package
    preset_pkg_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "presets",
        preset,
    )

    presets_root = os.path.dirname(preset_pkg_dir)
    razordl_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if not os.path.exists(preset_pkg_dir):
        available = sorted(
            d for d in os.listdir(presets_root)
            if os.path.isdir(os.path.join(presets_root, d)) and not d.startswith("_")
        )
        logger.error(f"Unknown preset: {preset}. Available: {', '.join(available)}")
        raise SystemExit(1)

    os.makedirs(project_dir, exist_ok=True)

    # All modes copy default_config.yaml from preset package
    config_src = os.path.join(preset_pkg_dir, "default_config.yaml")
    if os.path.isfile(config_src):
        shutil.copy2(config_src, os.path.join(project_dir, "config.yaml"))

    if mode == "simple":
        # Simple mode: generate run.sh that uses razordl CLI
        run_sh = _generate_run_sh(use_razordl_cli=True, preset=preset)
        run_sh_path = os.path.join(project_dir, "run.sh")
        with open(run_sh_path, "w") as f:
            f.write(run_sh)
        os.chmod(run_sh_path, 0o755)

        # Copy dataset from shared pool
        _copy_preset_data(preset_pkg_dir, project_dir, razordl_root)

    elif mode == "full":
        # Full mode: fully independent project (no razordl dependency)
        run_sh = _generate_run_sh(use_razordl_cli=False, preset=preset)
        run_sh_path = os.path.join(project_dir, "run.sh")
        with open(run_sh_path, "w") as f:
            f.write(run_sh)
        os.chmod(run_sh_path, 0o755)

        # Load _export_full.py directly from file to avoid triggering preset package init
        import importlib.util
        full_export_path = os.path.join(preset_pkg_dir, "_export_full.py")
        spec = importlib.util.spec_from_file_location("_export_full", full_export_path)
        export_full_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export_full_mod)
        created = export_full_mod.export_full_project(preset_pkg_dir, project_dir, razordl_root)

        # Copy dataset from shared pool
        _copy_preset_data(preset_pkg_dir, project_dir, razordl_root)

        logger.info(f"Created project '{project_name}' at {project_dir} (mode=full)")
        logger.info("")
        logger.info("Full mode — completely independent project generated.")
        logger.info("No dependency on the razordl package.")
        logger.info("Install deps:  pip install -r requirements.txt")
        logger.info(f"Run:           cd {project_name} && sh run.sh")
        logger.info("")
        logger.info(f"Generated files ({len(created)} total):")
        for rel in created:
            logger.info(f"  {rel}")
        return
    else:
        # Custom mode: generate standalone run.sh
        run_sh = _generate_run_sh(use_razordl_cli=False, preset=preset)
        run_sh_path = os.path.join(project_dir, "run.sh")
        with open(run_sh_path, "w") as f:
            f.write(run_sh)
        os.chmod(run_sh_path, 0o755)

        # All Python code goes under src/
        src_dir = os.path.join(project_dir, "src")
        os.makedirs(src_dir, exist_ok=True)

        # Generate full, self-contained dataset.py
        dataset_code = _generate_full_dataset(preset_pkg_dir)
        with open(os.path.join(src_dir, "dataset.py"), "w") as f:
            f.write(dataset_code)

        # Generate full, self-contained workgroup.py
        workgroup_code = _generate_full_workgroup(preset_pkg_dir)
        with open(os.path.join(src_dir, "workgroup.py"), "w") as f:
            f.write(workgroup_code)

        # Generate main.py — class/module names follow CamelCase preset convention
        camel = "".join(part.capitalize() for part in preset.split("_"))
        config_class = f"{camel}Config"
        preset_module = f"razordl.presets.{preset}"
        engine = _read_preset_engine(preset_pkg_dir)
        main_code = f'''import os
import yaml

from {preset_module} import {config_class}
from dataset import Dataset, Collator
from workgroup import WorkGroup
from razordl.core.engine.{engine}.main import main

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        flat_config = yaml.safe_load(f)

    config = {config_class}.from_flat_dict(flat_config)
    main(config, WorkGroup, Dataset, Collator)
'''
        with open(os.path.join(src_dir, "main.py"), "w") as f:
            f.write(main_code)

    # Copy preset sample data (or create empty data/)
    _copy_preset_data(preset_pkg_dir, project_dir, razordl_root)

    logger.info(f"Created project '{project_name}' at {project_dir} (mode={mode})")
    logger.info("")
    if mode == "simple":
        logger.info("Simple mode — only config.yaml + run.sh + data/ generated.")
        logger.info("Edit config.yaml, put your data in data/, then run:")
        logger.info(f"  cd {project_name}")
        logger.info("  razordl train")
    else:
        logger.info("Custom mode — full project with complete source code generated.")
        logger.info("All training logic is in your project. Edit any file, then run:")
        logger.info(f"  cd {project_name}")
        logger.info("  sh run.sh")
