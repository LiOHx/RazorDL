import os
import sys

from razordl.core.base import logging

logger = logging.getLogger(__name__)


def _to_camel_case(preset: str) -> str:
    """Convert snake_case preset name to CamelCase class prefix.

    Examples:  sft → SFT, video_embedding → VideoEmbedding, dft → DFT
    """
    return "".join(part.capitalize() for part in preset.split("_"))


def _run_simple_mode(config_dir: str, preset: str):
    """Execute built-in default training logic for simple mode."""
    import importlib
    import yaml

    # Class/module names follow convention: {CamelCase}Config, {CamelCase}WorkGroup, etc.
    preset_mod = importlib.import_module(f"razordl.presets.{preset}")
    camel = _to_camel_case(preset)
    config_class = getattr(preset_mod, f"{camel}Config")
    workgroup_class = getattr(preset_mod, f"{camel}WorkGroup")
    dataset_class = getattr(preset_mod, f"{camel}Dataset")
    collator_class = getattr(preset_mod, f"{camel}Collator")

    # Engine is declared by the preset (defaults to single_model for back-compat)
    engine = getattr(preset_mod, "ENGINE", "single_model")
    engine_main_mod = importlib.import_module(f"razordl.core.engine.{engine}.main")
    main = engine_main_mod.main

    config_path = os.path.join(config_dir, "config.yaml")
    with open(config_path, "r") as f:
        flat_config = yaml.safe_load(f)

    config = config_class.from_flat_dict(flat_config)
    main(config, workgroup_class, dataset_class, collator_class)


def handle_train(args):
    import runpy

    preset = getattr(args, "preset", "sft")
    config_path = os.path.abspath(args.config)

    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise SystemExit(1)

    config_dir = os.path.dirname(config_path) or "."
    main_script = os.path.join(config_dir, "main.py")

    original_cwd = os.getcwd()
    try:
        os.chdir(config_dir)
        sys.path.insert(0, config_dir)

        if os.path.exists(main_script):
            logger.info(f"Launching training from {main_script}")
            runpy.run_path("main.py", run_name="__main__")
        else:
            logger.info(f"Simple mode — using built-in {preset.upper()} training")
            _run_simple_mode(config_dir, preset)
    finally:
        os.chdir(original_cwd)
