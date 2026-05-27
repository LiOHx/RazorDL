import argparse
import os
import sys

from razordl.cli.diff import handle_diff
from razordl.cli.init import handle_init
from razordl.cli.train import handle_train
from razordl.cli.ckpt import handle_ckpt


def _available_presets() -> list[str]:
    """Auto-discover presets by scanning razordl/presets/."""
    presets_dir = os.path.join(os.path.dirname(__file__), "..", "presets")
    if not os.path.isdir(presets_dir):
        return ["sft"]
    return sorted(
        d for d in os.listdir(presets_dir)
        if os.path.isdir(os.path.join(presets_dir, d)) and not d.startswith("_")
    )


def main():
    presets = _available_presets()
    default_preset = "sft" if "sft" in presets else presets[0]

    parser = argparse.ArgumentParser(
        prog="razordl",
        description="RazorDL: A flexible distributed training framework for LLMs",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # razordl init
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new training project from a preset template",
    )
    init_parser.add_argument(
        "project_name",
        help="Name of the project directory to create",
    )
    init_parser.add_argument(
        "--preset",
        default=default_preset,
        choices=presets,
        help=f"Preset template to use (available: {', '.join(presets)}; default: {default_preset})",
    )
    init_parser.add_argument(
        "--path",
        default=".",
        help="Directory where the project will be created (default: current dir)",
    )
    init_parser.add_argument(
        "--mode",
        default="simple",
        choices=["simple", "custom", "full"],
        help="Project scaffolding mode: simple / custom / full (independent project)",
    )

    # razordl train
    train_parser = subparsers.add_parser(
        "train",
        help="Launch training from a config file",
    )
    train_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to training config file (default: config.yaml)",
    )
    train_parser.add_argument(
        "--preset",
        default=default_preset,
        choices=presets,
        help=f"Preset to use for training (available: {', '.join(presets)}; default: {default_preset})",
    )

    # razordl diff
    diff_parser = subparsers.add_parser(
        "diff",
        help="Compare code/config between experiments or with current project",
    )
    diff_parser.add_argument(
        "left",
        nargs="?",
        default=None,
        help="Left side: experiment path (default: current project)",
    )
    diff_parser.add_argument(
        "right",
        nargs="?",
        default=None,
        help="Right side: experiment path (default: latest experiment under outputs/)",
    )

    # razordl ckpt
    ckpt_parser = subparsers.add_parser(
        "ckpt",
        help="Inspect training checkpoints",
    )
    ckpt_subparsers = ckpt_parser.add_subparsers(dest="ckpt_action", help="ckpt sub-actions")
    ckpt_info_parser = ckpt_subparsers.add_parser(
        "info",
        help="Show checkpoint_info.json contents (or legacy marker info)",
    )
    ckpt_info_parser.add_argument(
        "checkpoint_dir",
        help="Path to a checkpoint directory (e.g. output/checkpoint_000030)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "init":
        handle_init(args)
    elif args.command == "train":
        handle_train(args)
    elif args.command == "diff":
        handle_diff(args)
    elif args.command == "ckpt":
        if getattr(args, "ckpt_action", None) is None:
            ckpt_parser.print_help()
            sys.exit(1)
        handle_ckpt(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
