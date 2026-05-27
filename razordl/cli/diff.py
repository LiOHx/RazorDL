"""``razordl diff`` — compare experiment code/config snapshots."""

import os
import sys

from razordl.ops.snapshot import diff_experiments, get_latest_experiment


def handle_diff(args):
    left_arg = getattr(args, "left", None)
    right_arg = getattr(args, "right", None)

    if left_arg is None and right_arg is None:
        # Current project vs latest experiment
        outputs_dir = os.path.join(os.getcwd(), "outputs")
        latest = get_latest_experiment(outputs_dir)
        if latest is None:
            print("No experiments found under outputs/")
            sys.exit(1)
        left_dir = os.path.abspath(os.getcwd())
        right_dir = latest
        left_label = "当前项目"
        right_label = os.path.basename(latest)
    elif left_arg is not None and right_arg is None:
        # Current project vs specified path
        left_dir = os.path.abspath(os.getcwd())
        right_dir = os.path.abspath(left_arg)
        left_label = "当前项目"
        right_label = os.path.basename(left_arg.rstrip("/"))
    else:
        # Two paths
        left_dir = os.path.abspath(left_arg)
        right_dir = os.path.abspath(right_arg)
        left_label = os.path.basename(left_arg.rstrip("/"))
        right_label = os.path.basename(right_arg.rstrip("/"))

    print(diff_experiments(left_dir, right_dir, left_label, right_label))
