"""Template export module for new presets.

Copy this file into a preset and replace the markers for your loss/model/data
classes.  AST helpers live in ``razordl.core.export.ast_utils``.
"""

import importlib.util
import os

from razordl.core.export.ast_utils import extract_class, replace_ident


def _sft_preset_dir(preset_pkg_dir: str) -> str:
    return os.path.join(os.path.dirname(preset_pkg_dir), "sft")


def export_workgroup(preset_pkg_dir: str) -> str:
    """[改] Generate a complete workgroup.py using the LMWorkGroup skeleton."""
    sft_dir = _sft_preset_dir(preset_pkg_dir)
    with open(os.path.join(sft_dir, "workgroup.py")) as f:
        sft_src = f.read()
    with open(os.path.join(preset_pkg_dir, "workgroup.py")) as f:
        new_src = f.read()

    model_cls = replace_ident(extract_class(sft_src, "SFTModelGroup"), "SFTModelGroup", "ModelGroup")
    loss_cls = extract_class(new_src, "DistNEWLoss")  # [改] your loss class

    header = """from razordl.core.base import logging
from razordl.core.engine.single_model.lm_workgroup import LMWorkGroup
from razordl.core.engine.single_model.workgroup import ModelGroup as _ModelGroup
from razordl.ops.model.huggingface import build_causal_lm, build_left_padding_tokenizer

logger = logging.getLogger(__name__)

"""
    wg_cls = """class WorkGroup(LMWorkGroup):
    model_group_class = ModelGroup

    def __init__(self, config):
        super().__init__(config)
        self.criterion = DistNEWLoss(ignore_index=-100, new_param=0.0)  # [改]
        self.chunked_loss = False
"""
    return "\n".join([header, loss_cls + "\n\n", model_cls + "\n\n", wg_cls])


def export_dataset(preset_pkg_dir: str) -> str:
    """[不改] Reuse SFT dataset export by default."""
    sft_dir = _sft_preset_dir(preset_pkg_dir)
    sft_export_path = os.path.join(sft_dir, "_export.py")
    spec = importlib.util.spec_from_file_location("_export_sft", sft_export_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.export_dataset(sft_dir)
