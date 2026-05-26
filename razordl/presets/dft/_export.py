"""Export DFT preset classes as source code for custom/full modes."""

import importlib.util
import os

from razordl.core.export.ast_utils import extract_class, replace_ident


def _sft_preset_dir(preset_pkg_dir: str) -> str:
    return os.path.join(os.path.dirname(preset_pkg_dir), "sft")


def export_workgroup(preset_pkg_dir: str) -> str:
    sft_dir = _sft_preset_dir(preset_pkg_dir)
    with open(os.path.join(sft_dir, "workgroup.py")) as f:
        sft_src = f.read()
    with open(os.path.join(preset_pkg_dir, "workgroup.py")) as f:
        dft_src = f.read()

    model_cls = replace_ident(extract_class(sft_src, "SFTModelGroup"), "SFTModelGroup", "ModelGroup")
    # Extract SFTWorkGroup (now contains all LM methods) as the base class
    base_wg_cls = replace_ident(extract_class(sft_src, "SFTWorkGroup"), "SFTWorkGroup", "_BaseWorkGroup")
    base_wg_cls = replace_ident(base_wg_cls, "SFTModelGroup", "ModelGroup")
    base_wg_cls = base_wg_cls.replace("class _BaseWorkGroup(WorkGroup):", "class _BaseWorkGroup(_WorkGroup):")
    loss_cls = extract_class(dft_src, "DistDFTLoss")

    header = """import torch

from razordl.core.base import logging
from razordl.core.engine.single_model.workgroup import ModelGroup as _ModelGroup, WorkGroup as _WorkGroup
from razordl.ops.loss.distributed import distributed_token_count
from razordl.ops.model.huggingface import build_causal_lm, build_left_padding_tokenizer

logger = logging.getLogger(__name__)

"""
    wg_cls = """class WorkGroup(_BaseWorkGroup):
    \"\"\"DFT preset: SFT-style model with confidence-weighted CE loss.\"\"\"

    model_group_class = ModelGroup

    def __init__(self, config):
        super().__init__(config)
        mc = config.worker_group_config.model_group_config.model_config
        mini_scale = getattr(mc, "dft_mini_scale", 0.0)
        self.criterion = DistDFTLoss(ignore_index=-100, mini_scale=mini_scale)
        self.chunked_loss = False
"""
    return "\n".join([header, loss_cls + "\n\n", model_cls + "\n\n", base_wg_cls + "\n\n", wg_cls])


def export_dataset(preset_pkg_dir: str) -> str:
    sft_dir = _sft_preset_dir(preset_pkg_dir)
    sft_export_path = os.path.join(sft_dir, "_export.py")
    spec = importlib.util.spec_from_file_location("_export_sft", sft_export_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.export_dataset(sft_dir)
