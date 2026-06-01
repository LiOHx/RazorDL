"""Export DFT preset classes as source code for custom/full modes."""

import importlib.util
import os
import re

from razordl.core.export.ast_utils import extract_class, extract_imports, replace_ident


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

    # Merge imports from both sources, deduplicating.  Filter out the
    # cross-preset import (from razordl.presets.sft.workgroup) since the
    # classes it imports are inlined above.
    sft_imports = extract_imports(sft_src).splitlines()
    dft_imports = [
        l for l in extract_imports(dft_src).splitlines()
        if "razordl.presets.sft" not in l
    ]
    seen = set()
    all_imports = []
    for line in sft_imports + dft_imports:
        if line not in seen:
            seen.add(line)
            all_imports.append(line)
    imports = "\n".join(all_imports)
    imports = re.sub(
        r",\s*WorkGroup\s*$",
        ", WorkGroup as _WorkGroup",
        imports,
        flags=re.MULTILINE,
    )
    header = imports + "\n\nlogger = logging.getLogger(__name__)\n\n"
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
