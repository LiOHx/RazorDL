"""Export SFT preset classes as source code for custom/full modes."""

import os

from razordl.core.export.ast_utils import extract_class, replace_ident


def export_workgroup(preset_pkg_dir: str) -> str:
    with open(os.path.join(preset_pkg_dir, "workgroup.py")) as f:
        wg_src = f.read()

    model_cls = extract_class(wg_src, "SFTModelGroup")
    wg_cls = extract_class(wg_src, "SFTWorkGroup")

    header = """from razordl.core.base import logging
from razordl.core.engine.single_model.workgroup import ModelGroup as _ModelGroup, WorkGroup as _WorkGroup
from razordl.ops.model.huggingface import build_causal_lm, build_left_padding_tokenizer

logger = logging.getLogger(__name__)

"""

    model_cls = replace_ident(model_cls, "SFTModelGroup", "ModelGroup")
    wg_cls = replace_ident(wg_cls, "SFTWorkGroup", "WorkGroup")
    wg_cls = replace_ident(wg_cls, "SFTModelGroup", "ModelGroup")
    # Fix self-referential parent: "class WorkGroup(WorkGroup):" → "class WorkGroup(_WorkGroup):"
    wg_cls = wg_cls.replace("class WorkGroup(WorkGroup):", "class WorkGroup(_WorkGroup):")

    return "\n".join([header, model_cls + "\n\n", wg_cls])


def export_dataset(preset_pkg_dir: str) -> str:
    with open(os.path.join(preset_pkg_dir, "dataset.py")) as f:
        src = f.read()

    dataset_cls = extract_class(src, "SFTDataset")
    collator_cls = extract_class(src, "SFTCollator")

    header_lines = []
    for line in src.splitlines():
        if line.startswith("import ") or line.startswith("from "):
            if "abstractmethod" in line:
                continue
            if "from torch.utils.data import Dataset" in line:
                line = line.replace(
                    "from torch.utils.data import Dataset",
                    "from torch.utils.data import Dataset as _Dataset",
                )
            header_lines.append(line)

    header = "\n".join(l for l in header_lines if l.strip())
    if header:
        header += "\n\nlogger = logging.getLogger(__name__)\n"

    dataset_cls = replace_ident(dataset_cls, "SFTDataset", "Dataset")
    collator_cls = replace_ident(collator_cls, "SFTCollator", "Collator")
    dataset_cls = dataset_cls.replace("class Dataset(Dataset):", "class Dataset(_Dataset):")

    return "\n\n".join([header, dataset_cls, collator_cls])
