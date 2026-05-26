"""Export video_embedding preset classes as self-contained source code.

Uses AST to read source files (never import), so preset code changes are
automatically reflected in generated output.
"""

import os

from razordl.core.export.ast_utils import extract_class, extract_function, replace_ident


def export_workgroup(preset_pkg_dir: str) -> str:
    """Generate a complete workgroup.py for custom mode."""
    with open(os.path.join(preset_pkg_dir, "workgroup.py")) as f:
        wg_src = f.read()

    model_cls = extract_class(wg_src, "VideoEmbeddingModelGroup")
    wg_cls = extract_class(wg_src, "VideoEmbeddingWorkGroup")
    loss_cls = extract_class(wg_src, "UnifiedContrastiveLoss")
    pooling_fn = extract_function(wg_src, "_pooling")

    header = """import os

import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict

from razordl.core.base import logging
from razordl.core.engine.single_model.workgroup import (
    ModelGroup as _ModelGroup,
    WorkGroup as _WorkGroup,
)
from razordl.ops.model.huggingface import enforce_model_profile
from razordl.ops.multimodal import split_multi_modal_input_dict

logger = logging.getLogger(__name__)

"""

    model_cls = replace_ident(model_cls, "VideoEmbeddingModelGroup", "ModelGroup")
    wg_cls = replace_ident(wg_cls, "VideoEmbeddingWorkGroup", "WorkGroup")
    wg_cls = replace_ident(wg_cls, "VideoEmbeddingModelGroup", "ModelGroup")
    wg_cls = replace_ident(wg_cls, "_get_embeddings", "get_embeddings")
    wg_cls = replace_ident(wg_cls, "_pre_process_batch", "pre_process_batch")
    wg_cls = replace_ident(wg_cls, "_compute_loss", "compute_loss")
    wg_cls = replace_ident(wg_cls, "_pooling", "pooling")
    wg_cls = replace_ident(wg_cls, "self.criterion", "self.unified_contrastive_criterion")

    pieces = [header, loss_cls + "\n\n", pooling_fn + "\n\n", model_cls + "\n\n", wg_cls]
    return "\n".join(pieces)


def export_dataset(preset_pkg_dir: str) -> str:
    """Generate a complete dataset.py for custom mode."""
    with open(os.path.join(preset_pkg_dir, "dataset.py")) as f:
        src = f.read()

    dataset_cls = extract_class(src, "VideoEmbeddingDataset")
    collator_cls = extract_class(src, "VideoEmbeddingCollator")

    header_lines = []
    for line in src.splitlines():
        if line.startswith("import ") or line.startswith("from "):
            if "from torch.utils.data import Dataset" in line:
                line = line.replace(
                    "from torch.utils.data import Dataset",
                    "from torch.utils.data import Dataset as _Dataset",
                )
            header_lines.append(line)

    header = "\n".join(l for l in header_lines if l.strip())
    if header:
        header += "\n\nlogger = logging.getLogger(__name__)\n"

    dataset_cls = replace_ident(dataset_cls, "VideoEmbeddingDataset", "Dataset")
    collator_cls = replace_ident(collator_cls, "VideoEmbeddingCollator", "Collator")
    dataset_cls = dataset_cls.replace("class Dataset(Dataset):", "class Dataset(_Dataset):")

    return "\n\n".join([header, dataset_cls, collator_cls])
