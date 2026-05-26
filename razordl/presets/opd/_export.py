"""Export OPD preset classes as source code for custom/full modes."""

import os

from razordl.core.export.ast_utils import extract_class, extract_function, replace_ident


def export_workgroup(preset_pkg_dir: str) -> str:
    with open(os.path.join(preset_pkg_dir, "workgroup.py")) as f:
        wg_src = f.read()

    base_cls = extract_class(wg_src, "OPDCausalLMModelGroup")
    policy_cls = extract_class(wg_src, "OPDPolicyModelGroup")
    teacher_cls = extract_class(wg_src, "OPDTeacherModelGroup")
    wg_cls = extract_class(wg_src, "OPDWorkGroup")

    header = """import contextlib
import copy
import os

import torch

from razordl.core.base import logging
from razordl.core.base.metrics import DistStats
from razordl.core.engine.on_policy_single_model.config import Config
from razordl.core.engine.on_policy_single_model.modelgroup import ModelGroup as _ModelGroup
from razordl.core.engine.on_policy_single_model.workgroup import WorkGroup as _WorkGroup
from razordl.ops.distributed.utils import all_gather_object
from razordl.ops.model.huggingface import build_causal_lm, build_left_padding_tokenizer
from razordl.ops.model.per_token_logp import compute_per_token_log_probs

logger = logging.getLogger(__name__)

"""

    base_cls = replace_ident(base_cls, "OPDCausalLMModelGroup", "CausalLMModelGroup")
    policy_cls = replace_ident(policy_cls, "OPDPolicyModelGroup", "PolicyModelGroup")
    policy_cls = replace_ident(policy_cls, "OPDCausalLMModelGroup", "CausalLMModelGroup")
    teacher_cls = replace_ident(teacher_cls, "OPDTeacherModelGroup", "TeacherModelGroup")
    teacher_cls = replace_ident(teacher_cls, "OPDCausalLMModelGroup", "CausalLMModelGroup")
    wg_cls = replace_ident(wg_cls, "OPDWorkGroup", "WorkGroup")
    wg_cls = replace_ident(wg_cls, "OPDPolicyModelGroup", "PolicyModelGroup")
    wg_cls = replace_ident(wg_cls, "OPDTeacherModelGroup", "TeacherModelGroup")

    helper_names = [
        "opd_vllm_max_lora_rank",
        "_sync_lora_weights",
        "_sync_full_weights",
        "asdict_peft",
        "kl_penalty",
        "_remove_left_padding_batch",
    ]
    helpers = "\n\n".join(extract_function(wg_src, name) for name in helper_names)

    return "\n".join([
        header,
        base_cls + "\n\n",
        policy_cls + "\n\n",
        teacher_cls + "\n\n",
        wg_cls + "\n\n",
        helpers,
    ])


def export_dataset(preset_pkg_dir: str) -> str:
    with open(os.path.join(preset_pkg_dir, "dataset.py")) as f:
        src = f.read()

    dataset_cls = extract_class(src, "OPDDataset")
    collator_cls = extract_class(src, "OPDCollator")

    header_lines = []
    for line in src.splitlines():
        if line.startswith("import ") or line.startswith("from "):
            if "from torch.utils.data import Dataset" in line:
                line = line.replace(
                    "from torch.utils.data import Dataset",
                    "from torch.utils.data import Dataset as _Dataset",
                )
            header_lines.append(line)

    # Capture top-level module constants (paren-wrapped / triple-quoted /
    # single-line string assignments).  Same scanner as the GRPO export.
    constants = []
    in_const = False
    for line in src.splitlines():
        if line.strip().endswith(" = (") or " = \"\"\"" in line or (" = '" in line and not line.startswith(" ")):
            in_const = True
        if in_const:
            constants.append(line)
            if line.strip() == ")" or line.strip().endswith("\"\"\"") or (
                line.strip().endswith("'") and not line.strip().startswith("from ")
            ):
                in_const = False
                constants.append("")

    header = "\n".join(l for l in header_lines if l.strip())
    if header:
        header += "\n\nlogger = logging.getLogger(__name__)\n"
    if constants:
        header += "\n".join(constants) + "\n"

    dataset_cls = replace_ident(dataset_cls, "OPDDataset", "Dataset")
    collator_cls = replace_ident(collator_cls, "OPDCollator", "Collator")
    dataset_cls = dataset_cls.replace("class Dataset(Dataset):", "class Dataset(_Dataset):")

    return "\n\n".join([header, dataset_cls, collator_cls])
