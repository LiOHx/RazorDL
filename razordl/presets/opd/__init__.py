from razordl.presets.opd.config import OPDConfig
from razordl.presets.opd.workgroup import OPDWorkGroup
from razordl.presets.opd.dataset import OPDDataset, OPDCollator

# Engine declaration — CLI uses this to determine which engine to load.
# "single_model" | "on_policy_single_model"
ENGINE = "on_policy_single_model"

# Dataset declaration — CLI copies this dataset from datasets/ to the project.
DATASET = "gsm8k"

# CLI convention: {CamelCase}Config from "opd" → "Opd"
OpdConfig = OPDConfig
OpdWorkGroup = OPDWorkGroup
OpdDataset = OPDDataset
OpdCollator = OPDCollator

__all__ = [
    "OPDConfig",
    "OPDWorkGroup",
    "OPDDataset",
    "OPDCollator",
    "OpdConfig",
    "OpdWorkGroup",
    "OpdDataset",
    "OpdCollator",
    "ENGINE",
    "DATASET",
]
