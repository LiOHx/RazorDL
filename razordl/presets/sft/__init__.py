from razordl.presets.sft.config import SFTConfig
from razordl.presets.sft.workgroup import SFTModelGroup, SFTWorkGroup
from razordl.presets.sft.dataset import SFTDataset, SFTCollator

DATASET = "demo_chat"

# CLI convention: {CamelCase}Config (e.g. SftConfig from "sft")
SftConfig = SFTConfig
SftModelGroup = SFTModelGroup
SftWorkGroup = SFTWorkGroup
SftDataset = SFTDataset
SftCollator = SFTCollator

__all__ = [
    "DATASET",
    "SFTConfig",
    "SFTModelGroup",
    "SFTWorkGroup",
    "SFTDataset",
    "SFTCollator",
    "SftConfig",
    "SftModelGroup",
    "SftWorkGroup",
    "SftDataset",
    "SftCollator",
]
