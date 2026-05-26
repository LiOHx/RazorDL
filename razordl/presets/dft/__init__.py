from razordl.presets.sft.config import SFTConfig as DFTConfig
from razordl.presets.sft.dataset import SFTDataset, SFTCollator
from razordl.presets.dft.workgroup import DistDFTLoss, DFTWorkGroup

DATASET = "demo_chat"

# CLI convention: {CamelCase}Config (e.g. DftConfig from "dft")
DftConfig = DFTConfig
DftDataset = SFTDataset
DftCollator = SFTCollator
DftWorkGroup = DFTWorkGroup

__all__ = [
    "DFTConfig",
    "SFTDataset",
    "SFTCollator",
    "DistDFTLoss",
    "DFTWorkGroup",
    "DftConfig",
    "DftDataset",
    "DftCollator",
    "DftWorkGroup",
]
