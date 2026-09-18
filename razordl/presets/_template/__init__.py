# [改] 把 NEWConfig / NEWWorkGroup / DistNEWLoss 换成你的名字 (e.g. RLConfig, RLWorkGroup)
# 如果 dataset 完全复用 SFT，保留 SFTDataset / SFTCollator 的导入
from razordl.presets.new.config import NEWConfig  # [改] razordl.presets.<目录名>.config
from razordl.presets.sft.dataset import SFTDataset, SFTCollator  # [不改] 所有 LM preset 通用 dataset
from razordl.presets.new.workgroup import DistNEWLoss, NEWWorkGroup  # [改] razordl.presets.<目录名>.workgroup

# Shared sample dataset copied into new projects' data/ (razordl/datasets/<name>).
DATASET = "demo_chat"

# Engine declaration — CLI uses this to determine which engine to load.
# "single_model" | "on_policy_single_model" | other custom engine
ENGINE = "single_model"

# CLI convention: the CLI resolves {CamelCase}Config / WorkGroup / Dataset /
# Collator, where CamelCase = "".join(p.capitalize() for p in <目录名>.split("_")).
# [改] 目录名 "new" -> "New"; e.g. "my_rl" -> MyRlConfig.
NewConfig = NEWConfig
NewDataset = SFTDataset
NewCollator = SFTCollator
NewWorkGroup = NEWWorkGroup

__all__ = [
    "NEWConfig",
    "SFTDataset",
    "SFTCollator",
    "DistNEWLoss",
    "NEWWorkGroup",
    "NewConfig",
    "NewDataset",
    "NewCollator",
    "NewWorkGroup",
    "DATASET",
    "ENGINE",
]
