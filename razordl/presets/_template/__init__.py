# [改] 把 NEWConfig/NEWWorkGroup/DistNEWLoss 替换为你的 preset 名 (e.g. RLLoss, RLWorkGroup)
# 如果 Config 和 dataset 完全复用 SFT，保留 from ..sft 导入
from ..sft.config import SFTConfig as NEWConfig  # [改] alias 成你的名，或单独写 config.py
from ..sft.dataset import SFTDataset, SFTCollator  # [不改] 所有 preset 通用 dataset
from .workgroup import DistNEWLoss, NEWWorkGroup    # [改] 你的 loss 类 + WorkGroup

# Engine declaration — CLI uses this to determine which engine to load.
# "single_model" | "on_policy_single_model" | other custom engine
# 如果省略，默认使用 "single_model"
ENGINE = "single_model"

__all__ = [
    "NEWConfig",
    "SFTDataset",
    "SFTCollator",
    "DistNEWLoss",
    "NEWWorkGroup",
    "ENGINE",
]
