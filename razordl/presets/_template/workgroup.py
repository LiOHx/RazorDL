"""NEW preset workgroup — [改] 把你的 preset 描述写在这是。

[改] 核心差异在这里：定义你的 loss function class + WorkGroup。
WorkGroup 通常继承 SFTWorkGroup，只 override __init__（换 criterion）或 _compute_loss。

参考: presets/sft/workgroup.py（SFT 标准 loss）
      presets/dft/workgroup.py（DFT 置信度加权 loss）
"""

import torch

from razordl.core.base import logging
from razordl.ops.distributed.utils import all_gather_object
from razordl.presets.sft.workgroup import SFTModelGroup, SFTWorkGroup  # [不改] 复用 SFT model loading

logger = logging.getLogger(__name__)


# [改] 定义你的 loss class
class DistNEWLoss(torch.nn.Module):
    """Your custom loss — replace this docstring and implementation."""

    def __init__(self, ignore_index: int = -100, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # [改] 实现你的 loss 逻辑
        # logits/labels 已经 shift 过 (caller does next-token prediction shift)
        return torch.tensor(0.0)  # placeholder


# [改] 定义你的 WorkGroup
class NEWWorkGroup(SFTWorkGroup):
    """Your WorkGroup — override __init__ to set self.criterion and any extra config.

    Inherits update_step, _compute_loss, _simple_loss_compute, _chunked_loss_compute
    from SFTWorkGroup.  Those methods use self.criterion internally, so changing
    self.criterion is usually all you need.
    """

    def __init__(self, config):
        super().__init__(config)
        mc = config.worker_group_config.model_group_config.model_config
        extra_param = getattr(mc, "new_extra_param", 0.0)  # [改] 你的特有配置
        self.criterion = DistNEWLoss(ignore_index=-100, extra_param=extra_param)
        self.chunked_loss = False  # 如果你的 loss 需要完整 logits
