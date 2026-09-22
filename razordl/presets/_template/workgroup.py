"""NEW preset workgroup — [改] 把你的 preset 描述写在这里。

[改] 核心差异在这里：定义你的 loss function class + WorkGroup。
WorkGroup 通常继承 SFTWorkGroup，只 override __init__（换 criterion）或 _compute_loss。

参考: presets/sft/workgroup.py（SFT 标准 loss）
      presets/dft/workgroup.py（DFT 置信度加权 loss）
"""

import torch
import torch.nn.functional as F

from razordl.core.base import logging
from razordl.ops.loss.distributed import global_token_denominator
from razordl.presets.sft.workgroup import SFTModelGroup, SFTWorkGroup  # [不改] 复用 SFT model loading

logger = logging.getLogger(__name__)


# [改] 定义你的 loss class
class DistNEWLoss(torch.nn.Module):
    """[改] Your custom loss.

    The placeholder below is a plain token-mean cross-entropy that trains
    as-is; replace the body, keep the contract: ``logits`` / ``labels`` are
    already shifted to next-token pairs, and the result must be divided by
    ``global_token_denominator(...)`` so the DP mean-reduce yields the
    global token mean (see ops/loss/distributed.py).
    """

    def __init__(self, ignore_index: int = -100, new_param: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.new_param = new_param  # [改]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits.float(), labels, ignore_index=self.ignore_index, reduction="none"
        )
        return self.reduce_per_token_ce(ce_loss, labels)

    def reduce_per_token_ce(self, ce_per_token: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """[改] 流式 loss 路径的归约入口：SFTWorkGroup._compute_loss 经
        FusedLinearCrossEntropy（不物化完整 logits）逐 token 调用它。
        与 forward 同数学；forward 保留给完整 logits 场景和既有测试。"""
        valid_tokens = (labels != self.ignore_index).sum().item()
        denominator = global_token_denominator(valid_tokens)
        return ce_per_token.sum() / max(denominator, 1)


# [改] 定义你的 WorkGroup
class NEWWorkGroup(SFTWorkGroup):
    """Your WorkGroup — override __init__ to set self.criterion and any extra config.

    Inherits update_step and the streaming _compute_loss from SFTWorkGroup;
    the loss path never materializes the full logits (FusedLinearCrossEntropy)
    and calls ``self.criterion.reduce_per_token_ce(ce_per_token, labels)`` --
    so changing self.criterion is usually all you need, as long as the new
    loss implements that method (DistNEWLoss below shows the contract).
    """

    def __init__(self, config):
        super().__init__(config)
        dc = config.data_config
        new_param = getattr(dc, "new_param", 0.0)  # [改] 你的特有配置（见 config.py）
        self.criterion = DistNEWLoss(ignore_index=-100, new_param=new_param)
