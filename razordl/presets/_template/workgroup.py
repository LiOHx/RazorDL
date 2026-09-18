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
        valid_tokens = (labels != self.ignore_index).sum().item()
        denominator = global_token_denominator(valid_tokens)
        return ce_loss.sum() / max(denominator, 1)


# [改] 定义你的 WorkGroup
class NEWWorkGroup(SFTWorkGroup):
    """Your WorkGroup — override __init__ to set self.criterion and any extra config.

    Inherits update_step, _compute_loss, _simple_loss_compute, _chunked_loss_compute
    from SFTWorkGroup.  Those methods use self.criterion internally, so changing
    self.criterion is usually all you need.
    """

    def __init__(self, config):
        super().__init__(config)
        dc = config.data_config
        new_param = getattr(dc, "new_param", 0.0)  # [改] 你的特有配置（见 config.py）
        self.criterion = DistNEWLoss(ignore_index=-100, new_param=new_param)
        self.chunked_loss = False  # 如果你的 loss 需要完整 logits
