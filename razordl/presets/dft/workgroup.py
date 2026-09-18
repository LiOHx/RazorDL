"""DFT (Direct Fine-Tuning) preset workgroup.

DFT uses confidence-weighted cross-entropy: each token's loss is
weighted by ``exp(-ce_loss)``, so high-confidence tokens contribute
more to the final loss.  This differs from standard SFT which treats
every token equally.
"""

import torch

from razordl.core.base import logging
from razordl.ops.loss.distributed import global_token_denominator
from razordl.presets.sft.workgroup import SFTModelGroup, SFTWorkGroup

logger = logging.getLogger(__name__)


class DistDFTLoss(torch.nn.Module):
    """Confidence-weighted cross-entropy loss for Direct Fine-Tuning.

    Weight = exp(-ce_loss_per_token), clamped to ``[mini_scale, inf)``.
    Aggregate: ``sum(ce_loss * weight) / global_token_denominator``.
    """

    def __init__(self, ignore_index: int = -100, mini_scale: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.mini_scale = mini_scale

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits and labels are already shifted by the caller.  fp32 for the
        # same reason as DistCrossEntropyLoss: the per-token sum overflows fp16.
        ce_loss = torch.nn.functional.cross_entropy(
            logits.float(), labels,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # The weight is a stop-gradient term (DFT: -sum sg(p) log p).  Without
        # detach the gradient is exp(-ce) * (1 - ce) * d(ce), which flips sign
        # for every token with ce > 1 and pushes low-confidence tokens further
        # away instead of towards the target.
        p_target = torch.exp(-ce_loss).detach()
        p_target = torch.clamp(p_target, min=self.mini_scale)

        valid_mask = (labels != self.ignore_index).float()
        valid_tokens = valid_mask.sum().item()
        denominator = global_token_denominator(valid_tokens)

        loss = (ce_loss * p_target).sum() / max(denominator, 1)
        return loss


class DFTWorkGroup(SFTWorkGroup):
    """DFT WorkGroup with confidence-weighted cross-entropy loss.

    Inherits the full training-step flow from :class:`SFTWorkGroup`
    (sequence-parallel split, gradient backward, etc.) and only
    replaces the loss criterion.
    """

    def __init__(self, config):
        super().__init__(config)
        dc = config.data_config
        mini_scale = getattr(dc, "dft_mini_scale", 0.0)
        self.criterion = DistDFTLoss(ignore_index=-100, mini_scale=mini_scale)
        self.chunked_loss = False  # DFT uses full loss for confidence weights
