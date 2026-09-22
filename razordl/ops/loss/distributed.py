import torch

from razordl.ops.distributed.utils import all_gather_object


def distributed_token_count(local_count: int | float) -> float:
    """Sum of *local_count* over every rank (1 rank when dist is off)."""
    return float(sum(all_gather_object(local_count)))


def global_token_denominator(local_count: int | float) -> float:
    """Denominator that turns a per-rank token sum into the global token mean.

    Returns ``global_count / world_size``.  FSDP2 and DDP *average* gradients
    across ranks, so a rank that computes ``local_sum / global_count`` ends up
    contributing ``1/W`` of the true global-mean gradient -- and the rank-mean
    of the logged losses (``all_gather_object(float_mean=True)``) is ``1/W`` of
    the real per-token loss (an 8-GPU run logged one eighth of its CE).
    Dividing by ``global_count / W`` instead makes both the reduced gradient
    and the logged value equal the global token mean exactly, on any world
    size, including the SP ranks that share a sequence.
    """
    gathered = all_gather_object(local_count)
    return float(sum(gathered)) / len(gathered)


class DistCrossEntropyLoss(torch.nn.Module):
    """Cross-entropy loss averaged by valid tokens across distributed ranks."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def forward(self, logits, labels):
        # Upcast to fp32 before the softmax and the sum.  In fp16 a per-token
        # loss sum over a few thousand tokens overflows 65504 to inf; even
        # below that the reduction loses precision.
        return self.reduce_per_token_ce(self.ce(logits.float(), labels), labels)

    def reduce_per_token_ce(self, ce_per_token, labels):
        """(per-token CE, labels) -> sum / global_token_denominator.

        Same math as forward(); the per-token CE may come from
        FusedLinearCrossEntropy (the streaming SFT/DFT loss path) instead of
        materialized logits."""
        valid_tokens = (labels != self.ignore_index).sum().item()
        denominator = global_token_denominator(valid_tokens)
        return ce_per_token.sum() / max(denominator, 1)
