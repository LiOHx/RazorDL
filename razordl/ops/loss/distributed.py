import torch

from razordl.ops.distributed.utils import all_gather_object


def distributed_token_count(local_count: int | float) -> float:
    return float(sum(all_gather_object(local_count)))


class DistCrossEntropyLoss(torch.nn.Module):
    """Cross-entropy loss averaged by valid tokens across distributed ranks."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        valid_tokens = (labels != self.ignore_index).sum().item()
        batch_valid_tokens = distributed_token_count(valid_tokens)
        return ce_loss.sum() / max(batch_valid_tokens, 1)
