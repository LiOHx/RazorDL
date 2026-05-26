"""Distributed metric leaves with explicit reduction semantics.

`_summarize_step_info` in `core/base/trainer.py` collapses every numeric
leaf of `step_info` to its mean across ranks and micro-steps.  That is
correct for plain scalars (loss, grad_norm, ...), but wrong for
distribution-shaped quantities — e.g. the per-rank `max` of a reward
batch must be combined with `max`, not averaged.

This module provides the escape hatch: ship a `Reducible` object instead
of a raw float, and the aggregator will call its `merge` method.  Add a
new distribution kind by subclassing `Reducible`; no aggregator change
needed.
"""

import abc
import math
from dataclasses import dataclass


__all__ = ["Reducible", "DistStats"]


class Reducible(abc.ABC):
    """A step_info leaf that owns its own cross-rank reduction.

    Subclasses must be importable from their defining module at the top
    level so `all_gather_object` (which pickles) can resolve them on the
    receiving rank.
    """

    @abc.abstractmethod
    def merge(self, other: "Reducible") -> "Reducible":
        """Combine with another value of the same type."""

    @abc.abstractmethod
    def to_logged(self) -> dict | float:
        """Unfold into the final loggable representation (float or dict)."""


@dataclass
class DistStats(Reducible):
    """Sufficient statistics for a distribution.

    Pools exactly across ranks and micro-steps — the merged `mean`,
    `std`, `min`, `max`, `n` describe the union of all samples seen.
    `std` is population std (var = sum_sq/n - mean²); we do not apply
    Bessel's correction because the n/(n-1) factor is awkward to track
    when merging variable-sized partitions and is not what callers want
    for logging anyway.
    """

    sum: float
    sum_sq: float
    n: int
    min: float
    max: float

    @classmethod
    def from_tensor(cls, t) -> "DistStats":
        t = t.detach().double()
        if t.numel() == 0:
            return cls.empty()
        return cls(
            sum=t.sum().item(),
            sum_sq=(t * t).sum().item(),
            n=int(t.numel()),
            min=t.min().item(),
            max=t.max().item(),
        )

    @classmethod
    def empty(cls) -> "DistStats":
        return cls(0.0, 0.0, 0, float("inf"), float("-inf"))

    def merge(self, o: "DistStats") -> "DistStats":
        return DistStats(
            sum=self.sum + o.sum,
            sum_sq=self.sum_sq + o.sum_sq,
            n=self.n + o.n,
            min=min(self.min, o.min),
            max=max(self.max, o.max),
        )

    def to_logged(self) -> dict:
        if self.n == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        mean = self.sum / self.n
        var = max(self.sum_sq / self.n - mean * mean, 0.0)
        return {
            "mean": mean,
            "std": math.sqrt(var),
            "min": self.min,
            "max": self.max,
            "n": self.n,
        }
