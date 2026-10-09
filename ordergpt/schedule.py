"""Three-milestone curriculum shared by all four data domains."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class OrderSchedule:
    distill_start: int
    distill_end: int
    anneal_end: int

    def __post_init__(self):
        if not 0 <= self.distill_start < self.distill_end < self.anneal_end:
            raise ValueError("require 0 <= distill_start < distill_end < anneal_end")

    def probability(self, step: int) -> float:
        return min(1.0, max(0.0, (step - self.distill_end) / (self.anneal_end - self.distill_end)))

    def captures_attention(self, step: int) -> bool:
        return self.distill_start <= step < self.anneal_end

    def distills(self, step: int) -> bool:
        return self.distill_start <= step < self.distill_end
