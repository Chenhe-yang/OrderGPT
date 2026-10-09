"""Sample-counter view of the shared three-milestone order lifecycle."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ordergpt.schedule import OrderSchedule


@dataclass(frozen=True)
class SampleTeacherScheduleSnapshot:
    train_samples_seen: int
    phase: str
    learned_order_probability: float
    collect_attention: bool
    cache_updates_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SampleTeacherSchedule:
    PHASE_TO_ID = {"warmup_random": 0, "discovery_random": 1, "anneal_refresh": 2, "fixed_order": 3}

    def __init__(self, distill_start_samples: int, distill_end_samples: int, anneal_end_samples: int):
        self.schedule = OrderSchedule(distill_start_samples, distill_end_samples, anneal_end_samples)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]):
        samples = int(config["batch_size"]) * int(config["gradient_accumulation_steps"])
        return cls(*(int(config[key]) * samples for key in ("distill_start", "distill_end", "anneal_end")))

    def snapshot(self, train_samples_seen: int) -> SampleTeacherScheduleSnapshot:
        samples = int(train_samples_seen)
        if samples < 0:
            raise ValueError("train_samples_seen must be non-negative")
        if samples < self.schedule.distill_start:
            phase = "warmup_random"
        elif samples < self.schedule.distill_end:
            phase = "discovery_random"
        elif samples < self.schedule.anneal_end:
            phase = "anneal_refresh"
        else:
            phase = "fixed_order"
        capture = self.schedule.captures_attention(samples)
        return SampleTeacherScheduleSnapshot(samples, phase, self.schedule.probability(samples), capture, capture)

    def to_dict(self) -> dict[str, Any]:
        return {"schedule_unit": "train_sample_presentations", **asdict(self.schedule)}
