from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .interfaces import Stopper
from .types import Candidate


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class MaxStepStopper(Stopper[StateT]):
    max_step: int

    def should_stop(self, step: int, selected: list[Candidate[StateT]], scores: list[float]) -> bool:
        return step >= self.max_step


@dataclass(frozen=True)
class ScoreThresholdStopper(Stopper[StateT]):
    threshold: float

    def should_stop(self, step: int, selected: list[Candidate[StateT]], scores: list[float]) -> bool:
        return max(scores) >= self.threshold if scores else False


