from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Generic, TypeVar

from .interfaces import Selector
from .types import Candidate


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class GreedySelector(Selector[StateT]):
    def select(self, candidates: list[Candidate[StateT]], scores: list[float], n_select: int) -> list[Candidate[StateT]]:
        if len(candidates) != len(scores):
            raise ValueError("candidates and scores must have the same length")
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in order[:n_select]]


@dataclass(frozen=True)
class SampleSelector(Selector[StateT]):
    seed: int | None = 42

    def select(self, candidates: list[Candidate[StateT]], scores: list[float], n_select: int) -> list[Candidate[StateT]]:
        if len(candidates) != len(scores):
            raise ValueError("candidates and scores must have the same length")
        rng = random.Random(self.seed)
        weights = [max(0.0, float(s)) for s in scores]
        total = sum(weights)
        if total <= 0:
            idx = list(range(len(candidates)))
            rng.shuffle(idx)
            return [candidates[i] for i in idx[:n_select]]

        chosen: list[int] = []
        pool = list(range(len(candidates)))
        for _ in range(min(n_select, len(pool))):
            r = rng.random() * total
            acc = 0.0
            pick = pool[-1]
            for i in pool:
                acc += weights[i]
                if acc >= r:
                    pick = i
                    break
            chosen.append(pick)
            total -= weights[pick]
            pool.remove(pick)
        return [candidates[i] for i in chosen]


