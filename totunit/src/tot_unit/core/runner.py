from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .interfaces import Evaluator, Generator, Selector, Stopper
from .types import Candidate, RunResult, StepLog


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class ToTConfig:
    steps: int
    n_generate: int
    n_select: int
    n_evaluate: int


class ToTRunner(Generic[StateT]):
    """
    Generic ToT engine. Does not assume any LLM usage.
    """

    def __init__(
        self,
        generator: Generator[StateT],
        evaluator: Evaluator[StateT],
        selector: Selector[StateT],
        stopper: Stopper[StateT],
        cfg: ToTConfig,
    ) -> None:
        self.generator = generator
        self.evaluator = evaluator
        self.selector = selector
        self.stopper = stopper
        self.cfg = cfg

    def run(self, initial_candidates: list[Candidate[StateT]]) -> RunResult[StateT]:
        current = initial_candidates
        logs: list[StepLog[StateT]] = []

        for step in range(self.cfg.steps):
            candidates = self.generator.generate(step, current, self.cfg.n_generate)
            scores = self.evaluator.evaluate(step, candidates, self.cfg.n_evaluate)
            selected = self.selector.select(candidates, scores, self.cfg.n_select)

            logs.append(StepLog(step=step, candidates=candidates, scores=scores, selected=selected))

            if self.stopper.should_stop(step, selected, scores):
                return RunResult(final_candidates=selected, logs=logs)

            current = selected

        return RunResult(final_candidates=current, logs=logs)


