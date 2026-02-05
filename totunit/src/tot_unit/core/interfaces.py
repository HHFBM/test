from __future__ import annotations

from typing import Generic, Protocol, TypeVar

from .types import Candidate


StateT = TypeVar("StateT")


class Generator(Protocol[StateT]):
    """
    Expand current candidates into next-step candidates.
    Implementation can be rule-based, model-based, programmatic, etc.
    """

    def generate(self, step: int, current: list[Candidate[StateT]], n_generate: int) -> list[Candidate[StateT]]: ...


class Evaluator(Protocol[StateT]):
    """
    Score candidates. Output is a list of floats aligned with candidates.
    Higher is better. Implementation is arbitrary (LLM judge, heuristic, RM).
    """

    def evaluate(self, step: int, candidates: list[Candidate[StateT]], n_evaluate: int) -> list[float]: ...


class Selector(Protocol[StateT]):
    """
    Select a subset of candidates based on scores.
    """

    def select(self, candidates: list[Candidate[StateT]], scores: list[float], n_select: int) -> list[Candidate[StateT]]: ...


class Stopper(Protocol[StateT]):
    """
    Decide whether ToT should stop early.
    """

    def should_stop(self, step: int, selected: list[Candidate[StateT]], scores: list[float]) -> bool: ...


