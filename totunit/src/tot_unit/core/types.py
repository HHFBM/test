from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class Candidate(Generic[StateT]):
    state: StateT
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trace:
    trace_id: str
    parent_span_id: str | None = None
    span_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepLog(Generic[StateT]):
    step: int
    candidates: list[Candidate[StateT]]
    scores: list[float]
    selected: list[Candidate[StateT]]


@dataclass(frozen=True)
class RunResult(Generic[StateT]):
    final_candidates: list[Candidate[StateT]]
    logs: list[StepLog[StateT]]


