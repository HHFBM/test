from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Protocol

from .core.interfaces import Evaluator, Generator, Selector, Stopper
from .core.runner import ToTConfig, ToTRunner
from .core.types import Candidate, RunResult
from .llm import LLMConfig, LLMGenerator, LLMVoteEvaluator, OpenAICompatibleClient, StepRouter


class PromptBuilder(Protocol):
    def __call__(self, step: int, candidate: Candidate) -> str: ...


class VotePromptBuilder(Protocol):
    def __call__(self, step: int, candidates: list[Candidate]) -> str: ...


class StopProvider(Protocol):
    def __call__(self, step: int, candidate: Candidate) -> Optional[str]: ...


@dataclass(frozen=True)
class LLMToTStepConfig:
    gen: LLMConfig
    judge: LLMConfig


@dataclass(frozen=True)
class LLMToTConfig:
    steps: int
    n_generate: int
    n_select: int
    n_evaluate: int
    step_llms: list[LLMToTStepConfig]
    prompt_builder: PromptBuilder
    vote_prompt_builder: VotePromptBuilder
    stop_provider: Optional[StopProvider] = None

    def __post_init__(self) -> None:
        if self.steps != len(self.step_llms):
            raise ValueError("steps must match len(step_llms)")


@dataclass
class LLMToT:
    """
    High-level ToT wrapper with LLM defaults, supporting component override.
    """

    cfg: LLMToTConfig
    generator: Generator | None = None
    evaluator: Evaluator | None = None
    selector: Selector | None = None
    stopper: Stopper | None = None

    def _build_generator(self) -> Generator:
        gen_clients = [OpenAICompatibleClient(s.gen) for s in self.cfg.step_llms]
        gen_router = StepRouter(gen_clients)
        return LLMGenerator(
            client_for_step=gen_router,
            prompt_builder=self.cfg.prompt_builder,
            stop_provider=self.cfg.stop_provider,
        )

    def _build_evaluator(self) -> Evaluator:
        judge_clients = [OpenAICompatibleClient(s.judge) for s in self.cfg.step_llms]
        judge_router = StepRouter(judge_clients)
        return LLMVoteEvaluator(
            client_for_step=judge_router,
            vote_prompt_builder=self.cfg.vote_prompt_builder,
        )

    def build_runner(self) -> ToTRunner:
        generator = self.generator or self._build_generator()
        evaluator = self.evaluator or self._build_evaluator()
        if self.selector is None or self.stopper is None:
            raise ValueError("selector and stopper must be provided")

        return ToTRunner(
            generator=generator,
            evaluator=evaluator,
            selector=self.selector,
            stopper=self.stopper,
            cfg=ToTConfig(
                steps=self.cfg.steps,
                n_generate=self.cfg.n_generate,
                n_select=self.cfg.n_select,
                n_evaluate=self.cfg.n_evaluate,
            ),
        )

    def run(self, initial_candidates: list[Candidate]) -> RunResult:
        runner = self.build_runner()
        return runner.run(initial_candidates=initial_candidates)

    def override(
        self,
        *,
        generator: Generator | None = None,
        evaluator: Evaluator | None = None,
        selector: Selector | None = None,
        stopper: Stopper | None = None,
    ) -> "LLMToT":
        return LLMToT(
            cfg=self.cfg,
            generator=generator if generator is not None else self.generator,
            evaluator=evaluator if evaluator is not None else self.evaluator,
            selector=selector if selector is not None else self.selector,
            stopper=stopper if stopper is not None else self.stopper,
        )


