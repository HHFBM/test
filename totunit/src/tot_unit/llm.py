from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from openai import OpenAI

from .core.interfaces import Evaluator, Generator
from .core.types import Candidate


StateT = type("StateT", (), {})


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    api_base: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 800


class PromptBuilder(Protocol):
    def __call__(self, step: int, candidate: Candidate) -> str: ...


class VotePromptBuilder(Protocol):
    def __call__(self, step: int, candidates: list[Candidate]) -> str: ...


class StopProvider(Protocol):
    def __call__(self, step: int, candidate: Candidate) -> Optional[str]: ...


class OpenAICompatibleClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.api_base)

    def _max_n_per_request(self) -> int:
        env_limit = os.getenv("TOT_MAX_N_PER_REQUEST")
        if env_limit:
            try:
                return max(1, int(env_limit))
            except ValueError:
                pass
        if "dashscope.aliyuncs.com" in self.cfg.api_base:
            return 4
        return 20

    def chat(self, prompt: str, n: int, stop: Optional[str]) -> list[str]:
        outputs: list[str] = []
        remaining = n
        max_n = self._max_n_per_request()
        while remaining > 0:
            cnt = min(remaining, max_n)
            remaining -= cnt
            response = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                n=cnt,
                stop=stop,
            )
            outputs.extend([c.message.content or "" for c in response.choices])
        return outputs


class StepRouter:
    def __init__(self, clients: list[OpenAICompatibleClient]):
        if not clients:
            raise ValueError("clients must not be empty")
        self.clients = clients

    def __call__(self, step: int) -> OpenAICompatibleClient:
        if step < len(self.clients):
            return self.clients[step]
        return self.clients[-1]


class LLMGenerator(Generator):
    def __init__(
        self,
        client_for_step: Callable[[int], OpenAICompatibleClient],
        prompt_builder: PromptBuilder,
        stop_provider: Optional[StopProvider] = None,
    ):
        self.client_for_step = client_for_step
        self.prompt_builder = prompt_builder
        self.stop_provider = stop_provider

    def generate(self, step: int, current: list[Candidate], n_generate: int) -> list[Candidate]:
        out: list[Candidate] = []
        client = self.client_for_step(step)
        for cand in current:
            prompt = self.prompt_builder(step, cand)
            stop = self.stop_provider(step, cand) if self.stop_provider else None
            samples = client.chat(prompt=prompt, n=n_generate, stop=stop)
            for sample in samples:
                out.append(Candidate(state=cand.state, text=cand.text + sample, meta={"step": step}))
        return out


class LLMVoteEvaluator(Evaluator):
    def __init__(
        self,
        client_for_step: Callable[[int], OpenAICompatibleClient],
        vote_prompt_builder: VotePromptBuilder,
    ):
        self.client_for_step = client_for_step
        self.vote_prompt_builder = vote_prompt_builder

    def evaluate(self, step: int, candidates: list[Candidate], n_evaluate: int) -> list[float]:
        if not candidates:
            return []
        client = self.client_for_step(step)
        prompt = self.vote_prompt_builder(step, candidates)
        outputs = client.chat(prompt=prompt, n=n_evaluate, stop=None)
        votes = [0 for _ in candidates]
        for out in outputs:
            # caller decides how to parse; default: use last integer in text
            import re

            match = re.findall(r"(\d+)", out)
            if not match:
                continue
            pick = int(match[-1]) - 1
            if 0 <= pick < len(votes):
                votes[pick] += 1
        return [float(v) for v in votes]


