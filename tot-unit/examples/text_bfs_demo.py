from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tot_unit.core import Candidate
from tot_unit.core.selectors import GreedySelector
from tot_unit.core.stoppers import MaxStepStopper
from tot_unit.llm import LLMConfig
from tot_unit.llm_tot import LLMToT, LLMToTConfig, LLMToTStepConfig


@dataclass(frozen=True)
class TextState:
    prompt: str
    text: str


COT_PROMPT = (
    "Write a coherent passage of 4 short paragraphs. "
    "The end sentence of each paragraph must be: {input}\n\n"
    "Make a plan then write. Your output should be of the following format:\n\n"
    "Plan:\nYour plan here.\n\n"
    "Passage:\nYour passage here.\n"
)

VOTE_PROMPT = (
    "Given an instruction and several choices, decide which choice is most promising. "
    'Analyze each choice in detail, then conclude in the last line "The best choice is {s}", '
    "where s the integer id of the choice.\n"
)


def _prompt_builder(step: int, cand: Candidate[TextState]) -> str:
    base_prompt = COT_PROMPT.format(input=cand.state.prompt)
    return base_prompt + cand.text


def _stop_provider(step: int, cand: Candidate[TextState]) -> Optional[str]:
    # step 0: stop at plan boundary; step 1: full passage
    return "\nPassage:\n" if step == 0 else None


def _vote_prompt_builder(step: int, candidates: List[Candidate[TextState]]) -> str:
    prompt = VOTE_PROMPT
    for i, cand in enumerate(candidates, 1):
        prompt += f"Choice {i}:\n{cand.text}\n"
    return prompt


def _llm_config(role: str) -> LLMConfig:
    """
    Minimal hard-coded config:
    - base/model are hard-coded
    - key must still be provided via env (do NOT hardcode secrets)
    """
    if role == "gen":
        api_key = os.getenv("TOT_GEN_API_KEY") or os.getenv("TOT_API_KEY") or ""
        api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = "deepseek-v3.2"
        temperature = 0.7
    else:
        api_key = os.getenv("TOT_JUDGE_API_KEY") or os.getenv("TOT_API_KEY") or ""
        api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = "deepseek-v3.2"
        temperature = 0.7

    if not api_key:
        raise RuntimeError("Missing API key. Set TOT_API_KEY or role-specific key env.")

    return LLMConfig(
        api_key=api_key,
        api_base=api_base,
        model=model,
        temperature=temperature,
    )


def main():
    # Minimal, single-task demo
    prompt = "I went to the park yesterday."
    initial = [Candidate(state=TextState(prompt=prompt, text=""), text="")]

    steps = 2  # plan -> passage
    gen_cfg = _llm_config("gen")
    judge_cfg = _llm_config("judge")

    # Per-step pairing for readability (keep length == steps)
    step_llms = [
        LLMToTStepConfig(gen=gen_cfg, judge=judge_cfg),  # step 0: plan
        LLMToTStepConfig(gen=gen_cfg, judge=judge_cfg),  # step 1: passage
    ]

    llm_tot = LLMToT(
        cfg=LLMToTConfig(
            steps=steps,
            n_generate=4,
            n_select=1,
            n_evaluate=5,
            step_llms=step_llms,
            prompt_builder=_prompt_builder,
            vote_prompt_builder=_vote_prompt_builder,
            stop_provider=_stop_provider,
        ),
        selector=GreedySelector(),
        stopper=MaxStepStopper(max_step=steps - 1),
    )

    result = llm_tot.run(initial_candidates=initial)
    best = result.final_candidates[0].text if result.final_candidates else ""

    print(f"\n=== Input ===\n{prompt}\n")
    for step_log in result.logs:
        scored = sorted(
            zip(step_log.candidates, step_log.scores),
            key=lambda x: x[1],
            reverse=True,
        )
        top = scored[0]
        snippet = top[0].text[:200].replace("\n", " ")
        print(f"[step {step_log.step}] top1 score={top[1]:.2f} | {snippet}...")

    print("\n--- Best ---\n")
    print(best)


if __name__ == "__main__":
    main()

