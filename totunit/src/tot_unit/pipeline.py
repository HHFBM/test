from __future__ import annotations

from typing import Generic, Protocol, TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Stage(Protocol[InputT, OutputT]):
    """
    Pipeline stage. Implementation decides how to interpret input/output.
    """

    def run(self, input_data: InputT, trace: dict | None = None) -> OutputT: ...


class Pipeline(Generic[InputT, OutputT]):
    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self, input_data: InputT, trace: dict | None = None):
        data = input_data
        for stage in self.stages:
            data = stage.run(data, trace=trace)
        return data


