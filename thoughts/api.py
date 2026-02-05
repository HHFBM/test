"""FastAPI router for the Tree-of-Thoughts service."""

from __future__ import annotations

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from thoughts.module import ThoughtInput
from thoughts.service import ThoughtsService


class ThoughtRequest(BaseModel):
    topic: str = Field(..., description="Main topic for the article.")
    seed_viewpoints: List[str] = Field(default_factory=list)
    target_audience: str = Field(default="general")
    language: str = Field(default="zh")


class OutlineNodeResponse(BaseModel):
    title: str
    bullets: List[str]
    children: List["OutlineNodeResponse"] = Field(default_factory=list)


OutlineNodeResponse.model_rebuild()


class WritingGuidelineResponse(BaseModel):
    structure_requirements: List[str]
    style_requirements: List[str]
    perspective_requirements: List[str]
    quality_requirements: List[str]
    format_template: List[str]


class ThoughtResponse(BaseModel):
    keywords: List[str]
    outline: OutlineNodeResponse
    guidelines: WritingGuidelineResponse
    constraints: List[str]
    knowledge_recall_notes: List[str]


def create_app(service: ThoughtsService | None = None) -> FastAPI:
    app = FastAPI(title="Tree-of-Thoughts API")
    thought_service = service or ThoughtsService()

    @app.post("/thoughts", response_model=ThoughtResponse)
    def generate_thoughts(payload: ThoughtRequest) -> ThoughtResponse:
        thought_input = ThoughtInput(
            topic=payload.topic,
            seed_viewpoints=payload.seed_viewpoints,
            target_audience=payload.target_audience,
            language=payload.language,
        )
        output = thought_service.generate_payload(thought_input)
        return ThoughtResponse(**output)

    return app
