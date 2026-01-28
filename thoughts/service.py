"""Service layer for the Tree-of-Thoughts module."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from thoughts.module import KnowledgeBaseClient, ThoughtInput, ThoughtOutput, ThoughtsModule


def _outline_to_dict(outline: Any) -> Dict[str, Any]:
    return {
        "title": outline.title,
        "bullets": list(outline.bullets),
        "children": [_outline_to_dict(child) for child in outline.children],
    }


class ThoughtsService:
    """Service wrapper to decouple the FastAPI layer from core logic."""

    def __init__(self, knowledge_client: Optional[KnowledgeBaseClient] = None) -> None:
        self.module = ThoughtsModule(knowledge_client=knowledge_client)

    def generate(self, thought_input: ThoughtInput) -> ThoughtOutput:
        return self.module.run(thought_input)

    def generate_payload(self, thought_input: ThoughtInput) -> Dict[str, Any]:
        output = self.generate(thought_input)
        guidelines = asdict(output.guidelines)
        return {
            "keywords": output.keywords,
            "outline": _outline_to_dict(output.outline),
            "guidelines": guidelines,
            "constraints": output.constraints,
            "knowledge_recall_notes": output.knowledge_recall_notes,
        }
