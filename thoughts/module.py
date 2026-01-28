"""Thoughts module built on a Tree-of-Thoughts flow.

This module generates innovative outlines and structured writing guidelines
that can be passed into a downstream writing module. Knowledge-base recall
is intentionally stubbed and can be filled once a DB interface is provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class ThoughtInput:
    topic: str
    seed_viewpoints: List[str] = field(default_factory=list)
    target_audience: str = "general"
    language: str = "zh"


@dataclass(frozen=True)
class OutlineNode:
    title: str
    bullets: List[str] = field(default_factory=list)
    children: List["OutlineNode"] = field(default_factory=list)


@dataclass(frozen=True)
class WritingGuideline:
    structure_requirements: List[str]
    style_requirements: List[str]
    perspective_requirements: List[str]
    quality_requirements: List[str]
    format_template: List[str]


@dataclass(frozen=True)
class ThoughtOutput:
    keywords: List[str]
    outline: OutlineNode
    guidelines: WritingGuideline
    constraints: List[str]
    knowledge_recall_notes: List[str]


class KnowledgeBaseClient:
    """Interface for knowledge-base recall.

    Implement fetch_related_articles to return snippets or metadata that can
    enrich the outline. The default implementation should be injected once
    DB callbacks are available.
    """

    def fetch_related_articles(self, keywords: Iterable[str]) -> List[str]:
        raise NotImplementedError


class NullKnowledgeBaseClient(KnowledgeBaseClient):
    """Stub knowledge-base client to reserve the recall interface."""

    def fetch_related_articles(self, keywords: Iterable[str]) -> List[str]:
        return []


class ThoughtsModule:
    """Tree-of-Thoughts pipeline for generating outline + writing rules."""

    def __init__(self, knowledge_client: Optional[KnowledgeBaseClient] = None) -> None:
        self.knowledge_client = knowledge_client or NullKnowledgeBaseClient()

    def run(self, thought_input: ThoughtInput) -> ThoughtOutput:
        keywords = self.extract_keywords(thought_input)
        knowledge_hits = self.search_knowledge_base(keywords)
        recall_notes = self.recall_and_learn(knowledge_hits)
        perspectives = self.multi_perspective_analysis(thought_input, keywords)
        outline = self.generate_outline(thought_input, keywords, perspectives)
        guidelines = self.build_writing_guidelines(thought_input)
        constraints = self.merge_with_benchmark(guidelines, perspectives)

        return ThoughtOutput(
            keywords=keywords,
            outline=outline,
            guidelines=guidelines,
            constraints=constraints,
            knowledge_recall_notes=recall_notes,
        )

    def extract_keywords(self, thought_input: ThoughtInput) -> List[str]:
        base_keywords = [thought_input.topic]
        base_keywords.extend(thought_input.seed_viewpoints)
        return list(dict.fromkeys([word.strip() for word in base_keywords if word.strip()]))

    def search_knowledge_base(self, keywords: Iterable[str]) -> List[str]:
        return self.knowledge_client.fetch_related_articles(keywords)

    def recall_and_learn(self, knowledge_hits: Iterable[str]) -> List[str]:
        return [
            f"recall: {item}" for item in knowledge_hits
        ]

    def multi_perspective_analysis(
        self, thought_input: ThoughtInput, keywords: Iterable[str]
    ) -> List[str]:
        perspectives = [
            "学术学者视角：数据真实可查，专业术语准确，摘要与关键词提取精准",
            "审稿人视角：目标清晰，方法与数据足够支撑观点，结构完整，句子不过长",
            "读者视角：内容清晰可理解，观点可验证，学术表达同时具备可读性",
        ]
        if thought_input.target_audience != "academic":
            perspectives.append("普适性视角：兼顾可读性与结构逻辑，避免过度术语化")
        if keywords:
            perspectives.append(f"关键词聚焦：围绕 {', '.join(keywords)} 展开新角度分析")
        return perspectives

    def generate_outline(
        self, thought_input: ThoughtInput, keywords: Iterable[str], perspectives: List[str]
    ) -> OutlineNode:
        return OutlineNode(
            title=f"{thought_input.topic} 文章大纲",
            bullets=[
                "明确研究/讨论目标与核心问题",
                "总结已有观点并提炼关键矛盾或空白",
                "提出创新视角与可验证的主张",
            ],
            children=[
                OutlineNode(
                    title="引言",
                    bullets=[
                        "背景与问题陈述",
                        "研究动机与意义",
                        "文章结构说明",
                    ],
                ),
                OutlineNode(
                    title="相关研究与知识回顾",
                    bullets=[
                        "关键词与概念定义",
                        "知识库检索与要点归纳（占位）",
                        "现有研究不足与争议",
                    ],
                ),
                OutlineNode(
                    title="创新视角与论点展开",
                    bullets=[
                        "多视角分析结果整合",
                        "提出新的理解框架",
                        "论点与证据链条",
                    ],
                    children=[
                        OutlineNode(
                            title="关键论点 1",
                            bullets=["论点描述", "数据/案例支撑", "可能反驳与回应"],
                        ),
                        OutlineNode(
                            title="关键论点 2",
                            bullets=["论点描述", "数据/案例支撑", "可能反驳与回应"],
                        ),
                    ],
                ),
                OutlineNode(
                    title="讨论与展望",
                    bullets=[
                        "与现有研究的对比",
                        "可推广的启示",
                        "后续研究方向",
                    ],
                ),
                OutlineNode(
                    title="结论",
                    bullets=["总结主旨", "回应核心问题", "强调创新贡献"],
                ),
                OutlineNode(
                    title="写作规范与格式",
                    bullets=perspectives,
                ),
            ],
        )

    def build_writing_guidelines(self, thought_input: ThoughtInput) -> WritingGuideline:
        return WritingGuideline(
            structure_requirements=[
                "包含摘要、关键词、引言、方法/论证、结果/讨论、结论",
                "段落层次清晰，逻辑递进",
                "各章节标题与内容一致",
            ],
            style_requirements=[
                "学术表达准确，避免口语化",
                "句子长度适中，避免冗长",
                "保证术语统一与定义明确",
            ],
            perspective_requirements=[
                "学术学者：数据真实可查，术语准确",
                "审稿人：目标清晰，证据充分，结构严谨",
                "读者：观点清晰，可验证，易理解",
            ],
            quality_requirements=[
                "论点可验证且有证据支撑",
                "观点与证据匹配，无逻辑跳跃",
                "保证完整性与一致性",
            ],
            format_template=[
                "标题",
                "摘要（Abstract）",
                "关键词（Keywords）",
                "引言（Introduction）",
                "方法/论证（Method/Argument）",
                "结果与讨论（Results & Discussion）",
                "结论（Conclusion）",
            ],
        )

    def merge_with_benchmark(
        self, guidelines: WritingGuideline, perspectives: List[str]
    ) -> List[str]:
        benchmark_constraints = [
            "格式排版正确，结构完整",
            "使用专业学术表达",
            "参考 benchmark 写作标准（占位）",
        ]
        return benchmark_constraints + guidelines.structure_requirements + perspectives
