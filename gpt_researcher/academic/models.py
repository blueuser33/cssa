"""Data models for the academic survey workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    citation_count: int | None = None
    source: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    relevance_score: float | None = None
    why_relevant: str | None = None
    key_contribution: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperSummary:
    paper_id: str
    problem: str
    method: str
    datasets: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaxonomyCategory:
    name: str
    description: str
    paper_ids: list[str]
    strengths: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    applicable_scenarios: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CitationAuditItem:
    claim: str
    citation: str
    status: str
    support_score: float
    evidence: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
