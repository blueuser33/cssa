"""Ranking for academic papers."""

from __future__ import annotations

import math
from datetime import datetime

from .models import Paper
from .utils import jaccard_similarity, tokenize


class PaperRanker:
    def __init__(self, embeddings=None):
        self.embeddings = embeddings

    async def rank(self, query: str, papers: list[Paper], top_k: int = 30) -> list[Paper]:
        if not papers:
            return []
        max_citations = max((paper.citation_count or 0 for paper in papers), default=0)
        current_year = datetime.now().year

        ranked: list[Paper] = []
        for paper in papers:
            text = f"{paper.title}\n{paper.abstract or ''}"
            semantic_similarity = jaccard_similarity(query, text)
            recency_score = self._recency_score(paper.year, current_year)
            citation_score = self._citation_score(paper.citation_count, max_citations)
            bonus = self._bonus(paper)

            score = (
                semantic_similarity * 0.55
                + recency_score * 0.20
                + citation_score * 0.20
                + bonus * 0.05
            )
            paper.relevance_score = round(score, 4)
            paper.why_relevant = self._why_relevant(query, paper)
            paper.key_contribution = self._key_contribution(paper)
            ranked.append(paper)

        ranked.sort(key=lambda paper: paper.relevance_score or 0, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _recency_score(year: int | None, current_year: int) -> float:
        if year is None:
            return 0.35
        age = max(0, current_year - year)
        return max(0.0, 1.0 - min(age, 12) / 12)

    @staticmethod
    def _citation_score(citation_count: int | None, max_citations: int) -> float:
        if not citation_count or max_citations <= 0:
            return 0.0
        return math.log1p(citation_count) / math.log1p(max_citations)

    @staticmethod
    def _bonus(paper: Paper) -> float:
        title = (paper.title or "").lower()
        keywords = ["survey", "benchmark", "evaluation", "review", "dataset", "foundational"]
        return 1.0 if any(keyword in title for keyword in keywords) else 0.0

    @staticmethod
    def _why_relevant(query: str, paper: Paper) -> str:
        overlaps = sorted(tokenize(query) & tokenize(f"{paper.title} {paper.abstract or ''}"))
        if overlaps:
            return f"Matches topic terms: {', '.join(overlaps[:6])}."
        return "Selected by academic metadata and available abstract."

    @staticmethod
    def _key_contribution(paper: Paper) -> str:
        abstract = (paper.abstract or "").strip().replace("\n", " ")
        if not abstract:
            return "Contribution not available from metadata."
        sentences = [part.strip() for part in abstract.split(".") if part.strip()]
        return sentences[0][:280] + ("..." if len(sentences[0]) > 280 else "")
