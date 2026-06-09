"""PaSa-style paper selector for query-specific relevance."""

from __future__ import annotations

import logging
import re

from .models import Paper
from .utils import jaccard_similarity, safe_json_loads, tokenize

logger = logging.getLogger(__name__)


class PaperSelector:
    """Evaluate whether papers satisfy the user's scholarly query.

    PaSa uses a selector model that predicts True/False relevance from title and
    abstract. This implementation mirrors the interface while remaining usable
    without local PaSa checkpoints.
    """

    def __init__(self, cfg=None, cost_callback=None):
        self.cfg = cfg
        self.cost_callback = cost_callback

    async def select(self, query: str, papers: list[Paper]) -> list[Paper]:
        selected: list[Paper] = []
        for paper in papers:
            score, reason = await self.score(query, paper)
            paper.selector_score = score
            paper.selector_reason = reason
            selected.append(paper)
        return selected

    async def score(self, query: str, paper: Paper) -> tuple[float, str]:
        llm_score = await self._llm_score(query, paper)
        if llm_score is not None:
            return llm_score
        return self._fallback_score(query, paper)

    async def _llm_score(self, query: str, paper: Paper) -> tuple[float, str] | None:
        if not self.cfg:
            return None
        prompt = f"""You are evaluating academic search results.

User query: {query}

Searched paper:
Title: {paper.title}
Abstract: {paper.abstract or ""}

Return ONLY JSON:
{{
  "decision": true,
  "score": 0.0,
  "reason": "brief explanation"
}}
"""
        try:
            from gpt_researcher.utils.llm import create_chat_completion

            response = await create_chat_completion(
                model=self.cfg.fast_llm_model,
                messages=[{"role": "user", "content": prompt}],
                llm_provider=self.cfg.fast_llm_provider,
                max_tokens=getattr(self.cfg, "fast_token_limit", 9000),
                temperature=0.0,
                llm_kwargs=getattr(self.cfg, "llm_kwargs", {}),
                cost_callback=self.cost_callback,
            )
            data = safe_json_loads(response)
            score = float(data.get("score", 1.0 if data.get("decision") else 0.0))
            reason = str(data.get("reason") or "")
            return max(0.0, min(score, 1.0)), reason
        except Exception as exc:
            logger.warning("Paper selector fell back for %s: %s", paper.paper_id, exc)
            return None

    @staticmethod
    def _fallback_score(query: str, paper: Paper) -> tuple[float, str]:
        text = f"{paper.title} {paper.abstract or ''}"
        overlap = sorted(tokenize(query) & tokenize(text))
        similarity = jaccard_similarity(query, text)
        title_boost = 0.15 if tokenize(query) & tokenize(paper.title) else 0.0
        survey_boost = 0.05 if re.search(r"\b(survey|benchmark|evaluation|taxonomy)\b", paper.title.lower()) else 0.0
        score = max(0.0, min(1.0, similarity * 1.8 + title_boost + survey_boost))
        if overlap:
            reason = f"Selector fallback matched topic terms: {', '.join(overlap[:6])}."
        else:
            reason = "Selector fallback found weak lexical evidence."
        return round(score, 4), reason
