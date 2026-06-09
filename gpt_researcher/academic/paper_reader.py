"""Structured paper reading and summarization."""

from __future__ import annotations

import asyncio
import logging

from .models import Paper, PaperSummary
from .utils import safe_json_loads

logger = logging.getLogger(__name__)


class PaperReader:
    def __init__(self, cfg, prompt_family=None, cost_callback=None):
        self.cfg = cfg
        self.prompt_family = prompt_family
        self.cost_callback = cost_callback

    async def summarize(self, paper: Paper) -> PaperSummary:
        if not paper.abstract:
            return self._fallback_summary(paper)

        prompt = self._prompt(paper)
        try:
            from gpt_researcher.utils.llm import create_chat_completion

            response = await create_chat_completion(
                model=self.cfg.fast_llm_model,
                messages=[{"role": "user", "content": prompt}],
                llm_provider=self.cfg.fast_llm_provider,
                max_tokens=getattr(self.cfg, "fast_token_limit", 3000),
                temperature=getattr(self.cfg, "temperature", 0.2),
                llm_kwargs=getattr(self.cfg, "llm_kwargs", {}),
                cost_callback=self.cost_callback,
            )
            data = safe_json_loads(response)
            return PaperSummary(
                paper_id=paper.paper_id,
                problem=str(data.get("problem") or ""),
                method=str(data.get("method") or ""),
                datasets=self._as_list(data.get("datasets")),
                metrics=self._as_list(data.get("metrics")),
                findings=self._as_list(data.get("findings")),
                limitations=self._as_list(data.get("limitations")),
                evidence=self._as_list(data.get("evidence")),
            )
        except Exception as exc:
            logger.warning("Paper summary fell back for %s: %s", paper.paper_id, exc)
            return self._fallback_summary(paper)

    async def summarize_many(self, papers: list[Paper]) -> list[PaperSummary]:
        tasks = [self.summarize(paper) for paper in papers]
        return await asyncio.gather(*tasks)

    @staticmethod
    def _prompt(paper: Paper) -> str:
        return f"""You are extracting structured information for an academic survey.

Return ONLY valid JSON with this schema:
{{
  "problem": "research problem addressed by the paper",
  "method": "core method or approach",
  "datasets": ["dataset names"],
  "metrics": ["metric names"],
  "findings": ["main empirical or conceptual findings"],
  "limitations": ["limitations or open issues"],
  "evidence": ["short evidence snippets grounded in the abstract"]
}}

Paper ID: {paper.paper_id}
Title: {paper.title}
Authors: {", ".join(paper.authors)}
Year: {paper.year}
Venue: {paper.venue}
Abstract:
{paper.abstract}
"""

    @staticmethod
    def _as_list(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)] if str(value).strip() else []

    @staticmethod
    def _fallback_summary(paper: Paper) -> PaperSummary:
        abstract = (paper.abstract or "").strip()
        first_sentence = abstract.split(".")[0].strip() if abstract else ""
        return PaperSummary(
            paper_id=paper.paper_id,
            problem=first_sentence or "Not available from metadata.",
            method=paper.title,
            findings=[paper.key_contribution or first_sentence] if (paper.key_contribution or first_sentence) else [],
            evidence=[abstract[:500]] if abstract else [],
        )
