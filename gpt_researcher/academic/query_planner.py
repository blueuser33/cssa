"""PaSa-style academic search query planning."""

from __future__ import annotations

import logging
import re

from .utils import safe_json_loads

logger = logging.getLogger(__name__)


class QueryPlanner:
    """Generate mutually useful paper-search queries from a user topic.

    Inspired by PaSa's crawler agent, this component expands a scholarly need
    into several distinct searches before retrieval. It is intentionally light:
    it uses the configured LLM when available and deterministic fallbacks when
    dependencies or API keys are missing.
    """

    def __init__(self, cfg=None, cost_callback=None):
        self.cfg = cfg
        self.cost_callback = cost_callback

    async def plan(self, query: str, max_queries: int = 5) -> list[str]:
        if max_queries <= 1:
            return [query]

        planned = await self._llm_plan(query, max_queries)
        if not planned:
            planned = self._fallback_plan(query, max_queries)
        return self._dedupe(planned, max_queries)

    async def _llm_plan(self, query: str, max_queries: int) -> list[str]:
        if not self.cfg:
            return []
        prompt = f"""Generate {max_queries} mutually distinct academic paper search queries.

User query: {query}

Guidelines:
- Prefer queries that retrieve survey, benchmark, taxonomy, and representative method papers.
- Keep each query concise.
- Return ONLY JSON: {{"queries": ["..."]}}
"""
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
            queries = data.get("queries", data) if isinstance(data, dict) else data
            return [str(item).strip() for item in queries if str(item).strip()]
        except Exception as exc:
            logger.warning("Academic query planning fell back: %s", exc)
            return []

    @staticmethod
    def _fallback_plan(query: str, max_queries: int) -> list[str]:
        templates = [
            "{query}",
            "{query} survey",
            "{query} benchmark evaluation",
            "{query} methods taxonomy",
            "{query} recent advances",
            "{query} representative papers",
        ]
        return [template.format(query=query) for template in templates[:max_queries]]

    @staticmethod
    def _dedupe(queries: list[str], max_queries: int) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for query in queries:
            normalized = re.sub(r"\s+", " ", query.lower()).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(query)
            if len(unique) >= max_queries:
                break
        return unique
