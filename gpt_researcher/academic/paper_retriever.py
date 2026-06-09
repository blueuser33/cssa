"""Academic paper retrieval from arXiv."""

from __future__ import annotations

import asyncio
import logging

from .arxiv_source import ArxivPaperSource
from .query_planner import QueryPlanner
from .models import Paper

logger = logging.getLogger(__name__)


class AcademicPaperRetriever:
    def __init__(self, cfg, sources: list[str] | None = None):
        self.cfg = cfg
        self.sources = ["arxiv"]
        self.query_planner = QueryPlanner(cfg)
        self.arxiv_source = ArxivPaperSource()

    async def search(
        self,
        query: str,
        year_from: int | None = None,
        year_to: int | None = None,
        max_results: int = 30,
        max_queries: int = 5,
    ) -> list[Paper]:
        queries = await self.query_planner.plan(query, max_queries=max_queries)
        per_query = max(1, max_results // max(1, len(queries)))
        tasks = []
        for planned_query in queries:
            tasks.append(asyncio.to_thread(self._search_arxiv, planned_query, year_from, year_to, per_query))

        papers: list[Paper] = []
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, Exception):
                logger.warning("Academic source search failed: %s", result)
                continue
            papers.extend(result)
        return papers

    def _search_arxiv(
        self,
        query: str,
        year_from: int | None,
        year_to: int | None,
        max_results: int,
    ) -> list[Paper]:
        return self.arxiv_source.search_by_query(query, year_from, year_to, max_results)
