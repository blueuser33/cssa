"""Academic paper retrieval from arXiv and Semantic Scholar."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

import requests

from .models import Paper
from .utils import stable_paper_id

logger = logging.getLogger(__name__)


class AcademicPaperRetriever:
    def __init__(self, cfg, sources: list[str] | None = None):
        self.cfg = cfg
        self.sources = sources or ["arxiv", "semantic_scholar"]

    async def search(
        self,
        query: str,
        year_from: int | None = None,
        year_to: int | None = None,
        max_results: int = 30,
    ) -> list[Paper]:
        per_source = max(1, max_results)
        tasks = []
        if "arxiv" in self.sources:
            tasks.append(asyncio.to_thread(self._search_arxiv, query, year_from, year_to, per_source))
        if "semantic_scholar" in self.sources or "semanticscholar" in self.sources:
            tasks.append(asyncio.to_thread(self._search_semantic_scholar, query, year_from, year_to, per_source))

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
        try:
            import arxiv
        except Exception as exc:
            logger.warning("arxiv package is unavailable: %s", exc)
            return []

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: list[Paper] = []
        for result in client.results(search):
            year = result.published.year if result.published else None
            if not self._within_year_range(year, year_from, year_to):
                continue
            authors = [author.name for author in result.authors]
            arxiv_id = result.entry_id.rstrip("/").split("/")[-1]
            paper = Paper(
                paper_id=stable_paper_id(authors, year, result.title),
                title=result.title,
                authors=authors,
                year=year,
                venue="arXiv",
                abstract=result.summary,
                url=result.entry_id,
                pdf_url=result.pdf_url,
                arxiv_id=arxiv_id,
                source="arxiv",
                raw={
                    "published": result.published.isoformat() if result.published else None,
                    "categories": result.categories,
                },
            )
            papers.append(paper)
        return papers

    def _search_semantic_scholar(
        self,
        query: str,
        year_from: int | None,
        year_to: int | None,
        max_results: int,
    ) -> list[Paper]:
        params: dict[str, Any] = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": ",".join(
                [
                    "paperId",
                    "title",
                    "abstract",
                    "url",
                    "venue",
                    "year",
                    "authors",
                    "citationCount",
                    "externalIds",
                    "openAccessPdf",
                    "publicationDate",
                ]
            ),
        }
        if year_from or year_to:
            start = year_from or 1900
            end = year_to or datetime.now().year
            params["year"] = f"{start}-{end}"

        try:
            response = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
                timeout=20,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Semantic Scholar search failed: %s", exc)
            return []

        papers: list[Paper] = []
        for item in response.json().get("data", []):
            year = item.get("year")
            if not self._within_year_range(year, year_from, year_to):
                continue
            authors = [author.get("name", "") for author in item.get("authors", []) if author.get("name")]
            external_ids = item.get("externalIds") or {}
            pdf_url = (item.get("openAccessPdf") or {}).get("url")
            title = item.get("title") or "Untitled"
            paper = Paper(
                paper_id=stable_paper_id(authors, year, title),
                title=title,
                authors=authors,
                year=year,
                venue=item.get("venue"),
                abstract=item.get("abstract"),
                url=item.get("url"),
                pdf_url=pdf_url,
                doi=external_ids.get("DOI"),
                arxiv_id=external_ids.get("ArXiv"),
                semantic_scholar_id=item.get("paperId"),
                citation_count=item.get("citationCount"),
                source="semantic_scholar",
                raw=item,
            )
            papers.append(paper)
        return papers

    @staticmethod
    def _within_year_range(year: int | None, year_from: int | None, year_to: int | None) -> bool:
        if year is None:
            return True
        if year_from is not None and year < year_from:
            return False
        if year_to is not None and year > year_to:
            return False
        return True
