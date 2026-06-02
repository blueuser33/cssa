"""Paper deduplication and metadata merging."""

from __future__ import annotations

from .models import Paper
from .utils import normalize_title


class PaperDeduplicator:
    def deduplicate(self, papers: list[Paper]) -> list[Paper]:
        by_key: dict[str, Paper] = {}
        order: list[str] = []

        for paper in papers:
            key = self._key_for(paper)
            if not key:
                continue
            if key not in by_key:
                by_key[key] = paper
                order.append(key)
                continue
            by_key[key] = self._merge(by_key[key], paper)

        return [by_key[key] for key in order]

    def _key_for(self, paper: Paper) -> str:
        for prefix, value in [
            ("doi", paper.doi),
            ("arxiv", paper.arxiv_id),
            ("s2", paper.semantic_scholar_id),
        ]:
            if value:
                return f"{prefix}:{value.lower()}"
        return f"title:{normalize_title(paper.title)}"

    def _merge(self, left: Paper, right: Paper) -> Paper:
        primary, secondary = self._choose_primary(left, right), self._choose_secondary(left, right)
        for field_name in [
            "title",
            "authors",
            "year",
            "venue",
            "abstract",
            "url",
            "pdf_url",
            "doi",
            "arxiv_id",
            "semantic_scholar_id",
            "citation_count",
        ]:
            if not getattr(primary, field_name) and getattr(secondary, field_name):
                setattr(primary, field_name, getattr(secondary, field_name))

        sources = {source for source in [primary.source, secondary.source] if source}
        primary.source = ",".join(sorted(sources)) if sources else primary.source
        primary.raw = {"merged": [left.raw, right.raw]}
        primary.paper_id = primary.paper_id or secondary.paper_id
        return primary

    @staticmethod
    def _choose_primary(left: Paper, right: Paper) -> Paper:
        left_citations = left.citation_count or 0
        right_citations = right.citation_count or 0
        if right_citations > left_citations:
            return right
        return left

    @staticmethod
    def _choose_secondary(left: Paper, right: Paper) -> Paper:
        return left if PaperDeduplicator._choose_primary(left, right) is right else right
