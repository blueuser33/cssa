"""BibTeX export utilities for academic papers."""

from __future__ import annotations

import re

from .models import Paper


class BibTeXExporter:
    def export(self, papers: list[Paper]) -> str:
        return "\n\n".join(self._entry(paper) for paper in papers)

    def _entry(self, paper: Paper) -> str:
        fields = {
            "title": paper.title,
            "author": " and ".join(paper.authors),
            "year": str(paper.year) if paper.year else "",
            "venue": paper.venue or "",
            "doi": paper.doi or "",
            "url": paper.url or paper.pdf_url or "",
        }
        body = ",\n".join(
            f"  {key} = {{{self._escape(value)}}}"
            for key, value in fields.items()
            if value
        )
        return f"@article{{{paper.paper_id},\n{body}\n}}"

    @staticmethod
    def _escape(value: str) -> str:
        return re.sub(r"([{}])", r"\\\1", value)
