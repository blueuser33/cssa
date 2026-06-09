"""Citation graph expansion inspired by PaSa's paper queue."""

from __future__ import annotations

from .arxiv_source import ArxivPaperSource
from .discovery_graph import PaperDiscoveryGraph
from .models import Paper
from .paper_selector import PaperSelector
from .utils import jaccard_similarity


class CitationExpander:
    """Expand high-scoring arXiv papers through ar5iv section citations."""

    def __init__(
        self,
        selector: PaperSelector,
        arxiv_source: ArxivPaperSource | None = None,
        expand_layers: int = 1,
        expand_papers: int = 10,
        linked_papers_per_seed: int = 8,
        selected_sections_per_seed: int = 4,
        min_selector_score: float = 0.15,
    ):
        self.selector = selector
        self.arxiv_source = arxiv_source or ArxivPaperSource()
        self.expand_layers = expand_layers
        self.expand_papers = expand_papers
        self.linked_papers_per_seed = linked_papers_per_seed
        self.selected_sections_per_seed = selected_sections_per_seed
        self.min_selector_score = min_selector_score

    async def expand(
        self,
        query: str,
        graph: PaperDiscoveryGraph,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[Paper]:
        expanded: list[Paper] = []
        for depth in range(self.expand_layers):
            frontier = graph.expansion_frontier(depth=depth, limit=self.expand_papers)
            if not frontier:
                continue

            layer_candidates: list[Paper] = []
            for parent in frontier:
                layer_candidates.extend(self._linked_candidates(query, parent, year_from, year_to))
            layer_candidates = self._filter_unseen(layer_candidates, graph)

            scored = await self.selector.select(query, layer_candidates)
            kept = [paper for paper in scored if (paper.selector_score or 0.0) >= self.min_selector_score]
            for paper in kept:
                graph.ingest(
                    [paper],
                    source_action="expand",
                    parent_paper_id=paper.parent_paper_id,
                    parent_section=paper.parent_section,
                    depth=paper.depth,
                )
            expanded.extend(kept)
        return expanded

    def _linked_candidates(
        self,
        query: str,
        parent: Paper,
        year_from: int | None,
        year_to: int | None,
    ) -> list[Paper]:
        candidates: list[Paper] = []
        sections = self._ensure_sections(parent)
        for section_name, titles in self._select_sections(query, sections).items():
            for title in titles[: self.linked_papers_per_seed]:
                paper = self.arxiv_source.search_by_title(title)
                if not paper or not self._within_year_range(paper.year, year_from, year_to):
                    continue
                paper.source_action = "expand"
                paper.source_query = parent.source_query
                paper.depth = parent.depth + 1
                paper.parent_paper_id = parent.paper_id
                paper.parent_section = section_name
                paper.raw = {"linked_from": parent.paper_id, "relation": section_name, "title_query": title, **paper.raw}
                candidates.append(paper)
        return candidates

    def _ensure_sections(self, parent: Paper) -> dict[str, list[str]]:
        if parent.sections:
            return parent.sections
        if not parent.arxiv_id:
            return {}
        parent.sections = self.arxiv_source.search_sections_by_arxiv_id(parent.arxiv_id)
        return parent.sections

    def _select_sections(self, query: str, sections: dict[str, list[str]]) -> dict[str, list[str]]:
        if not sections:
            return {}
        scored = []
        for section, titles in sections.items():
            title_text = " ".join(titles[: self.linked_papers_per_seed])
            score = max(jaccard_similarity(query, section), jaccard_similarity(query, title_text))
            scored.append((score, section, titles))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[: self.selected_sections_per_seed]
        if selected and selected[0][0] > 0:
            return {section: titles for _, section, titles in selected}
        return {section: titles for _, section, titles in scored[:1]}

    @staticmethod
    def _filter_unseen(candidates: list[Paper], graph: PaperDiscoveryGraph) -> list[Paper]:
        unseen: list[Paper] = []
        layer_keys: set[str] = set()
        for paper in candidates:
            key = graph._key_for(paper)
            if not key or key in graph.touch_keys or key in layer_keys:
                continue
            layer_keys.add(key)
            unseen.append(paper)
        return unseen

    @staticmethod
    def _within_year_range(year: int | None, year_from: int | None, year_to: int | None) -> bool:
        if year is None:
            return True
        if year_from is not None and year < year_from:
            return False
        if year_to is not None and year > year_to:
            return False
        return True
