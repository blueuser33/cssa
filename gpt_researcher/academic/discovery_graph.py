"""PaSa-style discovery graph for searched and expanded papers."""

from __future__ import annotations

from .models import Paper, PaperDiscoveryNode
from .paper_deduplicator import PaperDeduplicator
from .utils import normalize_title


class PaperDiscoveryGraph:
    """Track paper provenance as a lightweight search/expansion graph."""

    def __init__(self):
        self.nodes: dict[str, PaperDiscoveryNode] = {}
        self.paper_key_to_id: dict[str, str] = {}
        self.touch_keys: set[str] = set()
        self._deduplicator = PaperDeduplicator()

    def ingest(
        self,
        papers: list[Paper],
        source_action: str | None = None,
        source_query: str | None = None,
        parent_paper_id: str | None = None,
        parent_section: str | None = None,
        depth: int | None = None,
    ) -> list[Paper]:
        ingested: list[Paper] = []
        for paper in papers:
            if source_action is not None:
                paper.source_action = source_action
            if source_query is not None and not paper.source_query:
                paper.source_query = source_query
            if parent_paper_id is not None:
                paper.parent_paper_id = parent_paper_id
            if parent_section is not None:
                paper.parent_section = parent_section
            if depth is not None:
                paper.depth = depth

            key = self._key_for(paper)
            if not key:
                continue
            self.touch_keys.add(key)

            existing_id = self.paper_key_to_id.get(key)
            if existing_id and existing_id in self.nodes:
                existing = self.nodes[existing_id].paper
                self.nodes[existing_id].paper = self._deduplicator._merge(existing, paper)
                graph_paper = self.nodes[existing_id].paper
            else:
                graph_paper = paper
                self.nodes[paper.paper_id] = PaperDiscoveryNode(paper=paper)
                self.paper_key_to_id[key] = paper.paper_id

            if parent_paper_id and parent_paper_id in self.nodes:
                relation = parent_section or source_action or "related"
                self.nodes[parent_paper_id].add_child(relation, graph_paper.paper_id)

            ingested.append(graph_paper)
        return ingested

    def papers(self) -> list[Paper]:
        return [node.paper for node in self.nodes.values()]

    def to_dict(self) -> dict:
        return {
            "touch_count": len(self.touch_keys),
            "nodes": {paper_id: node.to_dict() for paper_id, node in self.nodes.items()},
        }

    def expansion_frontier(self, depth: int, limit: int) -> list[Paper]:
        candidates = [
            node.paper
            for node in self.nodes.values()
            if node.paper.depth == depth and (node.paper.arxiv_id or node.paper.sections or node.paper.references or node.paper.citations)
        ]
        candidates.sort(key=lambda paper: paper.selector_score or paper.relevance_score or 0.0, reverse=True)
        return candidates[:limit]

    @staticmethod
    def _key_for(paper: Paper) -> str:
        for prefix, value in [
            ("doi", paper.doi),
            ("arxiv", paper.arxiv_id),
        ]:
            if value:
                return f"{prefix}:{value.lower()}"
        return f"title:{normalize_title(paper.title)}"
