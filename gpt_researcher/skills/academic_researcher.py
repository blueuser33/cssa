"""High-level orchestration for academic survey generation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from gpt_researcher.academic import (
    AcademicPaperRetriever,
    CitationExpander,
    PaperDeduplicator,
    PaperDiscoveryGraph,
    PaperRanker,
    PaperReader,
    PaperSelector,
    SurveyWriter,
    TaxonomyBuilder,
)
from gpt_researcher.academic.bibtex_exporter import BibTeXExporter
from gpt_researcher.evaluation import CitationVerifier


class AcademicResearcher:
    def __init__(
        self,
        researcher,
        academic_sources: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        max_papers: int = 30,
        language: str = "zh",
        style: str = "mini_survey",
        enable_citation_audit: bool = True,
        citation_expand_layers: int = 1,
        expand_papers: int = 10,
    ):
        self.researcher = researcher
        self.academic_sources = ["arxiv"]
        self.year_from = year_from
        self.year_to = year_to
        self.max_papers = max_papers
        self.language = language
        self.style = style
        self.enable_citation_audit = enable_citation_audit
        self.citation_expand_layers = citation_expand_layers
        self.expand_papers = expand_papers

    async def run(self) -> dict[str, Any]:
        cfg = self.researcher.cfg
        query = self.researcher.query
        cost_callback = getattr(self.researcher, "add_costs", None)

        retriever = AcademicPaperRetriever(cfg, self.academic_sources)
        deduplicator = PaperDeduplicator()
        discovery_graph = PaperDiscoveryGraph()
        selector = PaperSelector(cfg, cost_callback)
        citation_expander = CitationExpander(
            selector=selector,
            expand_layers=self.citation_expand_layers,
            expand_papers=self.expand_papers,
        )
        ranker = PaperRanker()
        reader = PaperReader(cfg, getattr(self.researcher, "prompt_family", None), cost_callback)
        taxonomy_builder = TaxonomyBuilder(cfg, getattr(self.researcher, "prompt_family", None), cost_callback)
        survey_writer = SurveyWriter(cfg, getattr(self.researcher, "prompt_family", None), cost_callback)
        citation_verifier = CitationVerifier(cfg, cost_callback)

        retrieved_papers = await retriever.search(
            query=query,
            year_from=self.year_from,
            year_to=self.year_to,
            max_results=self.max_papers,
        )
        discovery_graph.ingest(retrieved_papers, source_action="search")
        deduped_papers = deduplicator.deduplicate(discovery_graph.papers())
        scored_papers = await selector.select(query, deduped_papers)
        discovery_graph.ingest(scored_papers)
        expanded_papers = await citation_expander.expand(
            query=query,
            graph=discovery_graph,
            year_from=self.year_from,
            year_to=self.year_to,
        )
        candidate_papers = deduplicator.deduplicate(discovery_graph.papers())
        selected_papers = await ranker.rank(query, candidate_papers, top_k=self.max_papers)
        summaries = await reader.summarize_many(selected_papers)
        taxonomy = await taxonomy_builder.build(query, selected_papers, summaries)
        report = await survey_writer.write(
            query=query,
            papers=selected_papers,
            summaries=summaries,
            taxonomy=taxonomy,
            language=self.language,
            style=self.style,
        )
        citation_audit = (
            await citation_verifier.verify(report, selected_papers, summaries)
            if self.enable_citation_audit
            else {}
        )
        bibtex = BibTeXExporter().export(selected_papers)

        return {
            "report": report,
            "retrieved_papers": self._serialize(retrieved_papers),
            "expanded_papers": self._serialize(expanded_papers),
            "discovery_graph": discovery_graph.to_dict(),
            "papers": self._serialize(selected_papers),
            "paper_summaries": self._serialize(summaries),
            "taxonomy": self._serialize(taxonomy),
            "citation_audit": citation_audit,
            "bibtex": bibtex,
        }

    @staticmethod
    def _serialize(value):
        if isinstance(value, list):
            return [AcademicResearcher._serialize(item) for item in value]
        if is_dataclass(value):
            return asdict(value)
        return value
