"""Offline smoke script for the academic survey components.

This script does not call live academic APIs or LLM providers. It exercises the
same downstream pipeline used by the new academic survey feature with a small
in-memory paper set.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpt_researcher.academic.bibtex_exporter import BibTeXExporter
from gpt_researcher.academic.models import Paper
from gpt_researcher.academic.paper_deduplicator import PaperDeduplicator
from gpt_researcher.academic.paper_ranker import PaperRanker
from gpt_researcher.academic.paper_reader import PaperReader
from gpt_researcher.academic.paper_selector import PaperSelector
from gpt_researcher.academic.survey_writer import SurveyWriter
from gpt_researcher.academic.taxonomy_builder import TaxonomyBuilder
from gpt_researcher.evaluation.citation_verifier import CitationVerifier


class FakeConfig:
    fast_llm_model = "fake-fast"
    fast_llm_provider = "fake"
    fast_token_limit = 1000
    smart_llm_model = "fake-smart"
    smart_llm_provider = "fake"
    smart_token_limit = 2000
    temperature = 0.1
    llm_kwargs = {}


async def main() -> None:
    query = "RAG query rewriting methods"
    papers = [
        Paper(
            paper_id="Lewis2020RAG",
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            authors=["Patrick Lewis"],
            year=2020,
            venue="NeurIPS",
            abstract="Retrieval-augmented generation combines parametric models with non-parametric memory for knowledge-intensive NLP tasks.",
            citation_count=1000,
            doi="10.1000/rag",
            arxiv_id="2005.11401",
            source="arxiv",
        ),
        Paper(
            paper_id="Ma2023QueryRewrite",
            title="Query Rewriting for Retrieval-Augmented Large Language Models",
            authors=["Xueguang Ma"],
            year=2023,
            venue="arXiv",
            abstract="Query rewriting improves retrieval quality by reformulating ambiguous user queries for retrieval-augmented generation.",
            citation_count=120,
            source="arxiv",
        ),
    ]

    cfg = FakeConfig()
    deduped = PaperDeduplicator().deduplicate(papers)
    scored = await PaperSelector().select(query, deduped)
    selected = await PaperRanker().rank(query, scored, top_k=10)
    summaries = await PaperReader(cfg).summarize_many(selected)
    taxonomy = await TaxonomyBuilder(cfg).build(query, selected, summaries)
    report = await SurveyWriter(cfg).write(query, selected, summaries, taxonomy, language="english")
    audit = await CitationVerifier(cfg).verify(report, selected, summaries)
    bibtex = BibTeXExporter().export(selected)

    print(json.dumps({
        "papers": [paper.to_dict() for paper in selected],
        "taxonomy": [category.to_dict() for category in taxonomy],
        "citation_audit": audit,
        "bibtex": bibtex,
        "report_preview": report[:800],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
