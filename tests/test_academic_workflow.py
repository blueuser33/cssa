"""Real-configuration integration tests for the academic survey workflow.

These tests intentionally use the project's real `.env` configuration and live
arXiv/LLM calls. They are skipped unless RUN_REAL_LLM_TESTS=1 is set.
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.split("#", 1)[0].strip().strip("\"'")


def parse_llm_env(name: str) -> tuple[str | None, str | None]:
    value = os.getenv(name)
    if not value or ":" not in value:
        return None, None
    provider, model = value.split(":", 1)
    return provider.strip(), model.strip()


load_env_file(Path(__file__).resolve().parents[1] / ".env")

from gpt_researcher.academic.arxiv_source import ArxivPaperSource
from gpt_researcher.academic.bibtex_exporter import BibTeXExporter
from gpt_researcher.academic.citation_expander import CitationExpander
from gpt_researcher.academic.discovery_graph import PaperDiscoveryGraph
from gpt_researcher.academic.paper_deduplicator import PaperDeduplicator
from gpt_researcher.academic.paper_ranker import PaperRanker
from gpt_researcher.academic.paper_reader import PaperReader
from gpt_researcher.academic.paper_retriever import AcademicPaperRetriever
from gpt_researcher.academic.paper_selector import PaperSelector
from gpt_researcher.academic.query_planner import QueryPlanner
from gpt_researcher.academic.survey_writer import SurveyWriter
from gpt_researcher.academic.taxonomy_builder import TaxonomyBuilder
from gpt_researcher.evaluation.citation_verifier import CitationVerifier


fast_env_provider, fast_env_model = parse_llm_env("FAST_LLM")
smart_env_provider, smart_env_model = parse_llm_env("SMART_LLM")


class RealAcademicConfig:
    fast_llm_model = os.getenv("ACADEMIC_TEST_FAST_LLM_MODEL") or os.getenv("ACADEMIC_TEST_LLM_MODEL") or fast_env_model
    fast_llm_provider = os.getenv("ACADEMIC_TEST_FAST_LLM_PROVIDER") or os.getenv("ACADEMIC_TEST_LLM_PROVIDER") or fast_env_provider
    fast_token_limit = int(os.getenv("ACADEMIC_TEST_FAST_TOKEN_LIMIT", "3000"))
    smart_llm_model = os.getenv("ACADEMIC_TEST_SMART_LLM_MODEL") or os.getenv("ACADEMIC_TEST_LLM_MODEL") or smart_env_model
    smart_llm_provider = os.getenv("ACADEMIC_TEST_SMART_LLM_PROVIDER") or os.getenv("ACADEMIC_TEST_LLM_PROVIDER") or smart_env_provider
    smart_token_limit = int(os.getenv("ACADEMIC_TEST_SMART_TOKEN_LIMIT", "6000"))
    temperature = float(os.getenv("ACADEMIC_TEST_TEMPERATURE", "0.1"))
    llm_kwargs = {}


class AcademicCitationParsingTests(unittest.TestCase):
    def test_parse_quoted_reference_metadata(self):
        metadata = ArxivPaperSource._parse_reference_metadata(
            [
                "L. Zhu, T. Chen, D. Ji, J. Ye, and J. Liu, "
                "“Llafs: When large language models meet few-shot segmentation,” "
                "in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), "
                "June 2024, pp. 3065–3075."
            ]
        )

        self.assertEqual(metadata["authors"], "L. Zhu, T. Chen, D. Ji, J. Ye, and J. Liu")
        self.assertEqual(metadata["title"], "Llafs: When large language models meet few-shot segmentation")
        self.assertIn("CVPR", metadata["journal"])

    def test_parse_quoted_reference_keeps_internal_commas_in_title(self):
        """A quoted title may contain commas; the parser must capture it whole
        (up to the closing quote) rather than truncating at the first comma."""
        meta = (
            "X. Pan et al., “Deep learning for drug repurposing: Methods, databases, "
            "and applications,” Wiley reviews, vol. 12, 2022."
        )
        out = ArxivPaperSource._parse_quoted_reference(meta)

        self.assertEqual(out["title"], "Deep learning for drug repurposing: Methods, databases, and applications")
        self.assertEqual(out["journal"], "Wiley reviews, vol. 12, 2022")

    def test_sections_to_reference_titles_handles_spaced_cite_ids(self):
        """`_clean_text` inserts a space after the dot in `~\\cite{bib.bib1}`,
        turning the marker into `~\\cite{bib. bib1}`. The id must still resolve
        against the un-spaced reference keys (`bib.bib1`)."""
        document = {
            "sections": [
                {
                    "title": "Introduction",
                    "id": "S1",
                    "text": "We build on prior work ~\\cite{bib. bib1, bib. bib2} extensively.",
                    "subsections": [],
                }
            ],
            "references": {
                "bib.bib1": {"title": "Attention Is All You Need"},
                "bib.bib2": {"title": "Deep Residual Learning"},
            },
        }

        result = ArxivPaperSource()._sections_to_reference_titles(document)

        self.assertEqual(
            result,
            {"Introduction": ["Attention Is All You Need", "Deep Residual Learning"]},
        )

    def test_normalize_title_hyphen_spacing_is_equivalent(self):
        """arXiv renders hyphens with surrounding spaces ("Property - Aware").
        A hyphenated query title must normalize to the same form so the exact
        title-match in `search_arxiv_id_by_title` succeeds."""
        from gpt_researcher.academic.utils import normalize_title

        query = "Property-aware relation networks for few-shot molecular property prediction"
        rendered = "Property - Aware Relation Networks for Few - Shot Molecular Property Prediction"
        self.assertEqual(normalize_title(query), normalize_title(rendered))
        self.assertEqual(
            normalize_title(query),
            "property aware relation networks for few shot molecular property prediction",
        )


@unittest.skipUnless(os.getenv("RUN_REAL_LLM_TESTS") == "1", "Set RUN_REAL_LLM_TESTS=1 to run real integration tests.")
class AcademicRealWorkflowTests(unittest.TestCase):
    query = os.getenv("ACADEMIC_TEST_QUERY", "graph neural networks for molecular property prediction")
    max_queries = int(os.getenv("ACADEMIC_TEST_MAX_QUERIES", "3"))
    max_papers = int(os.getenv("ACADEMIC_TEST_MAX_PAPERS", "5"))
    selected_papers = int(os.getenv("ACADEMIC_TEST_SELECTED_PAPERS", "3"))
    language = os.getenv("ACADEMIC_TEST_LANGUAGE", "english")

    def setUp(self):
        if not RealAcademicConfig.fast_llm_model or not RealAcademicConfig.fast_llm_provider:
            self.skipTest("Set FAST_LLM or ACADEMIC_TEST_LLM_PROVIDER/ACADEMIC_TEST_LLM_MODEL.")
        if not RealAcademicConfig.smart_llm_model or not RealAcademicConfig.smart_llm_provider:
            self.skipTest("Set SMART_LLM or ACADEMIC_TEST_LLM_PROVIDER/ACADEMIC_TEST_LLM_MODEL.")

    # def test_01_query_planner_real_llm(self):
    #     queries = asyncio.run(QueryPlanner(RealAcademicConfig()).plan(self.query, max_queries=self.max_queries))
    #     print("\n[query_planner]", queries)
    #     self.assertEqual(len(queries), self.max_queries)
    #     self.assertTrue(all(isinstance(query, str) and query.strip() for query in queries))

    # def test_02_arxiv_retriever_real_search(self):
    #     papers = asyncio.run(
    #         AcademicPaperRetriever(RealAcademicConfig(), sources=["arxiv"]).search(
    #             query=self.query,
    #             max_results=self.max_papers,
    #             max_queries=self.max_queries,
    #         )
    #     )

    #     print("\n[retriever]", [(paper.paper_id, paper.title, paper.arxiv_id) for paper in papers])
    #     self.assertGreater(len(papers), 0)
    #     self.assertTrue(all(paper.source == "arxiv" for paper in papers))
    #     self.assertTrue(any(paper.arxiv_id for paper in papers))

    # def test_03_paper_selector_real_llm(self):
    #     papers = self._retrieved_papers()
    #     print("\n[retriever]", [(paper.paper_id, paper.title, paper.arxiv_id) for paper in papers])
    #     scored = asyncio.run(PaperSelector(RealAcademicConfig()).select(self.query, papers[: self.selected_papers]))

    #     print("\n[selector]", [(paper.paper_id, paper.selector_score, paper.selector_reason) for paper in scored])
    #     self.assertEqual(len(scored), min(self.selected_papers, len(papers)))
    #     for paper in scored:
    #         self.assertIsNotNone(paper.selector_score)
    #         self.assertGreaterEqual(paper.selector_score, 0.0)
    #         self.assertLessEqual(paper.selector_score, 1.0)
    #         self.assertTrue(paper.selector_reason)

    # def test_04_discovery_graph_and_arxiv_expander_real(self):
    #     papers = self._selected_papers()
    #     print("\n[selector]", [(paper.paper_id, paper.selector_score, paper.selector_reason) for paper in papers])
    #     graph = PaperDiscoveryGraph()
    #     graph.ingest(papers, source_action="search", source_query=self.query)

    #     expander = CitationExpander(
    #         selector=PaperSelector(RealAcademicConfig()),
    #         arxiv_source=ArxivPaperSource(),
    #         expand_layers=int(os.getenv("ACADEMIC_TEST_EXPAND_LAYERS", "1")),
    #         expand_papers=int(os.getenv("ACADEMIC_TEST_EXPAND_PAPERS", "100")),
    #         linked_papers_per_seed=int(os.getenv("ACADEMIC_TEST_LINKED_PAPERS_PER_SEED", "200")),
    #         selected_sections_per_seed=int(os.getenv("ACADEMIC_TEST_SELECTED_SECTIONS_PER_SEED", "2")),
    #         min_selector_score=float(os.getenv("ACADEMIC_TEST_MIN_SELECTOR_SCORE", "0.0")),
    #     )
    #     expanded = asyncio.run(expander.expand(self.query, graph))

    #     print("\n[expander]", [(paper.paper_id, paper.title, paper.parent_section) for paper in expanded])
    #     print("[graph]", {"touch_count": graph.to_dict()["touch_count"], "node_count": len(graph.nodes)})
    #     self.assertGreaterEqual(len(graph.nodes), len(papers))
    #     self.assertIsInstance(expanded, list)
    #     for paper in expanded:
    #         self.assertEqual(paper.source_action, "expand")
    #         self.assertIsNotNone(paper.parent_paper_id)
    #         self.assertIsNotNone(paper.parent_section)

    # def test_05_ranker_real_candidates(self):
    #     papers = self._selected_papers()
    #     ranked = asyncio.run(PaperRanker().rank(self.query, papers, top_k=self.selected_papers))

    #     print("\n[ranker]", [(paper.paper_id, paper.relevance_score, paper.why_relevant) for paper in ranked])
    #     self.assertGreater(len(ranked), 0)
    #     for paper in ranked:
    #         self.assertIsNotNone(paper.relevance_score)
    #         self.assertTrue(paper.why_relevant)

    # def test_06_paper_reader_real_llm(self):
    #     papers = self._ranked_papers()
    #     summaries = asyncio.run(PaperReader(RealAcademicConfig()).summarize_many(papers))

    #     print("\n[reader]", [summary.to_dict() for summary in summaries])
    #     self.assertEqual(len(summaries), len(papers))
    #     for summary in summaries:
    #         self.assertTrue(summary.paper_id)
    #         self.assertTrue(summary.problem)
    #         self.assertTrue(summary.method)

    # def test_07_taxonomy_builder_real_llm(self):
    #     papers = self._ranked_papers()
    #     summaries = asyncio.run(PaperReader(RealAcademicConfig()).summarize_many(papers))
    #     taxonomy = asyncio.run(TaxonomyBuilder(RealAcademicConfig()).build(self.query, papers, summaries))

    #     print("\n[taxonomy]", [category.to_dict() for category in taxonomy])
    #     self.assertGreater(len(taxonomy), 0)
    #     self.assertTrue(all(category.name and category.paper_ids for category in taxonomy))

    # def test_08_survey_writer_real_llm(self):
    #     papers, summaries, taxonomy = self._survey_inputs()
    #     report = asyncio.run(
    #         SurveyWriter(RealAcademicConfig()).write(
    #             query=self.query,
    #             papers=papers,
    #             summaries=summaries,
    #             taxonomy=taxonomy,
    #             language=self.language,
    #         )
    #     )

    #     print("\n[survey]\n", report[:1200])
    #     self.assertIn("[", report)
    #     self.assertTrue(any(f"[{paper.paper_id}]" in report for paper in papers))

    # def test_09_citation_verifier_real_llm(self):
    #     papers, summaries, taxonomy = self._survey_inputs()
    #     report = asyncio.run(
    #         SurveyWriter(RealAcademicConfig()).write(
    #             query=self.query,
    #             papers=papers,
    #             summaries=summaries,
    #             taxonomy=taxonomy,
    #             language=self.language,
    #         )
    #     )
    #     audit = asyncio.run(CitationVerifier(RealAcademicConfig()).verify(report, papers, summaries))

    #     print("\n[citation_verifier]", audit)
    #     self.assertIn("checked", audit)
    #     self.assertIn("items", audit)

    # def test_10_bibtex_exporter_real_papers(self):
    #     papers = self._ranked_papers()
    #     bibtex = BibTeXExporter().export(papers)

    #     print("\n[bibtex]\n", bibtex)
    #     self.assertIn("@article", bibtex)
    #     self.assertTrue(any(paper.paper_id in bibtex for paper in papers))

    def _retrieved_papers(self):
        papers = asyncio.run(
            AcademicPaperRetriever(RealAcademicConfig(), sources=["arxiv"]).search(
                query=self.query,
                max_results=self.max_papers,
                max_queries=self.max_queries,
            )
        )
        deduped = PaperDeduplicator().deduplicate(papers)
        if not deduped:
            self.fail("No arXiv papers were retrieved. Try another ACADEMIC_TEST_QUERY or increase ACADEMIC_TEST_MAX_PAPERS.")
        return deduped

    def _selected_papers(self):
        papers = self._retrieved_papers()
        scored = asyncio.run(PaperSelector(RealAcademicConfig()).select(self.query, papers))
        return scored[: self.selected_papers]

    def _ranked_papers(self):
        selected = self._selected_papers()
        ranked = asyncio.run(PaperRanker().rank(self.query, selected, top_k=self.selected_papers))
        if not ranked:
            self.fail("No papers were ranked.")
        return ranked

    def _survey_inputs(self):
        papers = self._ranked_papers()
        summaries = asyncio.run(PaperReader(RealAcademicConfig()).summarize_many(papers))
        taxonomy = asyncio.run(TaxonomyBuilder(RealAcademicConfig()).build(self.query, papers, summaries))
        return papers, summaries, taxonomy


if __name__ == "__main__":
    unittest.main()
    # arxiv_source = ArxivPaperSource()
    # print(arxiv_source.search_by_title('Property-aware relation networks for few-shot molecular property prediction'))
