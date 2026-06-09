"""Build method taxonomy for selected papers."""

from __future__ import annotations

import logging
from collections import defaultdict

from .models import Paper, PaperSummary, TaxonomyCategory
from .utils import safe_json_loads, tokenize

logger = logging.getLogger(__name__)


class TaxonomyBuilder:
    def __init__(self, cfg, prompt_family=None, cost_callback=None):
        self.cfg = cfg
        self.prompt_family = prompt_family
        self.cost_callback = cost_callback

    async def build(
        self,
        query: str,
        papers: list[Paper],
        summaries: list[PaperSummary],
    ) -> list[TaxonomyCategory]:
        if not papers:
            return []
        prompt = self._prompt(query, papers, summaries)
        try:
            from gpt_researcher.utils.llm import create_chat_completion

            response = await create_chat_completion(
                model=self.cfg.smart_llm_model,
                messages=[{"role": "user", "content": prompt}],
                llm_provider=self.cfg.smart_llm_provider,
                max_tokens=min(getattr(self.cfg, "smart_token_limit", 6000), 4000),
                temperature=getattr(self.cfg, "temperature", 0.3),
                llm_kwargs=getattr(self.cfg, "llm_kwargs", {}),
                cost_callback=self.cost_callback,
            )
            data = safe_json_loads(response)
            categories = data.get("categories", data) if isinstance(data, dict) else data
            parsed = [
                TaxonomyCategory(
                    name=str(item.get("name") or "Other"),
                    description=str(item.get("description") or ""),
                    paper_ids=[str(pid) for pid in item.get("paper_ids", [])],
                    strengths=[str(x) for x in item.get("strengths", [])],
                    limitations=[str(x) for x in item.get("limitations", [])],
                    applicable_scenarios=[str(x) for x in item.get("applicable_scenarios", [])],
                )
                for item in categories
                if isinstance(item, dict) and item.get("paper_ids")
            ]
            return parsed or self._fallback_taxonomy(query, papers, summaries)
        except Exception as exc:
            logger.warning("Taxonomy generation fell back: %s", exc)
            return self._fallback_taxonomy(query, papers, summaries)

    @staticmethod
    def _prompt(query: str, papers: list[Paper], summaries: list[PaperSummary]) -> str:
        summary_by_id = {summary.paper_id: summary for summary in summaries}
        rows = []
        for paper in papers:
            summary = summary_by_id.get(paper.paper_id)
            rows.append(
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "year": paper.year,
                    "method": summary.method if summary else "",
                    "findings": summary.findings if summary else [],
                }
            )
        return f"""You are creating a method taxonomy for an academic survey.

Research topic: {query}

Papers and summaries:
{rows}

Create 3 to 8 specific academic method categories. Assign every paper to at least one category when possible.

Return ONLY valid JSON:
{{
  "categories": [
    {{
      "name": "category name",
      "description": "definition",
      "paper_ids": ["PaperID"],
      "strengths": ["strength"],
      "limitations": ["limitation"],
      "applicable_scenarios": ["scenario"]
    }}
  ]
}}
"""

    @staticmethod
    def _fallback_taxonomy(
        query: str,
        papers: list[Paper],
        summaries: list[PaperSummary],
    ) -> list[TaxonomyCategory]:
        keyword_categories = {
            "Evaluation and Benchmarking": {"benchmark", "evaluation", "dataset", "metric", "test"},
            "Retrieval and Ranking": {"retrieval", "retriever", "rank", "ranking", "rerank", "search"},
            "Generation and Reasoning": {"generation", "generate", "reasoning", "llm", "language"},
            "Optimization and Training": {"training", "fine", "tuning", "optimization", "learning"},
            "Survey and Analysis": {"survey", "review", "analysis", "taxonomy"},
        }
        summary_by_id = {summary.paper_id: summary for summary in summaries}
        buckets: dict[str, list[str]] = defaultdict(list)

        for paper in papers:
            summary = summary_by_id.get(paper.paper_id)
            text = f"{paper.title} {paper.abstract or ''} {summary.method if summary else ''}"
            tokens = tokenize(text)
            assigned = False
            for name, keywords in keyword_categories.items():
                if tokens & keywords:
                    buckets[name].append(paper.paper_id)
                    assigned = True
                    break
            if not assigned:
                buckets["General Methods"].append(paper.paper_id)

        return [
            TaxonomyCategory(
                name=name,
                description=f"Papers related to {name.lower()} for {query}.",
                paper_ids=paper_ids,
                strengths=["Provides a coherent group of related approaches."],
                limitations=["Category inferred from title and abstract metadata."],
            )
            for name, paper_ids in buckets.items()
            if paper_ids
        ]
