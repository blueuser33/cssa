"""Academic survey report writer."""

from __future__ import annotations

import logging

from .models import Paper, PaperSummary, TaxonomyCategory

logger = logging.getLogger(__name__)


class SurveyWriter:
    def __init__(self, cfg, prompt_family=None, cost_callback=None):
        self.cfg = cfg
        self.prompt_family = prompt_family
        self.cost_callback = cost_callback

    async def write(
        self,
        query: str,
        papers: list[Paper],
        summaries: list[PaperSummary],
        taxonomy: list[TaxonomyCategory],
        language: str = "zh",
        style: str = "mini_survey",
    ) -> str:
        prompt = self._prompt(query, papers, summaries, taxonomy, language, style)
        try:
            from gpt_researcher.utils.llm import create_chat_completion

            return await create_chat_completion(
                model=self.cfg.smart_llm_model,
                messages=[{"role": "user", "content": prompt}],
                llm_provider=self.cfg.smart_llm_provider,
                max_tokens=getattr(self.cfg, "smart_token_limit", 6000),
                temperature=getattr(self.cfg, "temperature", 0.35),
                llm_kwargs=getattr(self.cfg, "llm_kwargs", {}),
                cost_callback=self.cost_callback,
            )
        except Exception as exc:
            logger.warning("Survey writing fell back: %s", exc)
            return self._fallback_report(query, papers, summaries, taxonomy, language)

    def _prompt(
        self,
        query: str,
        papers: list[Paper],
        summaries: list[PaperSummary],
        taxonomy: list[TaxonomyCategory],
        language: str,
        style: str,
    ) -> str:
        return f"""You are writing an evidence-grounded academic {style}.

Research topic: {query}
Output language: {language}

Requirements:
- Write in Markdown.
- Use stable citation IDs exactly like [PaperID].
- Do not make claims without citing at least one paper ID.
- If evidence is limited, explicitly state the uncertainty.
- Include these sections: Title, Overview, Paper Selection, Method Taxonomy, Representative Works, Comparison Table, Open Challenges, Future Directions, References.

Paper table:
{self._paper_table(papers)}

Structured summaries:
{[summary.to_dict() for summary in summaries]}

Taxonomy:
{[category.to_dict() for category in taxonomy]}
"""

    @staticmethod
    def _paper_table(papers: list[Paper]) -> str:
        rows = [
            "| Title | Year | Venue | Authors | Why Relevant | Key Contribution | Citation Count |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for paper in papers:
            rows.append(
                "| {title} | {year} | {venue} | {authors} | {why} | {contribution} | {citations} |".format(
                    title=paper.title.replace("|", " "),
                    year=paper.year or "",
                    venue=(paper.venue or "").replace("|", " "),
                    authors=", ".join(paper.authors[:3]).replace("|", " "),
                    why=(paper.why_relevant or "").replace("|", " "),
                    contribution=(paper.key_contribution or "").replace("|", " "),
                    citations=paper.citation_count if paper.citation_count is not None else "",
                )
            )
        return "\n".join(rows)

    def _fallback_report(
        self,
        query: str,
        papers: list[Paper],
        summaries: list[PaperSummary],
        taxonomy: list[TaxonomyCategory],
        language: str,
    ) -> str:
        summary_by_id = {summary.paper_id: summary for summary in summaries}
        title = f"# Academic Survey: {query}"
        if language.lower().startswith("zh"):
            title = f"# 学术综述：{query}"

        parts = [
            title,
            "## Overview",
            "This draft is generated from retrieved academic metadata and abstracts. Claims are limited to the available evidence.",
            "## Paper Selection",
            self._paper_table(papers),
            "## Method Taxonomy",
        ]
        for category in taxonomy:
            parts.append(f"### {category.name}")
            parts.append(category.description)
            parts.append("Representative papers: " + ", ".join(f"[{pid}]" for pid in category.paper_ids))
            if category.strengths:
                parts.append("Strengths: " + "; ".join(category.strengths))
            if category.limitations:
                parts.append("Limitations: " + "; ".join(category.limitations))

        parts.extend(["## Representative Works"])
        for paper in papers:
            summary = summary_by_id.get(paper.paper_id)
            finding = "; ".join(summary.findings[:2]) if summary else paper.key_contribution
            parts.append(f"- {paper.title} [{paper.paper_id}]: {finding or 'Evidence is limited to metadata.'}")

        parts.extend(
            [
                "## Open Challenges",
                "- Evidence coverage depends on abstracts and available metadata.",
                "- Direct head-to-head comparisons may be missing across datasets and evaluation settings.",
                "## Future Directions",
                "- Extend retrieval to OpenAlex/Crossref and full-text PDF reading.",
                "- Add stronger citation graph and contradiction analysis.",
                "## References",
            ]
        )
        for paper in papers:
            parts.append(f"- [{paper.paper_id}] {paper.title}. {paper.venue or ''}, {paper.year or ''}. {paper.url or paper.pdf_url or ''}")
        return "\n\n".join(parts)
