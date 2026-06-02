"""Basic citation support verification for academic surveys."""

from __future__ import annotations

import logging
import re

from gpt_researcher.academic.models import CitationAuditItem, Paper, PaperSummary
from gpt_researcher.utils.llm import create_chat_completion

logger = logging.getLogger(__name__)


class CitationVerifier:
    STATUSES = {
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
        "not_enough_evidence",
    }

    def __init__(self, cfg, cost_callback=None):
        self.cfg = cfg
        self.cost_callback = cost_callback

    async def verify(
        self,
        report: str,
        papers: list[Paper],
        summaries: list[PaperSummary],
    ) -> dict:
        paper_by_id = {paper.paper_id: paper for paper in papers}
        summary_by_id = {summary.paper_id: summary for summary in summaries}
        claims = self._extract_claims(report)
        items: list[CitationAuditItem] = []

        for claim, citation in claims:
            paper = paper_by_id.get(citation)
            summary = summary_by_id.get(citation)
            if not paper:
                items.append(
                    CitationAuditItem(
                        claim=claim,
                        citation=citation,
                        status="not_enough_evidence",
                        support_score=0.0,
                        evidence="Citation ID was not found in selected papers.",
                        reason="Missing paper metadata.",
                    )
                )
                continue
            items.append(await self._verify_one(claim, citation, paper, summary))

        return self._summary(items)

    async def _verify_one(
        self,
        claim: str,
        citation: str,
        paper: Paper,
        summary: PaperSummary | None,
    ) -> CitationAuditItem:
        evidence = "\n".join(
            [
                f"Title: {paper.title}",
                f"Abstract: {paper.abstract or ''}",
                f"Problem: {summary.problem if summary else ''}",
                f"Method: {summary.method if summary else ''}",
                f"Findings: {'; '.join(summary.findings) if summary else ''}",
                f"Evidence: {'; '.join(summary.evidence) if summary else ''}",
            ]
        )
        prompt = f"""Assess whether the cited paper evidence supports the claim.

Return ONLY JSON:
{{
  "status": "supported|partially_supported|unsupported|contradicted|not_enough_evidence",
  "support_score": 0.0,
  "evidence": "short supporting or missing evidence",
  "reason": "brief reason"
}}

Claim: {claim}
Citation: [{citation}]
Paper evidence:
{evidence}
"""
        try:
            from gpt_researcher.academic.utils import safe_json_loads

            response = await create_chat_completion(
                model=self.cfg.fast_llm_model,
                messages=[{"role": "user", "content": prompt}],
                llm_provider=self.cfg.fast_llm_provider,
                max_tokens=1000,
                temperature=0.0,
                llm_kwargs=getattr(self.cfg, "llm_kwargs", {}),
                cost_callback=self.cost_callback,
            )
            data = safe_json_loads(response)
            status = str(data.get("status") or "not_enough_evidence")
            if status not in self.STATUSES:
                status = "not_enough_evidence"
            return CitationAuditItem(
                claim=claim,
                citation=citation,
                status=status,
                support_score=float(data.get("support_score") or 0.0),
                evidence=str(data.get("evidence") or ""),
                reason=str(data.get("reason") or ""),
            )
        except Exception as exc:
            logger.warning("Citation verifier fell back for [%s]: %s", citation, exc)
            return self._fallback_verify(claim, citation, paper, summary)

    @staticmethod
    def _extract_claims(report: str) -> list[tuple[str, str]]:
        claims: list[tuple[str, str]] = []
        for sentence in re.split(r"(?<=[.!?。！？])\s+", report):
            for citation in re.findall(r"\[([A-Za-z][A-Za-z0-9_-]{2,})\]", sentence):
                claims.append((sentence.strip(), citation))
        return claims

    @staticmethod
    def _fallback_verify(
        claim: str,
        citation: str,
        paper: Paper,
        summary: PaperSummary | None,
    ) -> CitationAuditItem:
        evidence_text = " ".join(
            [
                paper.title or "",
                paper.abstract or "",
                summary.problem if summary else "",
                summary.method if summary else "",
                " ".join(summary.findings) if summary else "",
            ]
        ).lower()
        claim_terms = {term for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", claim.lower())}
        evidence_terms = {term for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", evidence_text)}
        overlap = len(claim_terms & evidence_terms)
        score = overlap / max(1, min(len(claim_terms), 12))
        status = "supported" if score >= 0.45 else "partially_supported" if score >= 0.2 else "not_enough_evidence"
        return CitationAuditItem(
            claim=claim,
            citation=citation,
            status=status,
            support_score=round(min(score, 1.0), 3),
            evidence=(paper.abstract or paper.title or "")[:400],
            reason="Heuristic lexical overlap check; use LLM verification for stronger audit.",
        )

    @staticmethod
    def _summary(items: list[CitationAuditItem]) -> dict:
        supported = sum(1 for item in items if item.status == "supported")
        partially_supported = sum(1 for item in items if item.status == "partially_supported")
        unsupported = sum(1 for item in items if item.status in {"unsupported", "contradicted"})
        checked = len(items)
        precision = (supported + 0.5 * partially_supported) / checked if checked else 0.0
        return {
            "checked": checked,
            "supported": supported,
            "partially_supported": partially_supported,
            "unsupported": unsupported,
            "citation_precision": round(precision, 3),
            "items": [item.to_dict() for item in items],
        }
