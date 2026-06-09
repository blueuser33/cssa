import hashlib
import time
from typing import Any

from fastapi import WebSocket

from gpt_researcher import GPTResearcher
from gpt_researcher.actions import stream_output
from gpt_researcher.skills.academic_researcher import AcademicResearcher
from gpt_researcher.utils.enum import ReportSource


class AcademicSurveyReport:
    def __init__(
        self,
        query: str,
        tone: Any,
        config_path: str,
        websocket: WebSocket,
        headers=None,
        academic_sources: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        max_papers: int = 30,
        language: str = "zh",
        enable_citation_audit: bool = True,
    ):
        self.query = query
        self.tone = tone
        self.config_path = config_path
        self.websocket = websocket
        self.headers = headers or {}
        self.academic_sources = ["arxiv"]
        self.year_from = year_from
        self.year_to = year_to
        self.max_papers = max_papers
        self.language = language
        self.enable_citation_audit = enable_citation_audit
        self.research_id = self._generate_research_id(query)
        self.result: dict[str, Any] = {}

        self.gpt_researcher = GPTResearcher(
            query=self.query,
            report_type="academic_survey",
            report_source=ReportSource.Web.value,
            tone=self.tone,
            config_path=self.config_path,
            websocket=self.websocket,
            headers=self.headers,
        )

    def _generate_research_id(self, query: str) -> str:
        timestamp = str(int(time.time()))
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return f"academic_{timestamp}_{query_hash}"

    async def run(self) -> str:
        await stream_output(
            "logs",
            "academic_survey",
            "Starting academic survey workflow: retrieving papers, ranking, summarizing, building taxonomy, and auditing citations.",
            self.websocket,
        )
        self.result = await AcademicResearcher(
            self.gpt_researcher,
            academic_sources=self.academic_sources,
            year_from=self.year_from,
            year_to=self.year_to,
            max_papers=self.max_papers,
            language=self.language,
            enable_citation_audit=self.enable_citation_audit,
        ).run()
        self.gpt_researcher.academic_result = self.result
        report = self.result.get("report", "")
        await stream_output("report", "academic_survey", report, self.websocket)
        await stream_output(
            "academic_result",
            "artifacts",
            "Academic survey artifacts are ready.",
            self.websocket,
            metadata={
                "papers": self.result.get("papers", []),
                "paper_summaries": self.result.get("paper_summaries", []),
                "taxonomy": self.result.get("taxonomy", []),
                "citation_audit": self.result.get("citation_audit", {}),
                "bibtex": self.result.get("bibtex", ""),
            },
        )
        await stream_output(
            "logs",
            "academic_survey",
            f"Academic survey complete. Selected {len(self.result.get('papers', []))} papers.",
            self.websocket,
        )
        return report
