"""arXiv-only paper source and ar5iv section parser.

Adapted from the PaSa reference implementation's arXiv/ar5iv utilities
(Apache-2.0, Copyright 2024 Bytedance Ltd. and/or its affiliates).
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
from typing import Any

import requests

from .models import Paper
from .utils import normalize_title, stable_paper_id

logger = logging.getLogger(__name__)


class ArxivPaperSource:
    """Search arXiv and parse ar5iv sections for citation expansion."""

    citation_pattern = re.compile(r"~\\cite\{(.*?)\}", flags=re.DOTALL)

    def __init__(self, delay_seconds: float = 0.05, request_timeout: int = 30, request_retries: int = 3):
        self.delay_seconds = delay_seconds
        self.request_timeout = request_timeout
        self.request_retries = request_retries

    def search_by_query(
        self,
        query: str,
        year_from: int | None = None,
        year_to: int | None = None,
        max_results: int = 10,
    ) -> list[Paper]:
        try:
            import arxiv
        except Exception as exc:
            logger.warning("arxiv package is unavailable: %s", exc)
            return []

        client = arxiv.Client(delay_seconds=self.delay_seconds)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        papers: list[Paper] = []
        for result in client.results(search):
            paper = self._paper_from_arxiv_result(result, source_query=query, source_action="search")
            if paper and self._within_year_range(paper.year, year_from, year_to):
                papers.append(paper)
        return papers

    def search_by_arxiv_id(self, arxiv_id: str) -> Paper | None:
        try:
            import arxiv
        except Exception as exc:
            logger.warning("arxiv package is unavailable: %s", exc)
            return None

        clean_id = arxiv_id.split("v")[0]
        client = arxiv.Client(delay_seconds=self.delay_seconds)
        search = arxiv.Search(
            query="",
            id_list=[clean_id],
            max_results=10,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        try:
            results = list(client.results(search))
        except Exception as exc:
            logger.warning("Failed to search arXiv id %s: %s", clean_id, exc)
            return None

        for result in results:
            result_id = result.entry_id.rstrip("/").split("/")[-1].split("v")[0]
            if result_id == clean_id:
                return self._paper_from_arxiv_result(result, source_action="expand")
        return None

    def search_by_title(self, title: str) -> Paper | None:
        arxiv_id = self.search_arxiv_id_by_title(title)
        if not arxiv_id:
            return None
        return self.search_by_arxiv_id(arxiv_id)

    def search_arxiv_id_by_title(self, title: str) -> str | None:
        url = "https://arxiv.org/search/?" + urllib.parse.urlencode(
            {
                "query": title,
                "searchtype": "title",
                "abstracts": "hide",
                "size": 200,
            }
        )
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("arXiv title search failed for %s: %s", title, exc)
            return None

        soup = self._soup(response.text, "html.parser")
        if soup is None:
            return None

        results: list[tuple[str, str]] = []
        for item in soup.find_all("li", class_="arxiv-result"):
            title_tag = item.find("p", class_="title")
            id_tag = item.find("p", class_="list-title")
            id_link = id_tag.find("a") if id_tag else None
            if title_tag and id_link:
                result_title = title_tag.get_text(" ", strip=True)
                result_id = id_link.get_text(strip=True).replace("arXiv:", "").split("v")[0]
                results.append((result_title, result_id))

        if not results and soup.title and soup.title.string:
            match = re.match(r"\[(.*?)\]\s*(.*)", soup.title.string)
            if match:
                results.append((match.group(2), match.group(1).split("v")[0]))

        wanted = normalize_title(title)
        for result_title, result_id in results:
            if normalize_title(result_title) == wanted:
                return result_id
        return None

    def search_sections_by_arxiv_id(self, arxiv_id: str) -> dict[str, list[str]]:
        clean_id = arxiv_id.split("v")[0]
        if not re.match(r"^\d{4}\.\d+$", clean_id):
            return {}

        url = f"https://ar5iv.labs.arxiv.org/html/{clean_id}"
        response = self._get_with_retries(url, timeout=self.request_timeout)
        if response is None:
            logger.warning("ar5iv fetch failed for %s after %s attempts", clean_id, self.request_retries)
            return {}

        if "https://ar5iv.labs.arxiv.org/html" not in response.text:
            logger.warning("Invalid ar5iv HTML document for %s", clean_id)
            return {}

        try:
            document = self._parse_ar5iv_html(response.text)
        except Exception as exc:
            logger.warning("ar5iv parse failed for %s: %s", clean_id, exc)
            return {}

        return self._sections_to_reference_titles(document)

    def _get_with_retries(self, url: str, timeout: int) -> requests.Response | None:
        last_exc: requests.RequestException | None = None
        for attempt in range(1, self.request_retries + 1):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("Request failed (%s/%s) for %s: %s", attempt, self.request_retries, url, exc)
                if attempt < self.request_retries:
                    time.sleep(min(2 ** (attempt - 1), 5))
        if last_exc:
            logger.warning("Request gave up for %s: %s", url, last_exc)
        return None

    def _paper_from_arxiv_result(
        self,
        result: Any,
        source_query: str | None = None,
        source_action: str = "search",
    ) -> Paper:
        year = result.published.year if result.published else None
        authors = [author.name for author in result.authors]
        arxiv_id = result.entry_id.rstrip("/").split("/")[-1]
        return Paper(
            paper_id=stable_paper_id(authors, year, result.title),
            title=result.title.replace("\n", " "),
            authors=authors,
            year=year,
            venue="arXiv",
            abstract=result.summary.replace("\n", " "),
            url=result.entry_id,
            pdf_url=result.pdf_url,
            arxiv_id=arxiv_id,
            source="arxiv",
            source_query=source_query,
            source_action=source_action,
            raw={
                "published": result.published.isoformat() if result.published else None,
                "categories": result.categories,
            },
        )

    def _parse_ar5iv_html(self, html: str) -> dict[str, Any]:
        soup = self._soup(html, "lxml")
        if soup is None:
            return {"sections": [], "references": {}}
        references = self._parse_references(soup)
        toc = self._generate_toc(soup)
        sections = self._extract_section_text(toc, soup)
        return {"sections": sections, "references": references}

    def _parse_references(self, soup) -> dict[str, dict[str, str]]:
        bibliography = soup.find(class_="ltx_biblist")
        references: dict[str, dict[str, str]] = {}
        if bibliography is None:
            return references
        for item in bibliography.find_all("li", recursive=False):
            ref_id = item.get("id")
            if not ref_id:
                continue
            blocks = [block.get_text(" ", strip=True) for block in item.find_all("span", class_="ltx_bibblock")]
            metadata = self._parse_reference_metadata(blocks)
            references[ref_id] = metadata
        return references

    @staticmethod
    def _parse_reference_metadata(blocks: list[str]) -> dict[str, str]:
        clean_blocks = [block.replace("\n", " ") for block in blocks]
        meta_string = " ".join(clean_blocks)
        meta_string = re.sub(r"\s+", " ", meta_string.replace("\xa0", " ")).strip()
        quoted = ArxivPaperSource._parse_quoted_reference(meta_string)
        if quoted:
            return quoted

        if len(clean_blocks) >= 3:
            return {
                "authors": re.sub(r"\s+", " ", clean_blocks[0].replace("\xa0", " ")).strip(),
                "title": ArxivPaperSource._clean_reference_title(clean_blocks[1]),
                "journal": re.sub(r"\s+", " ", " ".join(clean_blocks[2:]).replace("\xa0", " ")).strip(),
                "meta_string": meta_string,
            }

        meta_string = re.sub(r"\.\s\d{4}[a-z]?\.", ".", meta_string)
        match = re.match(r"^(.*?\.\s)(.*?)(\.\s.*|$)", meta_string, re.DOTALL)
        if not match:
            return {"authors": "", "title": meta_string, "journal": "", "meta_string": meta_string}
        journal = match.group(3).strip()
        if journal.startswith(". "):
            journal = journal[2:]
        return {
            "authors": match.group(1).strip(),
            "title": ArxivPaperSource._clean_reference_title(match.group(2)),
            "journal": journal,
            "meta_string": meta_string,
        }

    @staticmethod
    def _parse_quoted_reference(meta_string: str) -> dict[str, str] | None:
        # Terminate the title on the closing quote only — not on a comma — so
        # titles with internal commas ("Methods, databases, and applications")
        # are captured whole. The trailing comma before the quote, if any, is
        # removed by _clean_reference_title.
        match = re.search(r"[“\"]([^”\"]{4,}?)[”\"]", meta_string)
        if not match:
            match = re.search(r"[‘']([^’']{4,}?)[’']", meta_string)
        if not match:
            return None

        authors = meta_string[: match.start()].strip(" ,.;")
        title = ArxivPaperSource._clean_reference_title(match.group(1))
        journal = meta_string[match.end() :].strip(" ,.;")
        if not title:
            return None
        return {
            "authors": authors,
            "title": title,
            "journal": journal,
            "meta_string": meta_string,
        }

    @staticmethod
    def _clean_reference_title(title: str) -> str:
        title = re.sub(r"\s+", " ", title.replace("\xa0", " ")).strip()
        return title.strip(" ,.;:，。、“”\"'‘’")

    def _generate_toc(self, soup) -> list[dict[str, Any]]:
        toc: list[dict[str, Any]] = []
        stack: list[tuple[int, list[dict[str, Any]]]] = [(0, toc)]
        heading_tags = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5}
        for tag in soup.find_all(heading_tags.keys()):
            level = heading_tags[tag.name]
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent = stack[-1][1]
            section = tag.find_parent("section", id=True)
            entry = {
                "title": tag.get_text(" ", strip=True),
                "id": section.get("id") if section else None,
                "subsections": [],
            }
            parent.append(entry)
            stack.append((level, entry["subsections"]))
        return toc

    def _extract_section_text(self, entries: list[dict[str, Any]], soup) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for entry in entries:
            title = entry.get("title", "")
            if self._is_stop_section(title):
                continue
            section_id = entry.get("id")
            if section_id:
                section = soup.find(id=section_id)
                if section is not None:
                    text_parts: list[str] = []
                    self._parse_text(text_parts, section)
                    if text_parts:
                        entry["text"] = self._clean_text("".join(text_parts))
            entry["subsections"] = self._extract_section_text(entry.get("subsections", []), soup)
            filtered.append(entry)
        return filtered

    def _sections_to_reference_titles(self, document: dict[str, Any]) -> dict[str, list[str]]:
        second_level_sections = self._get_second_level_sections(document.get("sections", []))
        sections_to_titles: dict[str, list[str]] = {}
        for section, text in second_level_sections.items():
            titles: set[str] = set()
            for citation_group in self.citation_pattern.findall(text):
                for citation_id in ["".join(item.split()) for item in citation_group.split(",")]:
                    reference = document.get("references", {}).get(citation_id)
                    title = (reference or {}).get("title")
                    if title:
                        titles.add(title)
            if titles:
                sections_to_titles[" ".join(section.split())] = sorted(titles)
        return sections_to_titles

    def _parse_text(self, output: list[str], tag) -> None:
        try:
            import bs4
        except Exception:
            return

        ignore_tags = {"a", "figure", "center", "caption", "td", "h1", "h2", "h3", "h4", "sup"}
        for child in tag.children:
            if isinstance(child, bs4.element.NavigableString):
                output.append(child.get_text())
            elif isinstance(child, bs4.element.Comment):
                continue
            elif isinstance(child, bs4.element.Tag):
                if child.name in ignore_tags or self._has_navigation_class(child):
                    continue
                if child.name == "cite":
                    hrefs = [link.get("href", "").strip("#") for link in child.find_all("a", class_="ltx_ref")]
                    output.append("~\\cite{" + ", ".join(hrefs) + "}")
                    continue
                if child.name == "img" and child.has_attr("alt"):
                    output.append(child.get("alt", ""))
                    continue
                if child.name == "section":
                    return
                self._parse_text(output, child)

    @staticmethod
    def _clean_text(text: str) -> str:
        for item in ["=-1", "\t", "\xa0", "[]", "()"]:
            text = text.replace(item, "")
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\.(?!\d)", ". ", text)
        return text

    @staticmethod
    def _get_subsections(sections: list[dict[str, Any]]) -> dict[str, str]:
        result: dict[str, str] = {}
        for section in sections:
            if section.get("text", "").strip():
                result[section["title"].strip()] = section["text"].strip()
            result.update(ArxivPaperSource._get_subsections(section.get("subsections", [])))
        return result

    @staticmethod
    def _get_first_level_sections(sections: list[dict[str, Any]]) -> dict[str, str]:
        result: dict[str, str] = {}
        for section in sections:
            subsections = ArxivPaperSource._get_subsections(section.get("subsections", []))
            text = section.get("text", "").strip()
            if text or subsections:
                result[section["title"].strip()] = text
                for value in subsections.values():
                    result[section["title"].strip()] += value.strip()
        return {" ".join(key.split()): value for key, value in result.items() if "appendix" not in key.lower()}

    @staticmethod
    def _get_second_level_sections(sections: list[dict[str, Any]]) -> dict[str, str]:
        result: dict[str, str] = {}
        for section in sections:
            if section.get("text", "").strip():
                result[section["title"].strip()] = section["text"].strip()
            for key, value in ArxivPaperSource._get_first_level_sections(section.get("subsections", [])).items():
                result[section["title"].strip() + " " + key.strip()] = value.strip()
        return {" ".join(key.split()): value for key, value in result.items() if "appendix" not in key.lower()}

    @staticmethod
    def _has_navigation_class(tag) -> bool:
        return tag.has_attr("class") and bool(tag["class"]) and tag["class"][0] == "navigation"

    @staticmethod
    def _is_stop_section(title: str) -> bool:
        stop_words = ["references", "acknowledgments", "about this document", "appendix"]
        return any(word in title.lower() for word in stop_words)

    @staticmethod
    def _within_year_range(year: int | None, year_from: int | None, year_to: int | None) -> bool:
        if year is None:
            return True
        if year_from is not None and year < year_from:
            return False
        if year_to is not None and year > year_to:
            return False
        return True

    @staticmethod
    def _soup(html: str, parser: str):
        try:
            import bs4
        except Exception as exc:
            logger.warning("BeautifulSoup is unavailable: %s", exc)
            return None
        return bs4.BeautifulSoup(html, parser)
