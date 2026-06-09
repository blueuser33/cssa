"""Shared helpers for academic survey modules."""

from __future__ import annotations

import json
import re
import string
from collections.abc import Iterable
from typing import Any


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    text = title.lower()
    # Replace punctuation with spaces (not delete) so hyphenated titles match
    # regardless of surrounding spaces: both "property-aware" and arXiv's
    # rendered "property - aware" normalize to "property aware".
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,}", text.lower())
        if token not in STOP_WORDS
    }


def jaccard_similarity(left: str | None, right: str | None) -> float:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def safe_json_loads(text: str) -> Any:
    """Parse JSON, accepting responses wrapped in markdown fences."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min([idx for idx in [text.find("{"), text.find("[")] if idx >= 0], default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def stable_paper_id(authors: Iterable[str], year: int | None, title: str) -> str:
    first_author = "Paper"
    author_list = list(authors)
    if author_list:
        first_author = re.sub(r"[^A-Za-z0-9]", "", author_list[0].split()[-1]) or "Paper"
    title_words = [word.capitalize() for word in re.findall(r"[A-Za-z0-9]+", title) if word.lower() not in STOP_WORDS]
    suffix = "".join(title_words[:3]) or "Untitled"
    return f"{first_author}{year or 'ND'}{suffix}"


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "using",
    "via",
    "towards",
    "toward",
    "survey",
    "paper",
}
