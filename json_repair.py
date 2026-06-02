"""Small compatibility shim for the optional json_repair package.

The full third-party package is more capable. This local fallback covers the
common GPT Researcher use case: parsing JSON that may be wrapped in markdown or
surrounded by explanatory text.
"""

from __future__ import annotations

import json
import re
from typing import Any


def loads(value: str, *args, **kwargs) -> Any:
    text = value.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text, *args, **kwargs)
    except json.JSONDecodeError:
        start = min([idx for idx in [text.find("{"), text.find("[")] if idx >= 0], default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1], *args, **kwargs)
        raise


def repair_json(value: str, *args, **kwargs) -> str:
    return json.dumps(loads(value), ensure_ascii=False)
