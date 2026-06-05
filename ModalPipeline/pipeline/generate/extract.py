from __future__ import annotations

import json
import re
from typing import Any


def extract_python_code(text: str) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    s = text.strip()

    m = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", s, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n", errors

    s = re.sub(r"^```(?:python|py)?\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    s = s.strip()
    if s:
        return s + "\n", errors

    errors.append("missing_python_fence")
    return None, errors


def parse_json_response(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    s = text.strip()

    fence = re.search(r"```(?:json)?\s*\n(.*?)\n```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()

    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return data, errors
        errors.append("json_not_object")
        return None, errors
    except json.JSONDecodeError as e:
        errors.append(f"invalid_json:{e}")
        return None, errors
