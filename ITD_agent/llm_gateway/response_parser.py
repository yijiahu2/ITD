from __future__ import annotations

import json
from typing import Any

from ITD_agent.llm_gateway.schemas import normalize_structured_plan


def strip_json_fence(content: str) -> str:
    text = (content or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    stripped = "\n".join(lines).strip()
    if stripped.startswith("json"):
        stripped = stripped[4:].strip()
    return stripped


def parse_json_response(content: str) -> dict[str, Any]:
    parsed = json.loads(strip_json_fence(content))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object.")
    return parsed


def parse_structured_plan(content: str) -> dict[str, Any]:
    return normalize_structured_plan(parse_json_response(content))
