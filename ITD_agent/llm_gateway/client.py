from __future__ import annotations

from typing import Any

from openai import OpenAI


def build_openai_compatible_client(cfg: Any) -> OpenAI:
    if not getattr(cfg, "api_key", None):
        raise ValueError("LLM api_key is not configured.")
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
