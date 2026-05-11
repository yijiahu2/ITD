from __future__ import annotations

from typing import Any

from ITD_agent.llm_gateway.schemas import normalize_structured_plan


def fallback_structured_plan(*, task_type: str, error: str | None = None, context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}
    preferred_family = context.get("preferred_expert_family") or context.get("failure_family")
    plan = normalize_structured_plan(
        {
            "recommended_action": "use_rule_guard",
            "preferred_expert_family": preferred_family,
            "preferred_expert_model": context.get("preferred_expert_model"),
            "reason": error or f"{task_type} used deterministic fallback.",
            "confidence": 0.0,
            "fallback_context": context,
        }
    )
    plan["error"] = error
    return plan
