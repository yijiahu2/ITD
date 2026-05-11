from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExpertRoutingPlan:
    recommended_action: str = "use_rule_guard"
    preferred_expert_family: str | None = None
    preferred_expert_model: str | None = None
    reason: str = "No validated LLM recommendation was available."
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_PLAN_FIELDS = {
    "recommended_action",
    "preferred_expert_family",
    "reason",
    "confidence",
}


def normalize_structured_plan(payload: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(payload or {})
    confidence = data.get("confidence", 0.0)
    try:
        normalized_confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        normalized_confidence = 0.0
    plan = ExpertRoutingPlan(
        recommended_action=str(data.get("recommended_action") or data.get("action") or "use_rule_guard"),
        preferred_expert_family=data.get("preferred_expert_family"),
        preferred_expert_model=data.get("preferred_expert_model") or data.get("preferred_expert_model"),
        reason=str(data.get("reason") or data.get("selection_reason") or "No reason provided."),
        confidence=normalized_confidence,
        metadata={key: value for key, value in data.items() if key not in REQUIRED_PLAN_FIELDS},
    )
    return plan.to_dict()
