from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


APPROVE = "approve"
REJECT = "reject"
DEFER = "defer"
NEED_HUMAN_REVIEW = "need_human_review"


@dataclass(frozen=True)
class ReviewDecision:
    candidate_id: str
    candidate_type: str
    trajectory_id: str
    decision: str
    reason: str
    evidence_refs: dict[str, Any] = field(default_factory=dict)
    target_asset_type: str = ""
    quality_score: float | None = None
    safe_to_write: bool = False
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rule_decision(
    *,
    candidate_id: str,
    candidate_type: str,
    trajectory_id: str,
    target_asset_type: str,
    quality_score: float | None,
    min_quality_score: float,
    evidence_refs: dict[str, Any] | None = None,
    reason: str = "",
    payload: dict[str, Any] | None = None,
) -> ReviewDecision:
    score = quality_score if quality_score is not None else 0.0
    approved = score >= min_quality_score
    return ReviewDecision(
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        trajectory_id=trajectory_id,
        decision=APPROVE if approved else REJECT,
        reason=reason or ("quality_score_passed" if approved else "quality_score_below_threshold"),
        evidence_refs=evidence_refs or {},
        target_asset_type=target_asset_type,
        quality_score=quality_score,
        safe_to_write=approved,
        payload=payload or {},
    )
