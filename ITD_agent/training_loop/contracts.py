from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainingCandidate:
    candidate_id: str
    trajectory_id: str
    roi_id: str
    sample_type: str
    target_model_role: str
    failure_category: str
    quality_status: str = "pending_review"
    approved: bool = False
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    trigger_training: bool = False
    reason: str = "V1 only supports dry-run training candidate intake."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
