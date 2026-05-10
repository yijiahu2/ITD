from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


REQUIRED_FIELDS = [
    "trajectory_id",
    "run_id",
    "image_id",
    "input_snapshot",
    "main_model_stage",
    "main_eval_stage",
    "geometry_review_stage",
    "roi_stage",
    "expert_task_stage",
    "expert_review_stage",
    "fusion_stage",
    "pending_review_candidates",
    "review_status",
]


@dataclass(frozen=True)
class TrajectoryIntegrity:
    trajectory_id: str
    valid: bool
    missing_fields: list[str] = field(default_factory=list)
    missing_artifacts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_trajectory_integrity(
    *,
    trajectory: dict[str, Any],
    artifacts: dict[str, dict[str, Any]] | None = None,
    cfg: dict[str, Any] | None = None,
) -> TrajectoryIntegrity:
    integrity_cfg = cfg or {}
    missing = [field for field in REQUIRED_FIELDS if field not in trajectory]
    warnings: list[str] = []
    if missing and not integrity_cfg.get("reject_trajectory_on_invalid_schema", True):
        warnings.append("schema_missing_fields_allowed_by_config")
    missing_artifacts = [name for name, ref in (artifacts or {}).items() if not ref.get("exists")]
    valid = not missing or not integrity_cfg.get("reject_trajectory_on_invalid_schema", True)
    if integrity_cfg.get("reject_missing_main_eval", True) and "main_eval_stage" in missing:
        valid = False
    return TrajectoryIntegrity(
        trajectory_id=str(trajectory.get("trajectory_id") or "unknown"),
        valid=valid,
        missing_fields=missing,
        missing_artifacts=missing_artifacts,
        warnings=warnings,
    )
