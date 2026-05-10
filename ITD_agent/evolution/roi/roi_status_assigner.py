from __future__ import annotations

from dataclasses import replace
from typing import Any

from .roi_candidate_builder import ROICandidate


def assign_roi_status(roi_candidates: list[ROICandidate], roi_policy: dict[str, Any] | None = None) -> list[ROICandidate]:
    policy = roi_policy or {}
    min_failure_instances = int((policy.get("min_trigger_per_tile") or {}).get("min_failure_instances", 3))
    severe_threshold = float((policy.get("severe_failure_override") or {}).get("min_severity_score", 0.75))
    trigger_all = len(roi_candidates) >= min_failure_instances
    assigned: list[ROICandidate] = []
    for roi in roi_candidates:
        actionable = trigger_all or roi.severity_score >= severe_threshold
        if roi.severity_score < 0.35 and not actionable:
            status = "record_only"
        elif actionable:
            status = "actionable"
        else:
            status = "monitor"
        assigned.append(
            replace(
                roi,
                review_status=status,
                expert_eligible=actionable,
                training_eligible=bool(roi.affected_gt_ids),
            )
        )
    return assigned


def is_global_failure(roi_candidates: list[ROICandidate], roi_policy: dict[str, Any] | None = None) -> bool:
    guard = (roi_policy or {}).get("global_failure_guard") or {}
    threshold = guard.get("tiny_roi_count_global_threshold")
    if threshold is not None and len(roi_candidates) >= int(threshold):
        return True
    return False
