from __future__ import annotations

from typing import Any

from ITD_agent.evolution.review.review_context_builder import ReviewContext
from ITD_agent.evolution.review.review_policy import APPROVE, ReviewDecision

from .base_reviewer import BaseReviewer


class DistillationReviewer(BaseReviewer):
    candidate_type = "distillation_candidate"

    def review(self, candidate: dict[str, Any], context: ReviewContext, cfg: dict[str, Any]) -> ReviewDecision:
        roi_id = str(candidate.get("roi_id") or "")
        roi = context.roi_by_id.get(roi_id) or {}
        score = float(roi.get("severity_score") or 0.0)
        approved = bool(roi_id and roi) and score >= float((cfg.get("distillation_review") or {}).get("min_quality_score", 0.7))
        return ReviewDecision(
            candidate_id=str(candidate.get("candidate_id") or f"distill_{context.trajectory_id}_{roi_id}"),
            candidate_type=self.candidate_type,
            trajectory_id=context.trajectory_id,
            decision=APPROVE if approved else "reject",
            reason="expert_success_roi_marked_for_future_distillation" if approved else "distillation_candidate_missing_accepted_roi_evidence",
            evidence_refs={"trajectory_id": context.trajectory_id, "roi_id": roi_id},
            target_asset_type="distillation_candidate",
            quality_score=score,
            safe_to_write=approved,
            payload={
                "source_run_id": context.source_run_id,
                "source_trajectory_id": context.trajectory_id,
                "source_roi_id": roi_id,
                "expert_model": candidate.get("expert_model") or "unknown",
                "quality_tier": "gold" if score >= 0.9 else "silver",
                "evidence_refs": {"roi": roi, "artifact_refs": context.artifact_refs},
                "status": "candidate_only",
            },
        )
