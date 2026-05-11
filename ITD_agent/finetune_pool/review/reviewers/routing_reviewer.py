from __future__ import annotations

from typing import Any

from ITD_agent.finetune_pool.review.review_context_builder import ReviewContext
from ITD_agent.finetune_pool.review.review_policy import APPROVE, ReviewDecision

from .base_reviewer import BaseReviewer


class RoutingReviewer(BaseReviewer):
    candidate_type = "routing_candidate"

    def review(self, candidate: dict[str, Any], context: ReviewContext, cfg: dict[str, Any]) -> ReviewDecision:
        decision = str(candidate.get("expert_decision") or "unknown")
        improvement = candidate.get("improvement") or {}
        score = 0.85 if decision in {"accept", "partial_accept"} else 0.65
        return ReviewDecision(
            candidate_id=str(candidate.get("candidate_id")),
            candidate_type=self.candidate_type,
            trajectory_id=context.trajectory_id,
            decision=APPROVE,
            reason="routing_evidence_recorded_without_policy_update",
            evidence_refs={"trajectory_id": context.trajectory_id},
            target_asset_type="routing_candidate",
            quality_score=score,
            safe_to_write=True,
            payload={
                "source_run_id": context.source_run_id,
                "source_trajectory_id": context.trajectory_id,
                "level1_error_type": candidate.get("level1_error_type"),
                "failure_family": candidate.get("failure_family"),
                "expert_model": candidate.get("expert_model"),
                "expert_decision": decision,
                "improvement_summary": improvement,
                "safety_summary": candidate.get("safety") or {},
                "recommendation": "evidence_only_do_not_update_route_map",
                "status": "evidence_only" if decision == "reject" else "draft_skill_candidate",
            },
        )
