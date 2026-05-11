from __future__ import annotations

from collections import Counter
from typing import Any

from ITD_agent.finetune_pool.review.review_context_builder import ReviewContext
from ITD_agent.finetune_pool.review.review_policy import APPROVE, ReviewDecision

from .base_reviewer import BaseReviewer


class MemoryReviewer(BaseReviewer):
    candidate_type = "memory_candidate"

    def synthesize_candidates(self, context: ReviewContext) -> list[dict[str, Any]]:
        summary = context.trajectory_summary
        final_source = (summary.get("fusion_summary") or {}).get("final_result_source")
        review_counts = (summary.get("expert_review_summary") or {}).get("by_decision") or {}
        roi_by_type = (summary.get("roi_summary") or {}).get("by_error_type") or {}
        candidates: list[dict[str, Any]] = []
        if roi_by_type:
            top_error, top_count = Counter(roi_by_type).most_common(1)[0]
            candidates.append(
                {
                    "candidate_id": f"mem_failure_{context.trajectory_id}",
                    "memory_type": "failure_pattern_memory",
                    "level1_error_type": top_error,
                    "failure_family": _family_for_error(top_error),
                    "quality_score": min(1.0, float(top_count) / 50.0),
                    "summary": f"Main model produced recurring {top_error} ROIs on image {context.image_id}.",
                }
            )
        if review_counts.get("accept", 0) or review_counts.get("partial_accept", 0):
            candidates.append(
                {
                    "candidate_id": f"mem_expert_success_{context.trajectory_id}",
                    "memory_type": "expert_success_memory",
                    "level1_error_type": "false_positive",
                    "failure_family": "false_positive_cleanup",
                    "quality_score": 0.9,
                    "summary": f"At least one expert result was accepted for trajectory {context.trajectory_id}.",
                }
            )
        if final_source == "rollback_to_main" or review_counts.get("reject", 0):
            candidates.append(
                {
                    "candidate_id": f"mem_rollback_{context.trajectory_id}",
                    "memory_type": "rollback_memory",
                    "level1_error_type": "multi_error",
                    "failure_family": "expert_guardrail",
                    "quality_score": 0.75,
                    "summary": f"Expert result was rejected or rolled back for trajectory {context.trajectory_id}.",
                }
            )
        return candidates

    def review(self, candidate: dict[str, Any], context: ReviewContext, cfg: dict[str, Any]) -> ReviewDecision:
        min_score = float((cfg.get("memory_review") or {}).get("min_quality_score", 0.6))
        score = float(candidate.get("quality_score") or 0.0)
        approved = score >= min_score
        return ReviewDecision(
            candidate_id=str(candidate.get("candidate_id")),
            candidate_type=self.candidate_type,
            trajectory_id=context.trajectory_id,
            decision=APPROVE if approved else "reject",
            reason="memory_candidate_has_structured_gt_evidence" if approved else "memory_candidate_quality_below_threshold",
            evidence_refs={"trajectory_id": context.trajectory_id, "artifact_refs": context.artifact_refs},
            target_asset_type="memory",
            quality_score=score,
            safe_to_write=approved,
            payload={
                "source_run_id": context.source_run_id,
                "source_trajectory_id": context.trajectory_id,
                "source_roi_ids": [],
                "memory_type": candidate.get("memory_type"),
                "level1_error_type": candidate.get("level1_error_type"),
                "failure_family": candidate.get("failure_family"),
                "summary": candidate.get("summary"),
                "metrics_snapshot": context.trajectory_summary.get("main_eval") or {},
                "artifact_refs": context.artifact_refs,
                "confidence": "high" if score >= 0.8 else "medium",
                "status": "active",
            },
        )


def _family_for_error(error_type: str) -> str:
    return {
        "false_negative": "small_crown_recall",
        "false_positive": "false_positive_cleanup",
        "under_segmentation": "crown_split",
        "over_segmentation": "crown_merge_cleanup",
    }.get(error_type, "boundary_refinement")
