from __future__ import annotations

from typing import Any

from ITD_agent.finetune_pool.review.review_context_builder import ReviewContext
from ITD_agent.finetune_pool.review.review_policy import APPROVE, ReviewDecision

from .base_reviewer import BaseReviewer


class FinetuneReviewer(BaseReviewer):
    candidate_type = "finetune_candidate"

    def review_many(self, candidates: list[dict[str, Any]], context: ReviewContext, cfg: dict[str, Any]) -> list[ReviewDecision]:
        selected = _select_candidates(candidates, context, cfg)
        return [self.review(candidate, context, cfg) for candidate in selected]

    def review(self, candidate: dict[str, Any], context: ReviewContext, cfg: dict[str, Any]) -> ReviewDecision:
        roi = context.roi_by_id.get(str(candidate.get("roi_id"))) or {}
        score = float(roi.get("severity_score") or 0.0)
        min_score = float((cfg.get("finetune_pool") or {}).get("min_quality_score", 0.6))
        approved = score >= min_score and bool(roi)
        error_type = str(candidate.get("failure_category") or roi.get("level1_error_type") or "unknown")
        return ReviewDecision(
            candidate_id=str(candidate.get("candidate_id")),
            candidate_type=self.candidate_type,
            trajectory_id=context.trajectory_id,
            decision=APPROVE if approved else "reject",
            reason="roi_has_gt_backed_failure_and_quality_score_passed" if approved else "roi_missing_or_quality_below_threshold",
            evidence_refs={"trajectory_id": context.trajectory_id, "roi_id": candidate.get("roi_id"), "artifact_refs": context.artifact_refs},
            target_asset_type="finetune_sample",
            quality_score=score,
            safe_to_write=approved,
            payload={
                "source_run_id": context.source_run_id,
                "source_trajectory_id": context.trajectory_id,
                "source_roi_id": candidate.get("roi_id"),
                "image_id": context.image_id,
                "sample_type": _sample_type_for_error(error_type),
                "target_model_role": candidate.get("target_model_role") or "main_model",
                "target_error_type": error_type,
                "quality_score": score,
                "review_status": "approved" if approved else "rejected",
                "roi": roi,
                "artifact_refs": context.artifact_refs,
            },
        )


def _sample_type_for_error(error_type: str) -> str:
    if error_type == "false_positive":
        return "hard_negative_sample"
    if error_type in {"under_segmentation", "over_segmentation"}:
        return "boundary_refine_sample"
    return "main_failure_sample"


def _select_candidates(candidates: list[dict[str, Any]], context: ReviewContext, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    pool_cfg = cfg.get("finetune_pool") or {}
    per_error = int(pool_cfg.get("max_samples_per_error_type_per_trajectory", pool_cfg.get("max_samples_per_trajectory", 8)))
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    ranked = sorted(
        candidates,
        key=lambda item: float((context.roi_by_id.get(str(item.get("roi_id"))) or {}).get("severity_score") or 0.0),
        reverse=True,
    )
    for item in ranked:
        roi = context.roi_by_id.get(str(item.get("roi_id"))) or {}
        error_type = str(item.get("failure_category") or roi.get("level1_error_type") or "unknown")
        if counts.get(error_type, 0) >= per_error:
            continue
        counts[error_type] = counts.get(error_type, 0) + 1
        selected.append(item)
    return selected
