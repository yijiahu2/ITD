from __future__ import annotations

from collections import Counter
from typing import Any


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(item.get(key) or "unknown") for item in items))


def _candidate_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(items),
        "by_failure_category": _count_by(items, "failure_category"),
        "by_target_model_role": _count_by(items, "target_model_role"),
        "by_quality_status": _count_by(items, "quality_status"),
        "by_sample_type": _count_by(items, "sample_type"),
        "sample_candidate_refs": [
            {
                "candidate_id": item.get("candidate_id") or f"candidate_{item.get('trajectory_id')}_{item.get('roi_id')}",
                "roi_id": item.get("roi_id"),
                "failure_category": item.get("failure_category"),
                "target_model_role": item.get("target_model_role"),
            }
            for item in items[:16]
        ],
    }


def _pending_candidate_summary(pending: dict[str, Any]) -> dict[str, Any]:
    memory = list(pending.get("memory_candidates") or [])
    skill = list(pending.get("skill_candidates") or [])
    training = list(pending.get("training_candidates") or [])
    distillation = list(pending.get("distillation_candidates") or [])
    return {
        "counts": {
            "memory": len(memory),
            "skill": len(skill),
            "training": len(training),
            "distillation": len(distillation),
        },
        "memory": _candidate_summary(memory),
        "skill": _candidate_summary(skill),
        "training": _candidate_summary(training),
        "distillation": {
            "count": len(distillation),
            "sample_candidate_refs": [
                {
                    "candidate_id": item.get("candidate_id") or f"distill_{item.get('trajectory_id')}_{item.get('roi_id')}",
                    "roi_id": item.get("roi_id"),
                    "expert_model": item.get("expert_model"),
                }
                for item in distillation[:16]
            ],
        },
        "full_pending_candidates_ref": {
            "source": "v1_trajectory_artifact",
            "json_pointer": "/pending_review_candidates",
        },
    }


def compress_trajectory_for_review(trajectory: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rois = list((trajectory.get("roi_stage") or {}).get("roi_candidates") or [])
    clusters = list((trajectory.get("roi_stage") or {}).get("roi_clusters") or [])
    tasks = list((trajectory.get("expert_task_stage") or {}).get("expert_tasks") or [])
    reviews = list((trajectory.get("expert_review_stage") or {}).get("expert_reviews") or [])
    pending = trajectory.get("pending_review_candidates") or {}
    metrics = (trajectory.get("main_eval_stage") or {}).get("coco_metrics") or {}
    summary = {
        "trajectory_id": trajectory.get("trajectory_id"),
        "run_id": trajectory.get("run_id"),
        "image_id": trajectory.get("image_id"),
        "input_snapshot": trajectory.get("input_snapshot") or {},
        "main_model": (trajectory.get("main_model_stage") or {}).get("model_id"),
        "main_eval": {"coco_metrics": metrics},
        "geometry_review": {
            "failure_tag_count": len((trajectory.get("geometry_review_stage") or {}).get("failure_tags") or []),
            "geometry_profile": (trajectory.get("geometry_review_stage") or {}).get("geometry_profile") or {},
        },
        "roi_summary": {
            "count": len(rois),
            "by_error_type": _count_by(rois, "level1_error_type"),
            "by_failure_family": _count_by(rois, "failure_family"),
            "clusters": len(clusters),
        },
        "expert_task_summary": {
            "count": len(tasks),
            "by_expert_model": _count_by(tasks, "expert_model"),
            "by_error_type": _count_by(tasks, "level1_error_type"),
        },
        "expert_review_summary": {
            "count": len(reviews),
            "by_decision": _count_by(reviews, "decision"),
            "accepted_roi_count": sum(len(item.get("accepted_roi_ids") or []) for item in reviews),
            "rejected_roi_count": sum(len(item.get("rejected_roi_ids") or []) for item in reviews),
        },
        "fusion_summary": {
            "decision": (trajectory.get("fusion_stage") or {}).get("decision"),
            "final_result_source": (trajectory.get("fusion_stage") or {}).get("final_result_source"),
            "fusion_event_count": len((trajectory.get("fusion_stage") or {}).get("fusion_events") or []),
        },
        "pending_candidates": {
            "memory": len(pending.get("memory_candidates") or []),
            "skill": len(pending.get("skill_candidates") or []),
            "training": len(pending.get("training_candidates") or []),
            "distillation": len(pending.get("distillation_candidates") or []),
        },
    }
    context = {
        **summary,
        "protected_stages": {
            "main_eval_stage": trajectory.get("main_eval_stage") or {},
            "roi_stage_summary": summary["roi_summary"],
            "expert_review_stage_summary": summary["expert_review_summary"],
            "fusion_stage": trajectory.get("fusion_stage") or {},
            "pending_candidate_summary": _pending_candidate_summary(pending),
        },
    }
    return summary, context
