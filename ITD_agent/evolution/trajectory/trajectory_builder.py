from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def start_trajectory(*, run_id: str, image: dict[str, Any], annotation_json: str) -> dict[str, Any]:
    image_id = str(image["id"])
    return {
        "trajectory_id": f"traj_{run_id}_{image_id}",
        "run_id": run_id,
        "image_id": image_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "adaptive_inference",
        "mainline_profile": "A_DOM_ONLY",
        "input_snapshot": {
            "image_path": image.get("file_name"),
            "annotation_json": annotation_json,
            "gt_instance_count": 0,
            "width": image.get("width"),
            "height": image.get("height"),
        },
        "main_model_stage": {},
        "main_eval_stage": {},
        "geometry_review_stage": {},
        "main_decision_stage": {},
        "roi_stage": {"roi_candidates": [], "roi_clusters": []},
        "expert_task_stage": {"expert_tasks": [], "routing_events": []},
        "expert_review_stage": {"expert_reviews": []},
        "expert_decision_stage": {},
        "fusion_stage": {"fusion_events": [], "final_result_source": "main_only"},
        "final_evaluation_stage": {},
        "pending_review_candidates": {
            "memory_candidates": [],
            "skill_candidates": [],
            "training_candidates": [],
            "distillation_candidates": [],
        },
        "review_status": "pending",
    }


def summarize_trajectory(trajectory: dict[str, Any], trajectory_path: str) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory["trajectory_id"],
        "image_id": trajectory["image_id"],
        "trajectory_path": trajectory_path,
        "final_result_source": trajectory["fusion_stage"].get("final_result_source", "main_only"),
        "roi_candidate_count": len(trajectory["roi_stage"].get("roi_candidates") or []),
        "expert_task_count": len(trajectory["expert_task_stage"].get("expert_tasks") or []),
        "training_candidate_count": len(trajectory["pending_review_candidates"].get("training_candidates") or []),
    }
