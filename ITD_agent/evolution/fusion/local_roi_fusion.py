from __future__ import annotations

from typing import Any

from ITD_agent.evolution.bbox import bbox_iou, instance_xyxy


def fuse_or_rollback(
    *,
    main_instances: list[dict[str, Any]],
    expert_results: list[dict[str, Any]],
    expert_reviews: list[dict[str, Any]],
    min_improvement_epsilon: float = 0.01,
) -> dict[str, Any]:
    accepted_reviews = [review for review in expert_reviews if review.get("decision") in {"accept", "partial_accept"}]
    if not accepted_reviews:
        return {
            "decision": "rollback_to_main",
            "final_result_source": "rollback_to_main",
            "instances": list(main_instances),
            "fusion_events": [
                {
                    "decision": "rollback_to_main",
                    "reason": "no_accepted_expert_reviews",
                }
            ],
        }

    result_by_task = {str(result.get("expert_task_id")): result for result in expert_results}
    accepted_boxes: list[tuple[float, float, float, float]] = []
    expert_instances: list[dict[str, Any]] = []
    accepted_roi_count = 0
    total_roi_count = 0
    for review in expert_reviews:
        total_roi_count += len(review.get("accepted_roi_ids") or []) + len(review.get("rejected_roi_ids") or [])
        if review not in accepted_reviews:
            continue
        roi_bboxes = review.get("roi_bboxes") or {}
        for roi_id in review.get("accepted_roi_ids") or []:
            if roi_id in roi_bboxes:
                accepted_boxes.append(tuple(float(v) for v in roi_bboxes[roi_id]))
                accepted_roi_count += 1
        expert_instances.extend(list((result_by_task.get(str(review.get("expert_task_id"))) or {}).get("instances") or []))

    if not accepted_boxes or accepted_roi_count * min_improvement_epsilon <= 0:
        return {
            "decision": "rollback_to_main",
            "final_result_source": "rollback_to_main",
            "instances": list(main_instances),
            "fusion_events": [{"decision": "rollback_to_main", "reason": "accepted_roi_missing"}],
        }

    kept_main = [
        instance
        for instance in main_instances
        if not any(bbox_iou(instance_xyxy(instance), box) > 0.0 for box in accepted_boxes)
    ]
    fused_expert = [
        instance
        for instance in expert_instances
        if any(bbox_iou(instance_xyxy(instance), box) > 0.0 for box in accepted_boxes)
    ]
    source = "expert_fused" if accepted_roi_count == total_roi_count else "partial_expert_fused"
    return {
        "decision": "fuse",
        "final_result_source": source,
        "instances": [*kept_main, *fused_expert],
        "fusion_events": [
            {
                "decision": source,
                "accepted_roi_count": accepted_roi_count,
                "total_roi_count": total_roi_count,
                "roi_outside_main_instances_preserved": len(kept_main),
            }
        ],
    }
