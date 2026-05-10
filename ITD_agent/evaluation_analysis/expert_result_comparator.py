from __future__ import annotations

from typing import Any

from ITD_agent.evolution.bbox import bbox_iou, instance_xyxy
from ITD_agent.evolution.expert.expert_task_builder import ExpertTask


def _count_instances_in_boxes(instances: list[dict[str, Any]], boxes: list[tuple[float, float, float, float]]) -> int:
    return sum(1 for instance in instances if any(bbox_iou(instance_xyxy(instance), box) > 0.0 for box in boxes))


def compare_expert_with_main(
    *,
    expert_tasks: list[ExpertTask],
    expert_results: list[dict[str, Any]],
    main_instances: list[dict[str, Any]],
    gt_instances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result_by_task = {str(result.get("expert_task_id")): result for result in expert_results}
    reviews: list[dict[str, Any]] = []
    for task in expert_tasks:
        result = result_by_task.get(task.expert_task_id, {})
        boxes = [tuple(box) for box in task.fusion_bboxes.values()]
        main_count = _count_instances_in_boxes(main_instances, boxes)
        expert_count = _count_instances_in_boxes(result.get("instances") or [], boxes)
        gt_count = _count_instances_in_boxes(gt_instances, boxes)
        main_abs_error = abs(main_count - gt_count)
        expert_abs_error = abs(expert_count - gt_count)
        improvement = main_abs_error - expert_abs_error
        if improvement > 0:
            decision = "accept"
            accepted = list(task.roi_ids)
            rejected: list[str] = []
        elif improvement == 0 and expert_count > 0 and task.level1_error_type == "false_positive":
            decision = "record_uncertain"
            accepted = []
            rejected = list(task.roi_ids)
        else:
            decision = "reject"
            accepted = []
            rejected = list(task.roi_ids)
        reviews.append(
            {
                "review_id": f"review_{task.expert_task_id}",
                "expert_task_id": task.expert_task_id,
                "decision": decision,
                "improvement": {
                    "roi_main_instance_count": main_count,
                    "roi_expert_instance_count": expert_count,
                    "roi_gt_instance_count": gt_count,
                    "absolute_error_reduction": improvement,
                },
                "safety": {
                    "roi_outside_preserved_by_fusion": True,
                    "side_effect_level": "none",
                },
                "accepted_roi_ids": accepted,
                "rejected_roi_ids": rejected,
                "roi_bboxes": dict(task.fusion_bboxes),
            }
        )
    return reviews
