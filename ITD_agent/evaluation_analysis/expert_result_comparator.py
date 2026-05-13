from __future__ import annotations

from typing import Any

from ITD_agent.evolution.bbox import bbox_iou, instance_xyxy
from ITD_agent.evolution.expert.expert_task_builder import ExpertTask


def _count_instances_in_boxes(instances: list[dict[str, Any]], boxes: list[tuple[float, float, float, float]]) -> int:
    return sum(1 for instance in instances if any(bbox_iou(instance_xyxy(instance), box) > 0.0 for box in boxes))


def _count_instances_in_box(instances: list[dict[str, Any]], box: tuple[float, float, float, float]) -> int:
    return sum(1 for instance in instances if bbox_iou(instance_xyxy(instance), box) > 0.0)


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
        boxes_by_roi = {str(roi_id): tuple(box) for roi_id, box in task.fusion_bboxes.items()}
        boxes = list(boxes_by_roi.values())
        main_count = _count_instances_in_boxes(main_instances, boxes)
        expert_count = _count_instances_in_boxes(result.get("instances") or [], boxes)
        gt_count = _count_instances_in_boxes(gt_instances, boxes)
        main_abs_error = abs(main_count - gt_count)
        expert_abs_error = abs(expert_count - gt_count)
        improvement = main_abs_error - expert_abs_error

        roi_reviews: list[dict[str, Any]] = []
        accepted: list[str] = []
        rejected: list[str] = []
        for roi_id in task.roi_ids:
            box = boxes_by_roi.get(str(roi_id))
            if box is None:
                rejected.append(str(roi_id))
                roi_reviews.append({"roi_id": str(roi_id), "decision": "reject", "reason": "missing_fusion_bbox"})
                continue
            roi_main_count = _count_instances_in_box(main_instances, box)
            roi_expert_count = _count_instances_in_box(result.get("instances") or [], box)
            roi_gt_count = _count_instances_in_box(gt_instances, box)
            roi_improvement = abs(roi_main_count - roi_gt_count) - abs(roi_expert_count - roi_gt_count)
            roi_decision = "accept" if roi_improvement > 0 else "reject"
            if roi_decision == "accept":
                accepted.append(str(roi_id))
            else:
                rejected.append(str(roi_id))
            roi_reviews.append(
                {
                    "roi_id": str(roi_id),
                    "decision": roi_decision,
                    "main_instance_count": roi_main_count,
                    "expert_instance_count": roi_expert_count,
                    "gt_instance_count": roi_gt_count,
                    "absolute_error_reduction": roi_improvement,
                }
            )

        if accepted and rejected:
            decision = "partial_accept"
        elif accepted:
            decision = "accept"
        elif improvement == 0 and expert_count > 0 and task.level1_error_type == "false_positive":
            decision = "record_uncertain"
            rejected = list(task.roi_ids)
        else:
            decision = "reject"
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
                "roi_reviews": roi_reviews,
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
