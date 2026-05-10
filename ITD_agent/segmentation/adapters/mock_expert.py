from __future__ import annotations

from typing import Any

from ITD_agent.evolution.bbox import bbox_iou, instance_xyxy
from ITD_agent.evolution.expert.expert_task_builder import ExpertTask


def run_mock_expert_task(
    *,
    task: ExpertTask,
    gt_instances: list[dict[str, Any]],
    main_instances: list[dict[str, Any]],
    strategy: str = "use_gt_or_perturbed_gt",
) -> dict[str, Any]:
    if strategy == "copy_main_result":
        instances = list(main_instances)
        oracle_mock = False
    elif strategy == "simulate_failure":
        instances = []
        oracle_mock = False
    else:
        fusion_boxes = [tuple(box) for box in task.fusion_bboxes.values()]
        instances = [
            {**gt, "id": f"expert_gt_{gt.get('id')}", "source": "mock_gt_oracle"}
            for gt in gt_instances
            if any(bbox_iou(instance_xyxy(gt), box) > 0.0 for box in fusion_boxes)
        ]
        oracle_mock = True
    return {
        "expert_task_id": task.expert_task_id,
        "expert_model": task.expert_model,
        "execution_mode": "mock",
        "oracle_mock": oracle_mock,
        "strategy": strategy,
        "instances": instances,
        "status": "completed",
    }
