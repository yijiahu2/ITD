from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ITD_agent.evolution.bbox import bbox_iou, bbox_overlap_ratio, instance_xyxy, union_bbox


@dataclass(frozen=True)
class CocoErrorDecompositionResult:
    matches: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _instance_id(instance: dict[str, Any], fallback: str) -> str:
    value = instance.get("id")
    if value is None:
        value = instance.get("annotation_id")
    return str(value if value is not None else fallback)


def _error(
    *,
    error_id: str,
    level1_error_type: str,
    gt_instances: list[dict[str, Any]],
    pred_instances: list[dict[str, Any]],
    score: float,
) -> dict[str, Any]:
    boxes = [instance_xyxy(item) for item in [*gt_instances, *pred_instances]]
    return {
        "error_id": error_id,
        "level1_error_type": level1_error_type,
        "affected_gt_ids": [_instance_id(item, f"gt_{idx}") for idx, item in enumerate(gt_instances)],
        "affected_pred_ids": [_instance_id(item, f"pred_{idx}") for idx, item in enumerate(pred_instances)],
        "bbox_px": list(union_bbox(boxes)),
        "severity_score": max(0.0, min(1.0, float(score))),
    }


def decompose_coco_errors(
    *,
    gt_instances: list[dict[str, Any]],
    pred_instances: list[dict[str, Any]],
    iou_threshold: float = 0.5,
    weak_overlap_threshold: float = 0.1,
) -> CocoErrorDecompositionResult:
    gt_by_id = {_instance_id(item, f"gt_{idx}"): item for idx, item in enumerate(gt_instances)}
    pred_by_id = {_instance_id(item, f"pred_{idx}"): item for idx, item in enumerate(pred_instances)}

    pair_scores: list[tuple[float, str, str]] = []
    weak_gt_by_pred: dict[str, list[str]] = {pred_id: [] for pred_id in pred_by_id}
    weak_pred_by_gt: dict[str, list[str]] = {gt_id: [] for gt_id in gt_by_id}
    for gt_id, gt in gt_by_id.items():
        gt_box = instance_xyxy(gt)
        for pred_id, pred in pred_by_id.items():
            pred_box = instance_xyxy(pred)
            iou = bbox_iou(gt_box, pred_box)
            if iou >= iou_threshold:
                pair_scores.append((iou, gt_id, pred_id))
            if bbox_overlap_ratio(gt_box, pred_box) >= weak_overlap_threshold:
                weak_gt_by_pred[pred_id].append(gt_id)
                weak_pred_by_gt[gt_id].append(pred_id)

    matches: list[dict[str, Any]] = []
    used_gt: set[str] = set()
    used_pred: set[str] = set()
    for iou, gt_id, pred_id in sorted(pair_scores, reverse=True):
        if gt_id in used_gt or pred_id in used_pred:
            continue
        used_gt.add(gt_id)
        used_pred.add(pred_id)
        matches.append({"gt_id": gt_id, "pred_id": pred_id, "iou": iou})

    errors: list[dict[str, Any]] = []
    for gt_id, gt in gt_by_id.items():
        if gt_id not in used_gt:
            errors.append(
                _error(
                    error_id=f"fn_{gt_id}",
                    level1_error_type="false_negative",
                    gt_instances=[gt],
                    pred_instances=[],
                    score=1.0,
                )
            )
    for pred_id, pred in pred_by_id.items():
        if pred_id not in used_pred:
            errors.append(
                _error(
                    error_id=f"fp_{pred_id}",
                    level1_error_type="false_positive",
                    gt_instances=[],
                    pred_instances=[pred],
                    score=float(pred.get("score", 1.0)),
                )
            )
    for pred_id, gt_ids in weak_gt_by_pred.items():
        if len(gt_ids) >= 2:
            errors.append(
                _error(
                    error_id=f"under_{pred_id}",
                    level1_error_type="under_segmentation",
                    gt_instances=[gt_by_id[gt_id] for gt_id in gt_ids],
                    pred_instances=[pred_by_id[pred_id]],
                    score=min(1.0, 0.5 + 0.15 * len(gt_ids)),
                )
            )
    for gt_id, pred_ids in weak_pred_by_gt.items():
        if len(pred_ids) >= 2:
            errors.append(
                _error(
                    error_id=f"over_{gt_id}",
                    level1_error_type="over_segmentation",
                    gt_instances=[gt_by_id[gt_id]],
                    pred_instances=[pred_by_id[pred_id] for pred_id in pred_ids],
                    score=min(1.0, 0.5 + 0.15 * len(pred_ids)),
                )
            )

    metrics = {
        "gt_count": len(gt_instances),
        "pred_count": len(pred_instances),
        "matched_count": len(matches),
        "false_negative_count": sum(1 for item in errors if item["level1_error_type"] == "false_negative"),
        "false_positive_count": sum(1 for item in errors if item["level1_error_type"] == "false_positive"),
        "under_segmentation_count": sum(1 for item in errors if item["level1_error_type"] == "under_segmentation"),
        "over_segmentation_count": sum(1 for item in errors if item["level1_error_type"] == "over_segmentation"),
        "mean_iou_matched": sum(item["iou"] for item in matches) / len(matches) if matches else 0.0,
    }
    return CocoErrorDecompositionResult(matches=matches, errors=errors, metrics=metrics)
