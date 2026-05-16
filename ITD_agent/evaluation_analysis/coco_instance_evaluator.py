from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ITD_agent.segmentation.coco_utils import segmentation_to_rle


@dataclass(frozen=True)
class InstanceMask:
    instance: dict[str, Any]
    mask: np.ndarray
    area: float
    mask_area: int
    bbox_xyxy: tuple[float, float, float, float]


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def _bbox_to_mask(bbox: Any, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if not isinstance(bbox, list) or len(bbox) < 4:
        return mask
    x, y, w, h = [float(value) for value in bbox[:4]]
    x0 = max(0, min(width, int(round(x))))
    y0 = max(0, min(height, int(round(y))))
    x1 = max(0, min(width, int(round(x + max(w, 0.0)))))
    y1 = max(0, min(height, int(round(y + max(h, 0.0)))))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def _bbox_xyxy_from_instance(instance: dict[str, Any], mask: np.ndarray) -> tuple[float, float, float, float]:
    bbox = instance.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        x, y, w, h = [float(value) for value in bbox[:4]]
        return (x, y, x + max(w, 0.0), y + max(h, 0.0))
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _bbox_intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _segmentation_to_mask(instance: dict[str, Any], height: int, width: int) -> np.ndarray:
    segmentation = instance.get("segmentation")
    if segmentation is not None:
        try:
            from pycocotools import mask as mask_utils

            rle = segmentation_to_rle(segmentation, height, width)
            return mask_utils.decode(rle).astype(bool)
        except Exception:
            pass
    return _bbox_to_mask(instance.get("bbox"), height, width)


def _image_dims(coco: dict[str, Any]) -> dict[str, tuple[int, int]]:
    dims: dict[str, tuple[int, int]] = {}
    for image in coco.get("images") or []:
        image_id = str(image.get("id"))
        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        if width > 0 and height > 0:
            dims[image_id] = (height, width)
    return dims


def _instances_by_image(instances: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in instances:
        image_id = str(item.get("image_id"))
        grouped.setdefault(image_id, []).append(dict(item))
    return grouped


def _build_masks(instances: list[dict[str, Any]], dims: dict[str, tuple[int, int]]) -> dict[str, list[InstanceMask]]:
    grouped: dict[str, list[InstanceMask]] = {}
    for image_id, values in _instances_by_image(instances).items():
        height, width = dims.get(str(image_id), (0, 0))
        if height <= 0 or width <= 0:
            continue
        image_masks: list[InstanceMask] = []
        for item in values:
            mask = _segmentation_to_mask(item, height, width)
            mask_area = int(mask.sum())
            image_masks.append(
                InstanceMask(
                    instance=item,
                    mask=mask,
                    area=float(item.get("area") or mask_area),
                    mask_area=mask_area,
                    bbox_xyxy=_bbox_xyxy_from_instance(item, mask),
                )
            )
        grouped[image_id] = image_masks
    return grouped


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    if inter <= 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return float(inter) / float(max(union, 1))


def evaluate_coco_instances(
    *,
    annotation_path: str | Path,
    prediction_path: str | Path,
    output_path: str | Path | None = None,
    image_ids: list[int | str] | None = None,
    iou_threshold: float = 0.5,
    weak_overlap_threshold: float = 0.1,
) -> dict[str, Any]:
    coco = _load_json(annotation_path)
    predictions = _load_json(prediction_path)
    if isinstance(predictions, dict):
        predictions = predictions.get("annotations") or predictions.get("predictions") or []
    selected_ids = {str(item) for item in (image_ids or [])}
    gt_instances = [
        dict(item)
        for item in coco.get("annotations") or []
        if not selected_ids or str(item.get("image_id")) in selected_ids
    ]
    pred_instances = [
        dict(item)
        for item in predictions or []
        if not selected_ids or str(item.get("image_id")) in selected_ids
    ]

    dims = _image_dims(coco)
    gt_masks = _build_masks(gt_instances, dims)
    pred_masks = _build_masks(pred_instances, dims)
    matches: list[dict[str, Any]] = []
    unmatched_pred: list[InstanceMask] = []
    unmatched_gt: list[InstanceMask] = []
    low_iou_examples: list[dict[str, Any]] = []
    low_iou_count = 0

    image_order = sorted(set(gt_masks) | set(pred_masks))
    total_images = len(image_order)
    print(
        f"[coco-eval] start images={total_images} gt={len(gt_instances)} predictions={len(pred_instances)}",
        flush=True,
    )
    for image_index, image_id in enumerate(image_order, start=1):
        gt_list = gt_masks.get(image_id, [])
        pred_list = pred_masks.get(image_id, [])
        print(
            f"[coco-eval] image {image_index}/{total_images} "
            f"image_id={image_id} gt={len(gt_list)} pred={len(pred_list)}",
            flush=True,
        )
        used_gt: set[int] = set()
        for pred_idx, pred in enumerate(pred_list):
            best_iou = 0.0
            best_gt_idx: int | None = None
            for gt_idx, gt in enumerate(gt_list):
                if gt_idx in used_gt:
                    continue
                bbox_intersection = _bbox_intersection_area(pred.bbox_xyxy, gt.bbox_xyxy)
                if bbox_intersection <= 0.0:
                    continue
                iou_upper_bound = bbox_intersection / float(max(pred.mask_area, gt.mask_area, 1))
                if iou_upper_bound <= best_iou:
                    continue
                value = _mask_iou(pred.mask, gt.mask)
                if value > best_iou:
                    best_iou = value
                    best_gt_idx = gt_idx
            if best_gt_idx is not None and best_iou >= float(iou_threshold):
                used_gt.add(best_gt_idx)
                matches.append(
                    {
                        "image_id": image_id,
                        "prediction": pred.instance,
                        "ground_truth": gt_list[best_gt_idx].instance,
                        "iou": best_iou,
                    }
                )
            else:
                if best_iou >= float(weak_overlap_threshold):
                    low_iou_count += 1
                    low_iou_examples.append(
                        {
                            **pred.instance,
                            "eval_type": "low_iou",
                            "best_iou": best_iou,
                            "is_low_iou": True,
                        }
                    )
                unmatched_pred.append(pred)
        for gt_idx, gt in enumerate(gt_list):
            if gt_idx not in used_gt:
                unmatched_gt.append(gt)

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    precision = float(tp) / float(max(tp + fp, 1))
    recall = float(tp) / float(max(tp + fn, 1))
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    mean_iou = float(sum(item["iou"] for item in matches) / max(tp, 1))
    gt_count = max(len(gt_instances), 1)
    pred_count = max(len(pred_instances), 1)
    false_positive_score = float(fp) / float(max(tp + fp, 1))
    false_negative_score = float(fn) / float(max(tp + fn, 1))
    boundary_quality_score = float(low_iou_count) / float(max(tp + low_iou_count, 1))
    under_segmentation_score = float(sum(1 for item in matches if float(item["prediction"].get("area") or 0) > 1.35 * float(item["ground_truth"].get("area") or 1))) / float(gt_count)
    over_segmentation_score = float(sum(1 for item in matches if float(item["prediction"].get("area") or 0) < 0.65 * float(item["ground_truth"].get("area") or 1))) / float(pred_count)
    scores = {
        "under_segmentation": under_segmentation_score,
        "over_segmentation": over_segmentation_score,
        "false_positive_cleanup": false_positive_score,
        "missed_crown_recall": false_negative_score,
        "boundary_quality": boundary_quality_score,
    }
    dominant_error_type = max(scores.items(), key=lambda item: item[1])[0] if scores else "boundary_quality"
    false_positive_examples = [
        {
            **item.instance,
            "eval_type": "false_positive",
            "is_fp": True,
        }
        for item in unmatched_pred[:50]
    ]
    false_negative_examples = [
        {
            **item.instance,
            "score": 0.0,
            "eval_type": "false_negative",
            "is_fn": True,
        }
        for item in unmatched_gt[:50]
    ]
    metrics = {
        "image_count": len(selected_ids) if selected_ids else len(coco.get("images") or []),
        "gt_instance_count": len(gt_instances),
        "pred_instance_count": len(pred_instances),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "miou": mean_iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "low_iou_count": low_iou_count,
        "under_segmentation_score": under_segmentation_score,
        "over_segmentation_score": over_segmentation_score,
        "false_positive_score": false_positive_score,
        "false_negative_score": false_negative_score,
        "boundary_quality_score": boundary_quality_score,
        "dominant_error_type": dominant_error_type,
        "matches": matches[:200],
        "false_positive_examples": false_positive_examples,
        "false_negative_examples": false_negative_examples,
        "low_iou_examples": low_iou_examples[:50],
    }
    if output_path:
        _write_json(output_path, metrics)
    return metrics
