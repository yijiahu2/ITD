from __future__ import annotations

from typing import Any


BBox = tuple[float, float, float, float]


def coco_bbox_to_xyxy(bbox: list[float] | tuple[float, ...]) -> BBox:
    x, y, w, h = [float(v) for v in bbox[:4]]
    return (x, y, x + max(0.0, w), y + max(0.0, h))


def xyxy_to_coco_bbox(bbox: BBox) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def instance_xyxy(instance: dict[str, Any]) -> BBox:
    bbox = instance.get("bbox_px") or instance.get("bbox")
    if bbox is None:
        return (0.0, 0.0, 0.0, 0.0)
    if len(bbox) == 4 and (instance.get("bbox_format") == "xyxy" or instance.get("bbox_px")):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 >= x1 and y2 >= y1:
            return (x1, y1, x2, y2)
    return coco_bbox_to_xyxy(bbox)


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_intersection(a: BBox, b: BBox) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return bbox_area((x1, y1, x2, y2))


def bbox_iou(a: BBox, b: BBox) -> float:
    inter = bbox_intersection(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def bbox_overlap_ratio(a: BBox, b: BBox) -> float:
    inter = bbox_intersection(a, b)
    smaller = min(bbox_area(a), bbox_area(b))
    if smaller <= 0.0:
        return 0.0
    return inter / smaller


def union_bbox(boxes: list[BBox]) -> BBox:
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def clamp_bbox(bbox: BBox, image_size: tuple[int, int]) -> BBox:
    width, height = image_size
    return (
        max(0.0, min(float(width), bbox[0])),
        max(0.0, min(float(height), bbox[1])),
        max(0.0, min(float(width), bbox[2])),
        max(0.0, min(float(height), bbox[3])),
    )


def expand_bbox(bbox: BBox, buffer_px: int | float, image_size: tuple[int, int] | None = None) -> BBox:
    expanded = (
        bbox[0] - float(buffer_px),
        bbox[1] - float(buffer_px),
        bbox[2] + float(buffer_px),
        bbox[3] + float(buffer_px),
    )
    return clamp_bbox(expanded, image_size) if image_size else expanded
