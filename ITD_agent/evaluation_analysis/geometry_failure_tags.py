from __future__ import annotations

from typing import Any


def build_geometry_failure_tags(geometry_profile: dict[str, Any]) -> list[dict[str, Any]]:
    measurements = list(geometry_profile.get("measurements") or [])
    if not measurements:
        return []
    areas = sorted(float(item.get("area", 0.0)) for item in measurements)
    p5 = areas[max(0, int(len(areas) * 0.05) - 1)]
    p95 = areas[min(len(areas) - 1, int(len(areas) * 0.95))]
    tags: list[dict[str, Any]] = []
    for item in measurements:
        instance_id = str(int(item["instance_id"])) if item.get("instance_id") else ""
        area = float(item.get("area", 0.0))
        axis_ratio = float(item.get("axis_ratio", 1.0))
        if area <= p5 and len(areas) >= 3:
            tags.append({"instance_id": instance_id, "tag": "tiny_false_positive", "severity_score": 0.55})
        if area >= p95 and len(areas) >= 3:
            tags.append({"instance_id": instance_id, "tag": "oversized_crown", "severity_score": 0.55})
        if axis_ratio >= 3.0:
            tags.append({"instance_id": instance_id, "tag": "elongated_false_positive", "severity_score": 0.65})
    return tags
