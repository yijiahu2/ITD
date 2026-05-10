from __future__ import annotations

from math import sqrt
from typing import Any

from ITD_agent.evolution.bbox import bbox_area, instance_xyxy


def build_geometry_profile(instances: list[dict[str, Any]]) -> dict[str, Any]:
    measurements: list[dict[str, float]] = []
    for instance in instances:
        x1, y1, x2, y2 = instance_xyxy(instance)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = bbox_area((x1, y1, x2, y2))
        axis_ratio = max(width, height) / max(1.0, min(width, height))
        equivalent_diameter = 2.0 * sqrt(area / 3.141592653589793) if area > 0 else 0.0
        measurements.append(
            {
                "instance_id": float(instance.get("id", 0)) if str(instance.get("id", "")).isdigit() else 0.0,
                "area": area,
                "equivalent_diameter": equivalent_diameter,
                "axis_ratio": axis_ratio,
                "compactness": min(width, height) / max(1.0, max(width, height)),
                "circularity": min(width, height) / max(1.0, max(width, height)),
                "solidity": 1.0,
                "boundary_complexity": axis_ratio,
                "hole_count": 0.0,
                "hole_area_ratio": 0.0,
            }
        )
    areas = [item["area"] for item in measurements]
    return {
        "instance_count": len(measurements),
        "measurements": measurements,
        "area_min": min(areas) if areas else 0.0,
        "area_max": max(areas) if areas else 0.0,
        "area_mean": sum(areas) / len(areas) if areas else 0.0,
    }
