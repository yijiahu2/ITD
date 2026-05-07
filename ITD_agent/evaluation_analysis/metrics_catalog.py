from __future__ import annotations

from typing import Any


REFERENCE_METRIC_CATALOG = {
    "tree_count_error_ratio": {
        "category": "inventory_count_alignment",
        "label": "Tree count alignment",
        "direction": "lower_is_better",
    },
    "mean_crown_width_error_ratio": {
        "category": "crown_boundary_alignment",
        "label": "Crown boundary alignment",
        "direction": "lower_is_better",
    },
    "closure_error_abs": {
        "category": "canopy_closure_alignment",
        "label": "Canopy closure alignment",
        "direction": "lower_is_better",
    },
    "density_error_ratio": {
        "category": "stand_density_alignment",
        "label": "Stand density alignment",
        "direction": "lower_is_better",
    },
}

ONLINE_METRIC_CATALOG = {
    "semantic_instance_consistency": {
        "category": "semantic_instance_alignment",
        "label": "Semantic-instance alignment",
        "direction": "higher_overlap_is_better",
    },
    "geometry_plausibility": {
        "category": "instance_geometry_plausibility",
        "label": "Instance geometry plausibility",
        "direction": "lower_fragmentation_is_better",
    },
    "height_consistency": {
        "category": "height_support_consistency",
        "label": "Height support consistency",
        "direction": "higher_support_is_better",
    },
}


def describe_metric(metric_name: str) -> dict[str, Any]:
    if metric_name in REFERENCE_METRIC_CATALOG:
        return dict(REFERENCE_METRIC_CATALOG[metric_name])
    if metric_name in ONLINE_METRIC_CATALOG:
        return dict(ONLINE_METRIC_CATALOG[metric_name])
    return {
        "category": "uncategorized",
        "label": metric_name,
        "direction": "unknown",
    }
