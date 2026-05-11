from __future__ import annotations

from typing import Any

from ITD_agent.common.values import safe_float as _safe_float


DEFAULT_POLICY_TEMPLATES: dict[str, dict[str, Any]] = {
    "default": {
        "diam_list": "96,192,320",
        "augment": False,
        "iou_merge_thr": 0.28,
        "enable_tile_fast_check": False,
        "fusion_priority": "normal",
    },
    "dense_small_crown": {
        "diam_list": "64,96,160",
        "augment": True,
        "iou_merge_thr": 0.30,
        "enable_tile_fast_check": True,
        "fusion_priority": "normal",
    },
    "large_sparse_crown": {
        "diam_list": "128,256,320",
        "augment": False,
        "iou_merge_thr": 0.24,
        "enable_tile_fast_check": False,
        "fusion_priority": "normal",
    },
    "shadow_weak_boundary": {
        "diam_list": "96,192,320",
        "augment": True,
        "iou_merge_thr": 0.28,
        "enable_tile_fast_check": True,
        "fusion_priority": "normal",
    },
    "high_heterogeneity": {
        "diam_list": "64,128,256,320",
        "augment": True,
        "iou_merge_thr": 0.28,
        "enable_tile_fast_check": True,
        "fusion_priority": "normal",
    },
}


def infer_expected_failure_modes(block_features: dict[str, Any]) -> list[str]:
    modes: list[str] = []
    if bool(block_features.get("dense_texture_flag")):
        modes.append("crown_merge")
        modes.append("under_segmentation")
    if bool(block_features.get("low_texture_flag")):
        modes.append("weak_boundary")
    if (_safe_float(block_features.get("shadow_ratio_estimate")) or 0.0) >= 0.18:
        modes.append("shadow_confusion")
    if (_safe_float(block_features.get("blur_score")) or 0.0) >= 0.35:
        modes.append("boundary_blur")
    if (block_features.get("block_heterogeneity_level") or "") == "high":
        modes.append("local_policy_mismatch")
    ordered: list[str] = []
    for item in modes:
        if item not in ordered:
            ordered.append(item)
    return ordered


def infer_risk_tags(block_features: dict[str, Any]) -> tuple[list[str], list[str]]:
    risk_tags: list[str] = []
    localized_risk_tags: list[str] = []

    shadow = _safe_float(block_features.get("shadow_ratio_estimate")) or 0.0
    blur = _safe_float(block_features.get("blur_score")) or 0.0
    valid_ratio = _safe_float(block_features.get("valid_pixel_ratio")) or 0.0
    heterogeneity_level = str(block_features.get("block_heterogeneity_level") or "")

    if bool(block_features.get("dense_texture_flag")):
        risk_tags.append("dense_texture")
    if bool(block_features.get("low_texture_flag")):
        risk_tags.append("low_texture")
    if shadow >= 0.35:
        risk_tags.append("heavy_shadow")
    elif shadow >= 0.18:
        risk_tags.append("moderate_shadow")
    if blur >= 0.35:
        risk_tags.append("blur_risk")
    if valid_ratio <= 0.50:
        risk_tags.append("low_valid_area")
    if heterogeneity_level == "high":
        risk_tags.append("high_heterogeneity")
        localized_risk_tags.append("local_shadow_patch")
        localized_risk_tags.append("local_texture_shift")

    return list(dict.fromkeys(risk_tags)), list(dict.fromkeys(localized_risk_tags))


def infer_quality_class(block_features: dict[str, Any]) -> str:
    score = 0
    if (_safe_float(block_features.get("shadow_ratio_estimate")) or 0.0) >= 0.18:
        score += 1
    if (_safe_float(block_features.get("blur_score")) or 0.0) >= 0.35:
        score += 1
    if bool(block_features.get("low_texture_flag")):
        score += 1
    if (block_features.get("block_heterogeneity_level") or "") == "high":
        score += 1
    if score >= 3:
        return "high_risk"
    if score >= 1:
        return "medium_risk"
    return "low_risk"


def infer_priority_score(block_features: dict[str, Any]) -> float:
    score = 0.35
    score += min((_safe_float(block_features.get("shadow_ratio_estimate")) or 0.0) * 0.35, 0.20)
    score += min((_safe_float(block_features.get("texture_complexity_score")) or 0.0) * 0.20, 0.15)
    if (block_features.get("block_heterogeneity_level") or "") == "high":
        score += 0.18
    if bool(block_features.get("dense_texture_flag")):
        score += 0.07
    if (_safe_float(block_features.get("valid_pixel_ratio")) or 1.0) < 0.50:
        score -= 0.15
    return max(0.0, min(score, 1.0))


def select_policy_template(block_features: dict[str, Any]) -> dict[str, Any]:
    if (block_features.get("block_heterogeneity_level") or "") == "high":
        template_name = "high_heterogeneity"
    elif (_safe_float(block_features.get("shadow_ratio_estimate")) or 0.0) >= 0.18 and bool(block_features.get("low_texture_flag")):
        template_name = "shadow_weak_boundary"
    elif bool(block_features.get("dense_texture_flag")) and (_safe_float(block_features.get("gradient_mean")) or 0.0) >= 0.20:
        template_name = "dense_small_crown"
    elif bool(block_features.get("low_texture_flag")) and (_safe_float(block_features.get("texture_complexity_score")) or 1.0) <= 0.35:
        template_name = "large_sparse_crown"
    else:
        template_name = "default"

    selected = dict(DEFAULT_POLICY_TEMPLATES[template_name])
    selected["policy_template_name"] = template_name
    selected["expected_failure_modes"] = infer_expected_failure_modes(block_features)
    selected["quality_class"] = infer_quality_class(block_features)
    selected["priority_score"] = infer_priority_score(block_features)
    selected["risk_tags"], selected["localized_risk_tags"] = infer_risk_tags(block_features)
    return selected
