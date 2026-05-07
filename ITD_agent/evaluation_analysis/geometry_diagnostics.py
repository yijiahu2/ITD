from __future__ import annotations

from typing import Any


def _clamp01(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _shape_anomaly_ratio(geometry: dict[str, Any]) -> float:
    tiny_width = float(geometry.get("tiny_width_ratio_lt_1m") or 0.0)
    extreme_small = float(geometry.get("small_fragment_ratio_lt_1m2") or 0.0)
    dominant_blob = max(float(geometry.get("max_instance_area_share") or 0.0) - 0.35, 0.0) / 0.65
    return _clamp01((tiny_width * 0.35) + (extreme_small * 0.35) + (dominant_blob * 0.30))


def build_geometry_diagnostics(metrics: dict[str, Any]) -> dict[str, Any]:
    patch = metrics.get("patch_context") or {}
    instance_stats = metrics.get("instance_stats") or {}
    semantic = metrics.get("semantic_instance_consistency") or {}
    geometry = metrics.get("geometry_plausibility") or {}

    pred_instance_count = int(geometry.get("instance_count") or instance_stats.get("valid_instance_count") or 0)
    raw_feature_count = int(instance_stats.get("raw_feature_count") or pred_instance_count)
    valid_instance_count = int(instance_stats.get("valid_instance_count") or pred_instance_count)
    invalid_instance_count = int(instance_stats.get("invalid_instance_count") or max(raw_feature_count - valid_instance_count, 0))
    valid_instance_ratio = _safe_div(valid_instance_count, raw_feature_count) if raw_feature_count > 0 else 0.0

    small_fragment_ratio = float(geometry.get("small_fragment_ratio_lt_4m2") or 0.0)
    large_blob_ratio = _clamp01(max(float(geometry.get("large_width_ratio_gt_6m") or 0.0), float(geometry.get("max_instance_area_share") or 0.0)))
    duplicate_overlap_ratio = _safe_div(
        float(geometry.get("overlap_pair_count") or 0.0),
        float(pred_instance_count),
    )
    edge_artifact_score = float(geometry.get("edge_touch_ratio") or 0.0)
    shape_anomaly_ratio = _shape_anomaly_ratio(geometry)
    semantic_instance_consistency = float(semantic.get("overlap_iou") or 0.0) if semantic.get("available") else None
    semantic_coverage_gap = float(semantic.get("semantic_gap") or 0.0) if semantic.get("available") else None
    semantic_instance_conflict_flag = bool(
        semantic.get("available")
        and (
            float(semantic.get("overlap_iou") or 0.0) < 0.50
            or float(semantic.get("semantic_gap") or 0.0) > 0.30
        )
    )
    fragmentation_score = _clamp01((small_fragment_ratio * 0.65) + (shape_anomaly_ratio * 0.35))
    merge_blob_score = _clamp01((large_blob_ratio * 0.60) + ((1.0 - float(semantic_instance_consistency or 0.0)) * 0.40))
    patch_area_m2 = float(patch.get("patch_area_m2") or 0.0)
    pred_cover_ratio = semantic.get("instance_cover_ratio")
    if pred_cover_ratio is None and patch_area_m2 > 0:
        pred_cover_ratio = _safe_div(float(geometry.get("union_area_m2") or 0.0), patch_area_m2)
    pred_cover_ratio = float(pred_cover_ratio or 0.0)
    empty_output_flag = bool(pred_instance_count <= 0 or pred_cover_ratio < 0.005)

    return {
        "pred_instance_count": pred_instance_count,
        "empty_output_flag": empty_output_flag,
        "pred_cover_ratio": pred_cover_ratio,
        "valid_instance_ratio": float(valid_instance_ratio or 0.0),
        "shape_anomaly_ratio": shape_anomaly_ratio,
        "small_fragment_ratio": small_fragment_ratio,
        "large_blob_ratio": large_blob_ratio,
        "duplicate_overlap_ratio": float(duplicate_overlap_ratio or 0.0),
        "edge_artifact_score": edge_artifact_score,
        "fragmentation_score": fragmentation_score,
        "merge_blob_score": merge_blob_score,
        "semantic_instance_consistency": semantic_instance_consistency,
        "semantic_coverage_gap": semantic_coverage_gap,
        "semantic_instance_conflict_flag": semantic_instance_conflict_flag,
        "raw_feature_count": raw_feature_count,
        "valid_instance_count": valid_instance_count,
        "invalid_instance_count": invalid_instance_count,
    }
