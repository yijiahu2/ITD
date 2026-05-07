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


def _pair_denominator(instance_count: int) -> float:
    if instance_count <= 1:
        return 0.0
    return float(instance_count * (instance_count - 1) / 2.0)


def _weighted_valid_mean(terms: list[tuple[float | None, float]]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for value, weight in terms:
        if value is None or weight <= 0:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _shape_anomaly_ratio(geometry: dict[str, Any]) -> float:
    tiny_width = float(geometry.get("tiny_width_ratio_relative") or geometry.get("tiny_width_ratio_lt_1m") or 0.0)
    extreme_small = float(geometry.get("small_fragment_ratio_relative") or geometry.get("small_fragment_ratio_lt_1m2") or 0.0)
    dominant_share_threshold = float(geometry.get("dominant_share_threshold") or 0.35)
    dominant_blob = _safe_div(float(geometry.get("max_instance_area_share") or 0.0), dominant_share_threshold)
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

    small_fragment_ratio = float(geometry.get("small_fragment_ratio_relative") or geometry.get("small_fragment_ratio_lt_4m2") or 0.0)
    width_outlier_ratio = float(geometry.get("width_outlier_ratio") or geometry.get("large_width_ratio_gt_6m") or 0.0)
    dominant_share_threshold = float(geometry.get("dominant_share_threshold") or 0.35)
    dominant_area_risk = _clamp01(_safe_div(float(geometry.get("max_instance_area_share") or 0.0), dominant_share_threshold))
    large_blob_ratio = _clamp01(max(width_outlier_ratio, dominant_area_risk))
    duplicate_overlap_ratio = _safe_div(
        float(geometry.get("overlap_pair_count") or 0.0),
        _pair_denominator(pred_instance_count),
    )
    edge_artifact_score = float(geometry.get("edge_touch_ratio") or 0.0)
    shape_anomaly_ratio = _shape_anomaly_ratio(geometry)
    overlap_iou = semantic.get("overlap_iou")
    semantic_gap = semantic.get("semantic_gap")
    semantic_instance_consistency = float(overlap_iou) if semantic.get("available") and overlap_iou is not None else None
    semantic_coverage_gap = float(semantic_gap) if semantic.get("available") and semantic_gap is not None else None
    semantic_instance_conflict_flag = bool(
        semantic.get("available")
        and (
            (semantic_instance_consistency is not None and float(semantic_instance_consistency) < 0.50)
            or (semantic_coverage_gap is not None and float(semantic_coverage_gap) > 0.30)
        )
    )
    fragmentation_score = _weighted_valid_mean(
        [
            (small_fragment_ratio, 0.65),
            (shape_anomaly_ratio, 0.35),
        ]
    )
    merge_blob_score = _weighted_valid_mean(
        [
            (large_blob_ratio, 0.45),
            (float(duplicate_overlap_ratio) if duplicate_overlap_ratio is not None else None, 0.20),
            (None if semantic_instance_consistency is None else (1.0 - float(semantic_instance_consistency)), 0.35),
        ]
    )
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
        "width_outlier_ratio": width_outlier_ratio,
        "dominant_area_risk": dominant_area_risk,
        "large_blob_ratio": large_blob_ratio,
        "duplicate_overlap_ratio": float(duplicate_overlap_ratio or 0.0),
        "edge_artifact_score": edge_artifact_score,
        "fragmentation_score": float(fragmentation_score or 0.0),
        "merge_blob_score": float(merge_blob_score or 0.0),
        "semantic_instance_consistency": semantic_instance_consistency,
        "semantic_coverage_gap": semantic_coverage_gap,
        "semantic_instance_conflict_flag": semantic_instance_conflict_flag,
        "raw_feature_count": raw_feature_count,
        "valid_instance_count": valid_instance_count,
        "invalid_instance_count": invalid_instance_count,
    }
