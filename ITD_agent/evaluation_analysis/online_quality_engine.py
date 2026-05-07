from __future__ import annotations

from typing import Any

from ITD_agent.data_processing.fusion.diagnostics import build_output_diagnostics

from .geometry_diagnostics import build_geometry_diagnostics


def _clamp01(value: float | None) -> float | None:
    if value is None:
        return None
    return float(min(max(float(value), 0.0), 1.0))


def _weighted_valid_mean(terms: dict[str, tuple[float | None, float]]) -> tuple[float | None, dict[str, Any]]:
    numerator = 0.0
    denominator = 0.0
    breakdown: dict[str, Any] = {}
    for name, (value, weight) in terms.items():
        normalized = _clamp01(value)
        breakdown[name] = {
            "value": normalized,
            "weight": float(weight),
            "active": normalized is not None and weight > 0,
        }
        if normalized is None or weight <= 0:
            continue
        numerator += normalized * float(weight)
        denominator += float(weight)
    if denominator <= 0:
        return None, breakdown
    for item in breakdown.values():
        item["effective_weight"] = 0.0
    for item in breakdown.values():
        if item["active"]:
            item["effective_weight"] = float(item["weight"] / denominator)
    return float(numerator / denominator), breakdown


def _coverage_ratio_risk(value: Any) -> float | None:
    if value is None:
        return None
    return float(min(abs(float(value) - 1.0), 1.0))


def build_online_quality_scores(metrics: dict[str, Any]) -> dict[str, Any]:
    semantic = metrics.get("semantic_instance_consistency") or {}
    height = metrics.get("height_consistency") or {}
    geometry = metrics.get("geometry_plausibility") or {}
    geometry_diag = metrics.get("geometry_diagnostics") or {}

    terms = {
        "coverage_ratio_risk": (_coverage_ratio_risk(semantic.get("coverage_ratio")), 0.35),
        "instance_leakage_risk": (_clamp01(semantic.get("instance_leakage")), 0.30),
        "semantic_gap_risk": (_clamp01(semantic.get("semantic_gap")), 0.25),
        "fragmentation_risk": (
            _clamp01(geometry_diag.get("fragmentation_score") or geometry.get("small_fragment_ratio_lt_4m2")),
            0.05,
        ),
        "merge_blob_risk": (
            _clamp01(geometry_diag.get("merge_blob_score") or geometry.get("max_instance_area_share")),
            0.03,
        ),
        "height_support_risk": (
            None
            if not height.get("available")
            else _clamp01(1.0 - float(height.get("instance_height_support_ratio") or 0.0)),
            0.02,
        ),
    }
    online_risk_score, breakdown = _weighted_valid_mean(terms)
    quality_score = None if online_risk_score is None else float(1.0 - online_risk_score)
    return {
        "online_risk_score": online_risk_score,
        "quality_score": quality_score,
        "risk_breakdown": breakdown,
    }


def evaluate_online_quality(
    *,
    inst_shp: str,
    m_sem_tif: str | None = None,
    chm_tif: str | None = None,
    patch_raster: str | None = None,
    quality_cfg: dict[str, Any] | None = None,
    reference_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = build_output_diagnostics(
        inst_shp=inst_shp,
        m_sem_tif=m_sem_tif,
        chm_tif=chm_tif,
        patch_raster=patch_raster,
        quality_cfg=quality_cfg,
        reference_metrics=reference_metrics,
    )
    metrics["geometry_diagnostics"] = build_geometry_diagnostics(metrics)
    score_payload = build_online_quality_scores(metrics)
    return {
        "assessment_mode": "online_output_quality",
        "metrics": metrics,
        "online_risk_score": score_payload["online_risk_score"],
        "quality_score": score_payload["quality_score"],
        "risk_breakdown": score_payload["risk_breakdown"],
    }
