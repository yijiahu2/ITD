from __future__ import annotations

from typing import Any

from ITD_agent.data_processing.fusion.diagnostics import build_output_diagnostics

from .geometry_diagnostics import build_geometry_diagnostics


def score_online_quality(metrics: dict[str, Any]) -> float | None:
    semantic = metrics.get("semantic_instance_consistency") or {}
    height = metrics.get("height_consistency") or {}
    geometry = metrics.get("geometry_plausibility") or {}
    geometry_diag = metrics.get("geometry_diagnostics") or {}
    if not semantic.get("available"):
        return None
    score = 0.0
    score += abs(float(semantic.get("coverage_ratio") or 1.0) - 1.0) * 0.35
    score += float(semantic.get("instance_leakage") or 0.0) * 0.30
    score += float(semantic.get("semantic_gap") or 0.0) * 0.25
    score += float(geometry_diag.get("fragmentation_score") or geometry.get("small_fragment_ratio_lt_4m2") or 0.0) * 0.03
    score += float(geometry_diag.get("merge_blob_score") or geometry.get("max_instance_area_share") or 0.0) * 0.02
    if height.get("available"):
        score += max(0.0, 0.60 - float(height.get("instance_height_support_ratio") or 0.0)) * 0.05
    return float(score)


def evaluate_online_quality(
    *,
    inst_shp: str,
    m_sem_tif: str | None = None,
    chm_tif: str | None = None,
    patch_raster: str | None = None,
) -> dict[str, Any]:
    metrics = build_output_diagnostics(
        inst_shp=inst_shp,
        m_sem_tif=m_sem_tif,
        chm_tif=chm_tif,
        patch_raster=patch_raster,
    )
    metrics["geometry_diagnostics"] = build_geometry_diagnostics(metrics)
    return {
        "assessment_mode": "online_output_quality",
        "metrics": metrics,
        "quality_score": score_online_quality(metrics),
    }
