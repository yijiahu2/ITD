from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import box


def _equivalent_crown_width(area_m2: float) -> float:
    if area_m2 <= 0:
        return 0.0
    return 2.0 * ((area_m2 / np.pi) ** 0.5)


def _safe_quantile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _resolve_quality_cfg(quality_cfg: dict[str, Any] | None) -> dict[str, Any]:
    return dict(quality_cfg or {})


def _resolve_expected_crown_width(
    *,
    widths: np.ndarray,
    reference_metrics: dict[str, Any] | None,
    quality_cfg: dict[str, Any],
) -> tuple[float | None, str]:
    configured = _safe_float(quality_cfg.get("expected_mean_crown_width_m"))
    if configured is not None and configured > 0:
        return float(configured), "configured"
    referenced = _safe_float((reference_metrics or {}).get("expected_mean_crown_width"))
    if referenced is not None and referenced > 0:
        return float(referenced), "reference_metrics"
    p90_width = _safe_quantile(widths, 0.90)
    if p90_width is not None and p90_width > 0:
        return float(p90_width), "observed_p90_width"
    median_width = float(np.median(widths)) if widths.size > 0 else None
    if median_width is not None and median_width > 0:
        return float(median_width), "observed_median_width"
    return None, "unavailable"


def _resolve_expected_crown_area(
    *,
    areas: np.ndarray,
    expected_mean_crown_width_m: float | None,
    quality_cfg: dict[str, Any],
) -> tuple[float | None, str]:
    configured = _safe_float(quality_cfg.get("expected_crown_area_m2"))
    if configured is not None and configured > 0:
        return float(configured), "configured"
    if expected_mean_crown_width_m is not None and expected_mean_crown_width_m > 0:
        radius = expected_mean_crown_width_m / 2.0
        return float(np.pi * radius * radius), "derived_from_width"
    p10_area = _safe_quantile(areas, 0.10)
    if p10_area is not None and p10_area > 0:
        return float(p10_area), "observed_p10_area"
    median_area = float(np.median(areas)) if areas.size > 0 else None
    if median_area is not None and median_area > 0:
        return float(median_area), "observed_median_area"
    return None, "unavailable"


def _resolve_patch_context(
    *,
    patch_raster: str | None = None,
    fallback_raster: str | None = None,
) -> dict[str, Any]:
    raster_path = patch_raster if patch_raster and Path(patch_raster).exists() else fallback_raster
    if not raster_path or not Path(raster_path).exists():
        return {"available": False}

    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        pixel_area = abs(src.transform.a * src.transform.e - src.transform.b * src.transform.d)
        patch_area = float(src.width * src.height * pixel_area)
        pixel_resolution_m = float(np.sqrt(pixel_area)) if pixel_area > 0 else None
        geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    return {
        "available": True,
        "patch_raster": str(raster_path),
        "patch_area_m2": patch_area,
        "pixel_area_m2": pixel_area,
        "pixel_resolution_m": pixel_resolution_m,
        "patch_bounds": [float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)],
        "patch_crs": None,
        "patch_geometry": geom,
    }


def _load_instances(inst_shp: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(inst_shp)
    return gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)].copy()


def _load_instances_with_stats(inst_shp: str) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    raw_gdf = gpd.read_file(inst_shp)
    valid_gdf = raw_gdf[raw_gdf.geometry.notnull() & (~raw_gdf.geometry.is_empty)].copy()
    raw_count = int(len(raw_gdf))
    valid_count = int(len(valid_gdf))
    return valid_gdf, {
        "raw_feature_count": raw_count,
        "valid_instance_count": valid_count,
        "invalid_instance_count": int(max(raw_count - valid_count, 0)),
    }


def _semantic_instance_consistency(
    inst_gdf: gpd.GeoDataFrame,
    m_sem_tif: str | None,
    *,
    patch_area_m2: float | None = None,
) -> dict[str, Any]:
    if not m_sem_tif or not Path(m_sem_tif).exists():
        return {"available": False}

    with rasterio.open(m_sem_tif) as src:
        sem = src.read(1)
        sem_mask = sem > 0
        pixel_area = abs(src.transform.a * src.transform.e)
        semantic_area = float(sem_mask.sum() * pixel_area)
        shapes = [(geom, 1) for geom in inst_gdf.geometry] if not inst_gdf.empty else []
        inst_union = features.rasterize(
            shapes,
            out_shape=sem.shape,
            transform=src.transform,
            fill=0,
            dtype="uint8",
        )
        inst_mask = inst_union > 0
        instance_union_area = float(inst_mask.sum() * pixel_area)
        overlap_area = float(np.logical_and(sem_mask, inst_mask).sum() * pixel_area)

    eps = 1.0e-6
    semantic_empty = semantic_area <= eps
    instance_empty = instance_union_area <= eps
    instance_present_without_semantic = bool(semantic_empty and not instance_empty)

    coverage_ratio = None if semantic_empty else float(instance_union_area / semantic_area)
    semantic_recall = None if semantic_empty else float(overlap_area / semantic_area)
    instance_leakage = None if instance_empty else float(max(instance_union_area - overlap_area, 0.0) / instance_union_area)
    semantic_gap = None if semantic_empty else float(max(semantic_area - overlap_area, 0.0) / semantic_area)
    union_denominator = semantic_area + instance_union_area - overlap_area
    overlap_iou = None if union_denominator <= eps else float(overlap_area / union_denominator)

    result = {
        "available": True,
        "semantic_area": semantic_area,
        "instance_union_area": instance_union_area,
        "overlap_area": overlap_area,
        "coverage_ratio": coverage_ratio,
        "semantic_recall": semantic_recall,
        "instance_leakage": instance_leakage,
        "semantic_gap": semantic_gap,
        "overlap_iou": overlap_iou,
        "semantic_instance_iou": overlap_iou,
        "semantic_empty": bool(semantic_empty),
        "instance_empty": bool(instance_empty),
        "semantic_empty_flag": bool(semantic_empty),
        "instance_empty_flag": bool(instance_empty),
        "instance_present_without_semantic_flag": instance_present_without_semantic,
    }
    if patch_area_m2 is not None and patch_area_m2 > 0:
        semantic_cover_ratio = semantic_area / patch_area_m2
        instance_cover_ratio = instance_union_area / patch_area_m2
        result.update(
            {
                "patch_area_m2": float(patch_area_m2),
                "semantic_cover_ratio": semantic_cover_ratio,
                "instance_cover_ratio": instance_cover_ratio,
                "cover_ratio_delta_abs": abs(instance_cover_ratio - semantic_cover_ratio),
            }
        )
    return result


def _overlap_pair_stats(inst_gdf: gpd.GeoDataFrame) -> tuple[int, float]:
    if inst_gdf.empty or len(inst_gdf) < 2:
        return 0, 0.0

    sindex = inst_gdf.sindex
    pair_count = 0
    overlap_area = 0.0
    for idx, geom in enumerate(inst_gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        for cand in sindex.intersection(geom.bounds):
            if cand <= idx:
                continue
            other = inst_gdf.geometry.iloc[cand]
            if other is None or other.is_empty or not geom.intersects(other):
                continue
            inter_area = float(geom.intersection(other).area)
            if inter_area <= 0:
                continue
            pair_count += 1
            overlap_area += inter_area
    return pair_count, overlap_area


def _edge_touch_stats(inst_gdf: gpd.GeoDataFrame, patch_geometry, band_width_m: float = 0.5) -> tuple[int, float]:
    if inst_gdf.empty or patch_geometry is None or patch_geometry.is_empty or band_width_m <= 0:
        return 0, 0.0
    boundary_band = patch_geometry.boundary.buffer(band_width_m)
    if boundary_band.is_empty:
        return 0, 0.0
    edge_touch = inst_gdf.geometry.intersects(boundary_band)
    count = int(edge_touch.sum())
    ratio = float(count / max(len(inst_gdf), 1))
    return count, ratio


def _geometry_plausibility(
    inst_gdf: gpd.GeoDataFrame,
    *,
    patch_geometry=None,
    patch_context: dict[str, Any] | None = None,
    quality_cfg: dict[str, Any] | None = None,
    reference_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if inst_gdf.empty:
        return {"available": False}
    cfg = _resolve_quality_cfg(quality_cfg)
    areas = inst_gdf.geometry.area.astype(float).to_numpy()
    widths = np.asarray([_equivalent_crown_width(float(area)) for area in areas], dtype=float)
    union_area = float(inst_gdf.geometry.union_all().area) if len(inst_gdf) else 0.0
    sum_area = float(np.sum(areas))
    sorted_areas = np.sort(areas)[::-1]
    overlap_pair_count, overlap_area_total = _overlap_pair_stats(inst_gdf)
    edge_touch_count, edge_touch_ratio = _edge_touch_stats(inst_gdf, patch_geometry)
    expected_mean_crown_width_m, expected_width_source = _resolve_expected_crown_width(
        widths=widths,
        reference_metrics=reference_metrics,
        quality_cfg=cfg,
    )
    expected_crown_area_m2, expected_area_source = _resolve_expected_crown_area(
        areas=areas,
        expected_mean_crown_width_m=expected_mean_crown_width_m,
        quality_cfg=cfg,
    )
    pixel_resolution_m = _safe_float((patch_context or {}).get("pixel_resolution_m"))
    small_fragment_beta = float(cfg.get("small_fragment_beta", 0.25))
    tiny_width_beta = float(cfg.get("tiny_width_beta", 0.40))
    large_width_alpha = float(cfg.get("large_width_alpha", 2.0))
    min_width_resolution_factor = float(cfg.get("min_width_resolution_factor", 2.0))
    dominant_share_threshold = float(cfg.get("dominant_share_threshold", 0.35))
    min_width_by_resolution_m = None if pixel_resolution_m is None else float(pixel_resolution_m * min_width_resolution_factor)
    small_fragment_area_threshold_m2 = (
        None if expected_crown_area_m2 is None else float(expected_crown_area_m2 * small_fragment_beta)
    )
    tiny_width_threshold_m = None
    if expected_mean_crown_width_m is not None:
        tiny_width_threshold_m = float(expected_mean_crown_width_m * tiny_width_beta)
    if min_width_by_resolution_m is not None:
        tiny_width_threshold_m = max(float(tiny_width_threshold_m or 0.0), min_width_by_resolution_m)
    large_width_threshold_m = (
        None if expected_mean_crown_width_m is None else float(expected_mean_crown_width_m * large_width_alpha)
    )
    small_fragment_ratio_relative = (
        None if small_fragment_area_threshold_m2 is None else float(np.mean(areas < small_fragment_area_threshold_m2))
    )
    tiny_width_ratio_relative = (
        None if tiny_width_threshold_m is None else float(np.mean(widths < tiny_width_threshold_m))
    )
    width_outlier_ratio = (
        None if large_width_threshold_m is None else float(np.mean(widths > large_width_threshold_m))
    )
    eps = 1.0e-6
    max_instance_area_share = None if union_area <= eps else float(sorted_areas[0] / union_area)
    top5_instance_area_share = None if union_area <= eps else float(np.sum(sorted_areas[:5]) / union_area)
    return {
        "available": True,
        "instance_count": int(len(inst_gdf)),
        "sum_area_m2": sum_area,
        "union_area_m2": union_area,
        "sum_to_union_ratio": float(sum_area / max(union_area, 1.0e-6)),
        "mean_area_m2": float(np.mean(areas)),
        "median_area_m2": float(np.median(areas)),
        "p10_area_m2": _safe_quantile(areas, 0.10),
        "p90_area_m2": _safe_quantile(areas, 0.90),
        "mean_equivalent_crown_width_m": float(np.mean(widths)),
        "median_equivalent_crown_width_m": float(np.median(widths)),
        "p10_equivalent_crown_width_m": _safe_quantile(widths, 0.10),
        "p90_equivalent_crown_width_m": _safe_quantile(widths, 0.90),
        "small_fragment_ratio_lt_1m2": float(np.mean(areas < 1.0)),
        "small_fragment_ratio_lt_2m2": float(np.mean(areas < 2.0)),
        "small_fragment_ratio_lt_4m2": float(np.mean(areas < 4.0)),
        "small_fragment_ratio_lt_6m2": float(np.mean(areas < 6.0)),
        "tiny_width_ratio_lt_1m": float(np.mean(widths < 1.0)),
        "tiny_width_ratio_lt_2m": float(np.mean(widths < 2.0)),
        "large_width_ratio_gt_6m": float(np.mean(widths > 6.0)),
        "small_fragment_ratio_relative": small_fragment_ratio_relative,
        "small_fragment_area_threshold_m2": small_fragment_area_threshold_m2,
        "tiny_width_ratio_relative": tiny_width_ratio_relative,
        "tiny_width_threshold_m": tiny_width_threshold_m,
        "width_outlier_ratio": width_outlier_ratio,
        "large_width_threshold_m": large_width_threshold_m,
        "dominant_share_threshold": dominant_share_threshold,
        "expected_mean_crown_width_m": expected_mean_crown_width_m,
        "expected_mean_crown_width_source": expected_width_source,
        "expected_crown_area_m2": expected_crown_area_m2,
        "expected_crown_area_source": expected_area_source,
        "min_width_by_resolution_m": min_width_by_resolution_m,
        "max_instance_area_share": max_instance_area_share,
        "top5_instance_area_share": top5_instance_area_share,
        "overlap_pair_count": int(overlap_pair_count),
        "overlap_area_total_m2": float(overlap_area_total),
        "edge_touch_count": int(edge_touch_count),
        "edge_touch_ratio": float(edge_touch_ratio),
    }


def _height_consistency(inst_gdf: gpd.GeoDataFrame, chm_tif: str | None, *, quality_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    if not chm_tif or not Path(chm_tif).exists() or inst_gdf.empty:
        return {"available": False}

    cfg = _resolve_quality_cfg(quality_cfg)
    height_support_thr_m = float(cfg.get("height_support_thr_m", 1.0))
    with rasterio.open(chm_tif) as src:
        chm = src.read(1).astype(np.float32)
        valid = np.isfinite(chm)
        support_mask = valid & (chm > height_support_thr_m)
        pixel_area = abs(src.transform.a * src.transform.e)
        inst_union = features.rasterize(
            [(geom, 1) for geom in inst_gdf.geometry],
            out_shape=chm.shape,
            transform=src.transform,
            fill=0,
            dtype="uint8",
        )
        inst_mask = inst_union > 0
        instance_pixels = max(int(inst_mask.sum()), 1)
        support_pixels = int(np.logical_and(inst_mask, support_mask).sum())
        height_values = chm[np.logical_and(inst_mask, valid)]
        if height_values.size == 0:
            return {"available": False, "reason": "no_valid_height_values"}
        gy, gx = np.gradient(np.where(valid, chm, np.nanmean(height_values)))
        edge_strength = float(np.nanmean(np.sqrt(gx * gx + gy * gy)[inst_mask]))
        return {
            "available": True,
            "height_support_thr_m": height_support_thr_m,
            "instance_height_support_ratio": support_pixels / instance_pixels,
            "height_mean": float(np.mean(height_values)),
            "height_p95": float(np.percentile(height_values, 95)),
            "height_std": float(np.std(height_values)),
            "height_edge_strength": edge_strength,
            "support_area": float(support_pixels * pixel_area),
        }


def build_output_diagnostics(
    *,
    inst_shp: str,
    m_sem_tif: str | None = None,
    chm_tif: str | None = None,
    patch_raster: str | None = None,
    quality_cfg: dict[str, Any] | None = None,
    reference_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inst_gdf, instance_stats = _load_instances_with_stats(inst_shp)
    patch_context = _resolve_patch_context(patch_raster=patch_raster, fallback_raster=m_sem_tif)
    return {
        "patch_context": {key: value for key, value in patch_context.items() if key != "patch_geometry"},
        "instance_stats": instance_stats,
        "semantic_instance_consistency": _semantic_instance_consistency(
            inst_gdf,
            m_sem_tif,
            patch_area_m2=patch_context.get("patch_area_m2"),
        ),
        "height_consistency": _height_consistency(inst_gdf, chm_tif, quality_cfg=quality_cfg),
        "geometry_plausibility": _geometry_plausibility(
            inst_gdf,
            patch_geometry=patch_context.get("patch_geometry"),
            patch_context=patch_context,
            quality_cfg=quality_cfg,
            reference_metrics=reference_metrics,
        ),
    }


def rasterize_instances_to_label_raster(
    *,
    inst_shp: str,
    reference_raster: str,
    output_tif: str,
) -> str:
    out_path = Path(output_tif)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inst_gdf = _load_instances(inst_shp)
    with rasterio.open(reference_raster) as src:
        if not inst_gdf.empty and inst_gdf.crs is not None and src.crs is not None and inst_gdf.crs != src.crs:
            inst_gdf = inst_gdf.to_crs(src.crs)
        shapes = [(geom, idx) for idx, geom in enumerate(inst_gdf.geometry, 1)]
        labels = features.rasterize(
            shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="int32",
        )
        profile = src.profile.copy()
        profile.update(count=1, dtype="int32", nodata=0)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(labels, 1)
    return str(out_path)
