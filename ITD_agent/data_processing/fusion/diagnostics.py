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
        geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    return {
        "available": True,
        "patch_raster": str(raster_path),
        "patch_area_m2": patch_area,
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
    if not m_sem_tif or not Path(m_sem_tif).exists() or inst_gdf.empty:
        return {"available": False}

    with rasterio.open(m_sem_tif) as src:
        sem = src.read(1)
        sem_mask = sem > 0
        pixel_area = abs(src.transform.a * src.transform.e)
        semantic_area = float(sem_mask.sum() * pixel_area)
        inst_union = features.rasterize(
            [(geom, 1) for geom in inst_gdf.geometry],
            out_shape=sem.shape,
            transform=src.transform,
            fill=0,
            dtype="uint8",
        )
        inst_mask = inst_union > 0
        instance_union_area = float(inst_mask.sum() * pixel_area)
        overlap_area = float(np.logical_and(sem_mask, inst_mask).sum() * pixel_area)

    coverage_ratio = instance_union_area / max(semantic_area, 1.0e-6)
    semantic_recall = overlap_area / max(semantic_area, 1.0e-6)
    instance_leakage = max(instance_union_area - overlap_area, 0.0) / max(instance_union_area, 1.0e-6)
    semantic_gap = max(semantic_area - overlap_area, 0.0) / max(semantic_area, 1.0e-6)
    union_denominator = max(semantic_area + instance_union_area - overlap_area, 1.0e-6)
    overlap_iou = overlap_area / union_denominator
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
) -> dict[str, Any]:
    if inst_gdf.empty:
        return {"available": False}
    areas = inst_gdf.geometry.area.astype(float).to_numpy()
    widths = np.asarray([_equivalent_crown_width(float(area)) for area in areas], dtype=float)
    union_area = float(inst_gdf.geometry.union_all().area) if len(inst_gdf) else 0.0
    sum_area = float(np.sum(areas))
    sorted_areas = np.sort(areas)[::-1]
    overlap_pair_count, overlap_area_total = _overlap_pair_stats(inst_gdf)
    edge_touch_count, edge_touch_ratio = _edge_touch_stats(inst_gdf, patch_geometry)
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
        "small_fragment_ratio_lt_1m2": float(np.mean(areas < 1.0)),
        "small_fragment_ratio_lt_2m2": float(np.mean(areas < 2.0)),
        "small_fragment_ratio_lt_4m2": float(np.mean(areas < 4.0)),
        "small_fragment_ratio_lt_6m2": float(np.mean(areas < 6.0)),
        "tiny_width_ratio_lt_1m": float(np.mean(widths < 1.0)),
        "tiny_width_ratio_lt_2m": float(np.mean(widths < 2.0)),
        "large_width_ratio_gt_6m": float(np.mean(widths > 6.0)),
        "max_instance_area_share": float(sorted_areas[0] / max(union_area, 1.0e-6)),
        "top5_instance_area_share": float(np.sum(sorted_areas[:5]) / max(union_area, 1.0e-6)),
        "overlap_pair_count": int(overlap_pair_count),
        "overlap_area_total_m2": float(overlap_area_total),
        "edge_touch_count": int(edge_touch_count),
        "edge_touch_ratio": float(edge_touch_ratio),
    }


def _height_consistency(inst_gdf: gpd.GeoDataFrame, chm_tif: str | None) -> dict[str, Any]:
    if not chm_tif or not Path(chm_tif).exists() or inst_gdf.empty:
        return {"available": False}

    with rasterio.open(chm_tif) as src:
        chm = src.read(1).astype(np.float32)
        valid = np.isfinite(chm)
        support_mask = valid & (chm > 1.0)
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
        "height_consistency": _height_consistency(inst_gdf, chm_tif),
        "geometry_plausibility": _geometry_plausibility(
            inst_gdf,
            patch_geometry=patch_context.get("patch_geometry"),
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
