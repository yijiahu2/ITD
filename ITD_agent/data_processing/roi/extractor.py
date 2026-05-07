from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, shapes
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from shapely.geometry import shape
from shapely.ops import unary_union

from ITD_agent.data_processing.vector import (
    crop_raster_to_geometry,
    enrich_xiaoban_clip_fields,
    standardize_inventory_crown_width,
)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(data: dict[str, Any], path: str | Path) -> str:
    out = _ensure_parent(path)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out)


def _normalize_ids(ids: list[str] | None) -> list[str]:
    return [str(x) for x in (ids or [])]


def _metric_crs(gdf: gpd.GeoDataFrame):
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS.")
    if getattr(gdf.crs, "is_projected", False):
        return gdf.crs
    utm = gdf.estimate_utm_crs()
    return utm if utm is not None else "EPSG:3857"


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _normalize_closure(value: Any) -> float | None:
    closure = _safe_float(value)
    if closure is None:
        return None
    if closure > 1.5:
        closure = closure / 100.0
    if closure < 0:
        return None
    return min(float(closure), 1.0)


def _box_mean(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return arr.astype(np.float32, copy=False)
    radius = int(radius)
    padded = np.pad(arr.astype(np.float32, copy=False), ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    k = 2 * radius + 1
    total = (
        integral[k:, k:]
        - integral[:-k, k:]
        - integral[k:, :-k]
        + integral[:-k, :-k]
    )
    return total / float(k * k)


def _read_raster_to_grid(
    raster_path: str | None,
    *,
    dst_crs,
    dst_transform,
    dst_height: int,
    dst_width: int,
    band: int = 1,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray | None:
    if not raster_path:
        return None
    path = Path(raster_path)
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        src_arr = src.read(band)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        return dst


def _read_rgb_gray_to_grid(
    raster_path: str,
    *,
    max_dim: int = 768,
) -> tuple[np.ndarray, Any, Any]:
    with rasterio.open(raster_path) as src:
        scale = max(float(src.width) / float(max_dim), float(src.height) / float(max_dim), 1.0)
        dst_width = max(int(math.ceil(src.width / scale)), 1)
        dst_height = max(int(math.ceil(src.height / scale)), 1)
        left, bottom, right, top = src.bounds
        dst_transform = from_bounds(left, bottom, right, top, dst_width, dst_height)
        if src.count >= 3:
            rgb = src.read(
                indexes=[1, 2, 3],
                out_shape=(3, dst_height, dst_width),
                resampling=Resampling.bilinear,
            ).astype(np.float32)
            gray = 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]
        else:
            gray = src.read(
                1,
                out_shape=(dst_height, dst_width),
                resampling=Resampling.bilinear,
            ).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            gray[np.isclose(gray, float(nodata))] = np.nan
        return gray, src.crs, dst_transform


def _robust_norm(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.float32)
    valid_mask = np.isfinite(arr)
    if mask is not None:
        valid_mask &= mask
    values = arr[valid_mask]
    if values.size == 0:
        return out
    lo = float(np.nanpercentile(values, 2))
    hi = float(np.nanpercentile(values, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def _instance_boundary_density(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int32, copy=False)
    boundary = np.zeros(labels.shape, dtype=np.float32)
    boundary[1:, :] = np.maximum(boundary[1:, :], (labels[1:, :] != labels[:-1, :]).astype(np.float32))
    boundary[:, 1:] = np.maximum(boundary[:, 1:], (labels[:, 1:] != labels[:, :-1]).astype(np.float32))
    boundary *= (labels > 0).astype(np.float32)
    return _box_mean(boundary, 5)


def _pixel_size_m(transform) -> float:
    try:
        res_x = abs(float(transform.a))
        res_y = abs(float(transform.e))
        if res_x > 0 and res_y > 0:
            return (res_x + res_y) / 2.0
    except Exception:
        pass
    return 1.0


def _mask_to_geometries(mask_arr: np.ndarray, *, transform) -> list[Any]:
    geoms = []
    for geom, value in shapes(mask_arr.astype(np.uint8), mask=mask_arr.astype(bool), transform=transform):
        if int(value) != 1:
            continue
        obj = shape(geom)
        if obj.is_empty:
            continue
        geoms.append(obj)
    return geoms


def _keep_top_connected_components(mask_arr: np.ndarray, *, max_components: int) -> np.ndarray:
    if max_components <= 0 or not np.any(mask_arr):
        return mask_arr
    work = mask_arr.astype(np.uint8, copy=True)
    geoms = _mask_to_geometries(work, transform=from_bounds(0, 0, work.shape[1], work.shape[0], work.shape[1], work.shape[0]))
    if len(geoms) <= max_components:
        return work
    areas = sorted(((geom.area, geom) for geom in geoms), key=lambda item: item[0], reverse=True)[:max_components]
    kept = np.zeros_like(work, dtype=np.uint8)
    for geom in [item[1] for item in areas]:
        geom_mask = ~geometry_mask([geom.__geo_interface__], out_shape=work.shape, transform=from_bounds(0, 0, work.shape[1], work.shape[0], work.shape[1], work.shape[0]), invert=False)
        kept[geom_mask] = 1
    return kept


def _load_prior_gdf(base_cfg: dict[str, Any]) -> tuple[gpd.GeoDataFrame | None, str | None, dict[str, Any]]:
    xiaoban_path = base_cfg.get("xiaoban_shp")
    prior_id_field = str(base_cfg.get("xiaoban_id_field") or "").strip() or None
    if not xiaoban_path or not Path(str(xiaoban_path)).exists() or not prior_id_field:
        return None, None, {}

    prior_gdf = gpd.read_file(str(xiaoban_path))
    if prior_gdf.crs is None or prior_id_field not in prior_gdf.columns:
        return None, None, {}

    prior_gdf = prior_gdf.copy()
    prior_gdf[prior_id_field] = prior_gdf[prior_id_field].astype(str)

    crown_field = str(base_cfg.get("crown_field") or "").strip()
    closure_field = str(base_cfg.get("closure_field") or "").strip()
    density_field = str(base_cfg.get("density_field") or "").strip()
    tree_count_field = str(base_cfg.get("tree_count_field") or "").strip()
    area_ha_field = str(base_cfg.get("area_ha_field") or "").strip()

    if crown_field and crown_field in prior_gdf.columns:
        prior_gdf["_expected_crown_width_m"] = prior_gdf[crown_field].apply(standardize_inventory_crown_width)
    else:
        prior_gdf["_expected_crown_width_m"] = np.nan

    if closure_field and closure_field in prior_gdf.columns:
        prior_gdf["_expected_closure"] = prior_gdf[closure_field].apply(_normalize_closure)
    else:
        prior_gdf["_expected_closure"] = np.nan

    if density_field and density_field in prior_gdf.columns:
        prior_gdf["_expected_density"] = prior_gdf[density_field].apply(_safe_float)
    elif tree_count_field and area_ha_field and tree_count_field in prior_gdf.columns and area_ha_field in prior_gdf.columns:
        tree_count = prior_gdf[tree_count_field].apply(_safe_float)
        area_ha = prior_gdf[area_ha_field].apply(_safe_float)
        prior_gdf["_expected_density"] = [
            (float(tc) / float(ah)) if tc is not None and ah not in (None, 0.0) else np.nan
            for tc, ah in zip(tree_count, area_ha)
        ]
    else:
        prior_gdf["_expected_density"] = np.nan

    metric_crs = _metric_crs(prior_gdf)
    metric_prior = prior_gdf.to_crs(metric_crs)
    scene_profile = {
        "crown_width_mean_m": float(np.nanmean(metric_prior["_expected_crown_width_m"])) if np.isfinite(metric_prior["_expected_crown_width_m"]).any() else None,
        "closure_mean": float(np.nanmean(metric_prior["_expected_closure"])) if np.isfinite(metric_prior["_expected_closure"]).any() else None,
        "density_mean": float(np.nanmean(metric_prior["_expected_density"])) if np.isfinite(metric_prior["_expected_density"]).any() else None,
    }
    return metric_prior, prior_id_field, scene_profile


def _classify_prior_structure_tag(
    *,
    crown_width_m: float | None,
    density_per_ha: float | None,
    closure: float | None,
) -> str:
    crown = _safe_float(crown_width_m)
    density = _safe_float(density_per_ha)
    closure_val = _safe_float(closure)

    if crown is not None and crown >= 8.0 and (density is None or density <= 700):
        return "very_large_open"
    if (density is not None and density >= 1600) or (closure_val is not None and closure_val >= 0.70):
        if crown is not None and crown <= 4.5:
            return "small_dense_closed"
        return "dense_closed"
    if crown is not None and crown >= 6.5:
        return "large_open" if closure_val is not None and closure_val <= 0.55 else "large_mixed"
    if density is not None and density <= 700:
        return "sparse_open"
    return "mixed_medium"


def _aggregate_prior_profile(
    geom_metric,
    *,
    prior_metric_gdf: gpd.GeoDataFrame | None,
    prior_id_field: str | None,
) -> dict[str, Any]:
    area_m2 = float(geom_metric.area)
    if prior_metric_gdf is None or not prior_id_field:
        return {
            "prior_overlap_ratio": 0.0,
            "prior_xiaoban_ids": [],
            "expected_crown_width_m": None,
            "expected_density": None,
            "expected_closure": None,
            "prior_structure_tag": "unknown",
        }

    inter = prior_metric_gdf[prior_metric_gdf.geometry.intersects(geom_metric)].copy()
    if inter.empty:
        return {
            "prior_overlap_ratio": 0.0,
            "prior_xiaoban_ids": [],
            "expected_crown_width_m": None,
            "expected_density": None,
            "expected_closure": None,
            "prior_structure_tag": "unknown",
        }

    inter["overlap_area_m2"] = inter.geometry.intersection(geom_metric).area
    inter = inter[inter["overlap_area_m2"] > 0].copy()
    if inter.empty:
        return {
            "prior_overlap_ratio": 0.0,
            "prior_xiaoban_ids": [],
            "expected_crown_width_m": None,
            "expected_density": None,
            "expected_closure": None,
            "prior_structure_tag": "unknown",
        }

    overlap_sum = float(inter["overlap_area_m2"].sum())
    weights = inter["overlap_area_m2"] / max(overlap_sum, 1.0e-6)

    def _weighted_mean(column: str) -> float | None:
        if column not in inter.columns:
            return None
        vals = np.asarray(inter[column], dtype=float)
        mask = np.isfinite(vals)
        if not np.any(mask):
            return None
        return float(np.average(vals[mask], weights=np.asarray(weights[mask], dtype=float)))

    crown = _weighted_mean("_expected_crown_width_m")
    density = _weighted_mean("_expected_density")
    closure = _weighted_mean("_expected_closure")
    return {
        "prior_overlap_ratio": overlap_sum / max(area_m2, 1.0e-6),
        "prior_xiaoban_ids": sorted(inter[prior_id_field].astype(str).tolist()),
        "expected_crown_width_m": crown,
        "expected_density": density,
        "expected_closure": closure,
        "prior_structure_tag": _classify_prior_structure_tag(
            crown_width_m=crown,
            density_per_ha=density,
            closure=closure,
        ),
    }


def _resolve_dynamic_min_area_m2(
    *,
    roi_cfg: dict[str, Any],
    prior_profile: dict[str, Any],
    signal_profile: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    base_min_area = float(roi_cfg.get("signal_min_area_m2", 150.0))
    cap_min_area = float(roi_cfg.get("signal_dynamic_min_area_cap_m2", max(base_min_area * 3.0, 450.0)))
    floor_ratio = float(roi_cfg.get("signal_dynamic_min_area_floor_ratio", 0.70))
    crown_width = _safe_float(prior_profile.get("expected_crown_width_m"))
    density = _safe_float(prior_profile.get("expected_density"))
    closure = _safe_float(prior_profile.get("expected_closure"))
    structure_tag = str(prior_profile.get("prior_structure_tag") or "unknown")

    if crown_width is None or crown_width <= 0:
        return base_min_area, {
            "strategy": "fallback_base_min_area",
            "structure_tag": structure_tag,
            "expected_crown_width_m": crown_width,
            "expected_density": density,
            "expected_closure": closure,
        }

    crown_area = math.pi * (float(crown_width) / 2.0) ** 2
    crown_count_floor = 3.5
    if structure_tag in {"very_large_open", "large_open"}:
        crown_count_floor = 3.0
    elif structure_tag in {"small_dense_closed", "dense_closed"}:
        crown_count_floor = 4.5
    elif structure_tag in {"sparse_open"}:
        crown_count_floor = 3.2

    complexity_boost = 1.0
    if float(signal_profile.get("shadow_score_mean") or 0.0) >= 0.45 or float(signal_profile.get("terrain_score_mean") or 0.0) >= 0.45:
        complexity_boost += 0.15
    if float(signal_profile.get("boundary_score_mean") or 0.0) >= 0.35:
        complexity_boost += 0.10
    if float(signal_profile.get("texture_score_mean") or 0.0) >= 0.40:
        complexity_boost += 0.05

    dynamic_area = crown_area * crown_count_floor * complexity_boost
    resolved = max(base_min_area * floor_ratio, dynamic_area)
    resolved = min(max(resolved, base_min_area * floor_ratio), cap_min_area)
    return resolved, {
        "strategy": "prior_guided_dynamic_min_area",
        "structure_tag": structure_tag,
        "expected_crown_width_m": crown_width,
        "expected_density": density,
        "expected_closure": closure,
        "expected_crown_area_m2": crown_area,
        "target_crown_count_floor": crown_count_floor,
        "complexity_boost": complexity_boost,
    }


def _dominant_signal_profile(summary: dict[str, Any]) -> tuple[str, list[str]]:
    signal_scores = {
        "boundary": float(summary.get("boundary_score_mean") or 0.0),
        "texture": float(summary.get("texture_score_mean") or 0.0),
        "shadow": float(summary.get("shadow_score_mean") or 0.0),
        "terrain": float(summary.get("terrain_score_mean") or 0.0),
    }
    ordered = sorted(signal_scores.items(), key=lambda item: item[1], reverse=True)
    dominant_signal = ordered[0][0] if ordered else "mixed"
    top_signals = [name for name, value in ordered if value >= max((ordered[0][1] if ordered else 0.0) * 0.80, 0.25)]

    canopy_fraction = float(summary.get("canopy_fraction") or 0.0)
    structure_tag = str(summary.get("prior_structure_tag") or "")
    if dominant_signal in {"shadow", "terrain"}:
        roi_signal_type = "shadow_topography"
    elif dominant_signal == "boundary":
        roi_signal_type = "boundary_fragmented"
    elif dominant_signal == "texture" and canopy_fraction >= 0.60:
        roi_signal_type = "dense_adhesion"
    elif structure_tag in {"very_large_open", "large_open", "large_mixed"}:
        roi_signal_type = "large_crown_complex"
    else:
        roi_signal_type = "mixed_complex"

    signal_tags = sorted(set(top_signals + [roi_signal_type, structure_tag]))
    return roi_signal_type, signal_tags


def _merge_candidate_records(
    records: list[dict[str, Any]],
    *,
    roi_cfg: dict[str, Any],
    grid_crs,
    grid_transform,
    score_map: np.ndarray,
    texture_map: np.ndarray,
    shadow_map: np.ndarray,
    terrain_map: np.ndarray,
    boundary_map: np.ndarray,
    canopy_map: np.ndarray,
    prior_metric_gdf: gpd.GeoDataFrame | None,
    prior_id_field: str | None,
    round_idx: int,
) -> list[dict[str, Any]]:
    if len(records) <= 1:
        return records

    merge_distance_default = float(roi_cfg.get("signal_same_type_merge_distance_m", 8.0))
    metric_crs = _metric_crs(gpd.GeoDataFrame({"candidate_id": [1]}, geometry=[shape({"type": "Point", "coordinates": [0, 0]})], crs=grid_crs))
    record_geoms = []
    for item in records:
        geom = shape({"type": "Polygon", "coordinates": []})
        try:
            from shapely import wkt

            geom = wkt.loads(str(item.get("geometry_wkt") or ""))
        except Exception:
            continue
        if geom.is_empty:
            continue
        metric_geom = gpd.GeoSeries([geom], crs=grid_crs).to_crs(metric_crs).iloc[0]
        record_geoms.append((item, metric_geom))
    if len(record_geoms) <= 1:
        return records

    visited: set[int] = set()
    merged: list[dict[str, Any]] = []
    for idx, (item, geom) in enumerate(record_geoms):
        if idx in visited:
            continue
        visited.add(idx)
        component = [idx]
        queue = [idx]
        while queue:
            current = queue.pop()
            current_item, current_geom = record_geoms[current]
            current_type = str(current_item.get("roi_signal_type") or "")
            current_ids = set(current_item.get("prior_xiaoban_ids") or [])
            current_merge_distance = max(
                float(current_item.get("merge_distance_m") or merge_distance_default),
                merge_distance_default,
            )
            for other_idx, (other_item, other_geom) in enumerate(record_geoms):
                if other_idx in visited:
                    continue
                other_type = str(other_item.get("roi_signal_type") or "")
                other_ids = set(other_item.get("prior_xiaoban_ids") or [])
                if current_type and other_type and current_type != other_type and not (current_ids & other_ids):
                    continue
                distance_m = float(current_geom.distance(other_geom))
                other_merge_distance = max(
                    float(other_item.get("merge_distance_m") or merge_distance_default),
                    merge_distance_default,
                )
                if distance_m <= max(current_merge_distance, other_merge_distance):
                    visited.add(other_idx)
                    queue.append(other_idx)
                    component.append(other_idx)

        if len(component) == 1:
            merged.append(record_geoms[component[0]][0])
            continue

        component_items = [record_geoms[i][0] for i in component]
        component_geoms = [record_geoms[i][1] for i in component]
        bridge_distance = max(
            [float(item.get("merge_distance_m") or merge_distance_default) for item in component_items] + [merge_distance_default]
        )
        buffered_union = unary_union([geom.buffer(bridge_distance / 2.0) for geom in component_geoms]).buffer(-bridge_distance / 2.0)
        merged_metric_geom = buffered_union if not buffered_union.is_empty else unary_union(component_geoms)
        merged_geom = gpd.GeoSeries([merged_metric_geom], crs=metric_crs).to_crs(grid_crs).iloc[0]
        merged_summary = _summarize_candidate_geometry(
            merged_geom,
            grid_crs=grid_crs,
            grid_transform=grid_transform,
            score_map=score_map,
            texture_map=texture_map,
            shadow_map=shadow_map,
            terrain_map=terrain_map,
            boundary_map=boundary_map,
            canopy_map=canopy_map,
            prior_metric_gdf=prior_metric_gdf,
            prior_id_field=prior_id_field,
            roi_cfg=roi_cfg,
        )
        merged_summary["candidate_id"] = f"signal_roi_{round_idx:02d}_{len(merged) + 1:02d}"
        merged_summary["merged_member_ids"] = [str(item.get("candidate_id") or "") for item in component_items]
        merged.append(merged_summary)

    return merged


def _build_global_candidate_mask(
    *,
    score_map: np.ndarray,
    valid_mask: np.ndarray,
    top_k: int,
    quantile: float,
    support_quantile: float,
    grow_radius_px: int,
    fill_radius_px: int,
) -> tuple[np.ndarray, list[float]]:
    quantiles_used: list[float] = []
    if not np.any(valid_mask):
        return np.zeros_like(valid_mask, dtype=np.uint8), quantiles_used

    union_mask = np.zeros_like(valid_mask, dtype=np.uint8)
    for seed_q in [quantile, 0.90, 0.88, 0.85]:
        seed_q = float(min(max(seed_q, 0.0), 0.999))
        support_q_eff = float(min(seed_q - 0.04, support_quantile)) if seed_q > support_quantile else float(support_quantile)
        seed_thr = float(np.nanquantile(score_map[valid_mask], seed_q))
        support_thr = float(np.nanquantile(score_map[valid_mask], support_q_eff))
        seed_mask = ((score_map >= seed_thr) & valid_mask).astype(np.uint8)
        if not np.any(seed_mask):
            continue
        support_mask = ((score_map >= support_thr) & valid_mask).astype(np.uint8)
        influence = (_box_mean(seed_mask.astype(np.float32), grow_radius_px) > 0.0).astype(np.uint8)
        grown_mask = ((support_mask > 0) & (influence > 0)).astype(np.uint8)
        if fill_radius_px > 0:
            grown_mask = ((_box_mean(grown_mask.astype(np.float32), fill_radius_px) >= 0.10) & (support_mask > 0)).astype(np.uint8)
        grown_mask = _keep_top_connected_components(grown_mask, max_components=max(top_k * 8, 16))
        if np.any(grown_mask):
            union_mask = np.maximum(union_mask, grown_mask)
            quantiles_used.append(seed_q)

    if not np.any(union_mask):
        return union_mask, quantiles_used

    union_mask = ((_box_mean(union_mask.astype(np.float32), max(1, fill_radius_px)) >= 0.08) & valid_mask).astype(np.uint8)
    union_mask = _keep_top_connected_components(union_mask, max_components=max(top_k * 8, 16))
    return union_mask, quantiles_used


def _summarize_candidate_geometry(
    geom,
    *,
    grid_crs,
    grid_transform,
    score_map: np.ndarray,
    texture_map: np.ndarray,
    shadow_map: np.ndarray,
    terrain_map: np.ndarray,
    boundary_map: np.ndarray,
    canopy_map: np.ndarray,
    prior_metric_gdf: gpd.GeoDataFrame | None,
    prior_id_field: str | None,
    roi_cfg: dict[str, Any],
) -> dict[str, Any]:
    geom_gdf = gpd.GeoDataFrame({"candidate_id": [1]}, geometry=[geom], crs=grid_crs)
    metric_crs = _metric_crs(geom_gdf)
    geom_metric = geom_gdf.to_crs(metric_crs)
    area_m2 = float(geom_metric.geometry.iloc[0].area)
    mask = ~geometry_mask([geom.__geo_interface__], out_shape=score_map.shape, transform=grid_transform, invert=False)
    score_vals = score_map[mask]
    texture_vals = texture_map[mask]
    shadow_vals = shadow_map[mask]
    terrain_vals = terrain_map[mask]
    boundary_vals = boundary_map[mask]
    canopy_vals = canopy_map[mask]
    prior_profile = _aggregate_prior_profile(
        geom_metric.geometry.iloc[0],
        prior_metric_gdf=prior_metric_gdf,
        prior_id_field=prior_id_field,
    )
    signal_profile = {
        "texture_score_mean": float(np.nanmean(texture_vals)) if texture_vals.size else 0.0,
        "shadow_score_mean": float(np.nanmean(shadow_vals)) if shadow_vals.size else 0.0,
        "terrain_score_mean": float(np.nanmean(terrain_vals)) if terrain_vals.size else 0.0,
        "boundary_score_mean": float(np.nanmean(boundary_vals)) if boundary_vals.size else 0.0,
    }
    dynamic_min_area_m2, dynamic_rule = _resolve_dynamic_min_area_m2(
        roi_cfg=roi_cfg,
        prior_profile=prior_profile,
        signal_profile=signal_profile,
    )
    summary = {
        "area_m2": area_m2,
        "bounds": [float(v) for v in geom.bounds],
        "score": float(np.nanmean(score_vals)) if score_vals.size else 0.0,
        "texture_score_mean": signal_profile["texture_score_mean"],
        "shadow_score_mean": signal_profile["shadow_score_mean"],
        "terrain_score_mean": signal_profile["terrain_score_mean"],
        "boundary_score_mean": signal_profile["boundary_score_mean"],
        "canopy_fraction": float(np.nanmean(canopy_vals)) if canopy_vals.size else 0.0,
        "prior_overlap_ratio": float(prior_profile.get("prior_overlap_ratio") or 0.0),
        "prior_xiaoban_ids": list(prior_profile.get("prior_xiaoban_ids") or []),
        "expected_crown_width_m": prior_profile.get("expected_crown_width_m"),
        "expected_density": prior_profile.get("expected_density"),
        "expected_closure": prior_profile.get("expected_closure"),
        "prior_structure_tag": str(prior_profile.get("prior_structure_tag") or "unknown"),
        "dynamic_min_area_m2": float(dynamic_min_area_m2),
        "dynamic_min_area_rule": dynamic_rule,
        "geometry_wkt": geom.wkt,
        "geometry_crs": str(grid_crs),
    }
    roi_signal_type, signal_tags = _dominant_signal_profile(summary)
    merge_distance_m = max(
        float(roi_cfg.get("signal_same_type_merge_distance_m", 8.0)),
        min(
            float(roi_cfg.get("signal_same_type_merge_distance_cap_m", 14.0)),
            max(
                float(roi_cfg.get("signal_same_type_merge_distance_floor_m", 4.0)),
                float(summary.get("expected_crown_width_m") or 0.0) * 0.85,
            ),
        ),
    )
    summary.update(
        {
            "roi_signal_type": roi_signal_type,
            "signal_tags": signal_tags,
            "merge_distance_m": merge_distance_m,
        }
    )
    return summary


def extract_signal_driven_roi_candidates(
    *,
    base_cfg: dict[str, Any],
    y_inst_tif: str | None,
    m_sem_tif: str | None,
    terrain_info: dict[str, Any],
    top_k: int,
    round_idx: int = 0,
) -> dict[str, Any]:
    roi_cfg = (((base_cfg.get("ITD_agent") or {}).get("planning") or {}).get("roi_extraction") or {})
    max_dim = int(roi_cfg.get("signal_grid_max_dim", 768))
    quantile = float(roi_cfg.get("signal_score_quantile", 0.92))
    buffer_m = float(roi_cfg.get("signal_buffer_m", roi_cfg.get("buffer_m", 5.0)))
    support_quantile = float(roi_cfg.get("signal_support_quantile", 0.80))
    out_root = _ensure_dir(Path(base_cfg["output_dir"]) / "data_processing" / "roi_signal_candidates")
    out_prefix = out_root / f"round_{int(round_idx):02d}"

    gray, grid_crs, grid_transform = _read_rgb_gray_to_grid(base_cfg["input_image"], max_dim=max_dim)
    height, width = gray.shape
    labels = _read_raster_to_grid(
        y_inst_tif,
        dst_crs=grid_crs,
        dst_transform=grid_transform,
        dst_height=height,
        dst_width=width,
        resampling=Resampling.nearest,
    )
    labels = np.nan_to_num(labels, nan=0.0).astype(np.int32) if labels is not None else np.zeros((height, width), dtype=np.int32)
    canopy = _read_raster_to_grid(
        m_sem_tif,
        dst_crs=grid_crs,
        dst_transform=grid_transform,
        dst_height=height,
        dst_width=width,
        resampling=Resampling.nearest,
    )
    canopy_mask = (np.nan_to_num(canopy, nan=0.0) > 0).astype(np.float32) if canopy is not None else (labels > 0).astype(np.float32)

    slope = _read_raster_to_grid(
        terrain_info.get("slope_tif"),
        dst_crs=grid_crs,
        dst_transform=grid_transform,
        dst_height=height,
        dst_width=width,
        resampling=Resampling.bilinear,
    )
    dem = _read_raster_to_grid(
        terrain_info.get("dem_tif"),
        dst_crs=grid_crs,
        dst_transform=grid_transform,
        dst_height=height,
        dst_width=width,
        resampling=Resampling.bilinear,
    )

    gray_norm = _robust_norm(gray)
    grad_y, grad_x = np.gradient(np.nan_to_num(gray_norm, nan=0.0))
    texture_map = _robust_norm(np.sqrt(grad_x ** 2 + grad_y ** 2), mask=np.isfinite(gray))
    shadow_map = np.clip(1.0 - gray_norm, 0.0, 1.0).astype(np.float32) * canopy_mask
    boundary_map = _instance_boundary_density(labels)
    crown_density = _box_mean((labels > 0).astype(np.float32), 7)

    terrain_components = []
    if slope is not None:
        terrain_components.append(_robust_norm(slope))
    if dem is not None:
        dem_norm = np.nan_to_num(dem, nan=np.nanmedian(dem) if np.isfinite(dem).any() else 0.0)
        dem_local = _box_mean(dem_norm.astype(np.float32), 7)
        terrain_components.append(_robust_norm(np.abs(dem_norm - dem_local)))
    if terrain_components:
        terrain_map = np.mean(np.stack(terrain_components, axis=0), axis=0).astype(np.float32)
    else:
        terrain_map = np.zeros((height, width), dtype=np.float32)

    score_map = (
        0.30 * boundary_map
        + 0.24 * texture_map
        + 0.18 * shadow_map
        + 0.16 * terrain_map
        + 0.12 * crown_density
    ).astype(np.float32)
    score_map *= np.clip(0.35 + 0.65 * canopy_mask, 0.0, 1.0)

    valid = canopy_mask > 0
    prior_metric_gdf, prior_id_field, prior_scene_profile = _load_prior_gdf(base_cfg)
    pixel_size_m = max(_pixel_size_m(grid_transform), 1.0e-6)
    expected_crown_width_m = _safe_float(prior_scene_profile.get("crown_width_mean_m"))
    expected_crown_px = (expected_crown_width_m / pixel_size_m) if expected_crown_width_m else None
    grow_radius_px = int(
        roi_cfg.get(
            "signal_seed_grow_radius_px",
            min(8, max(2, round((expected_crown_px or 10.0) * 0.25))),
        )
    )
    fill_radius_px = int(
        roi_cfg.get(
            "signal_fill_radius_px",
            min(5, max(1, round((expected_crown_px or 8.0) * 0.12))),
        )
    )
    candidate_mask, quantiles_used = _build_global_candidate_mask(
        score_map=score_map,
        valid_mask=valid,
        top_k=top_k,
        quantile=quantile,
        support_quantile=support_quantile,
        grow_radius_px=grow_radius_px,
        fill_radius_px=fill_radius_px,
    )
    geoms = _mask_to_geometries(candidate_mask, transform=grid_transform)
    candidate_threshold = float(np.nanquantile(score_map[valid], quantile)) if np.any(valid) else 1.0

    records: list[dict[str, Any]] = []
    for idx, geom in enumerate(geoms, 1):
        geom_gdf = gpd.GeoDataFrame({"candidate_id": [idx]}, geometry=[geom], crs=grid_crs)
        metric_crs = _metric_crs(geom_gdf)
        geom_metric = geom_gdf.to_crs(metric_crs)
        buffered = geom_metric.geometry.iloc[0].buffer(buffer_m)
        buffered_back = gpd.GeoSeries([buffered], crs=metric_crs).to_crs(grid_crs).iloc[0]
        summary = _summarize_candidate_geometry(
            buffered_back,
            grid_crs=grid_crs,
            grid_transform=grid_transform,
            score_map=score_map,
            texture_map=texture_map,
            shadow_map=shadow_map,
            terrain_map=terrain_map,
            boundary_map=boundary_map,
            canopy_map=canopy_mask,
            prior_metric_gdf=prior_metric_gdf,
            prior_id_field=prior_id_field,
            roi_cfg=roi_cfg,
        )
        if float(summary.get("area_m2") or 0.0) < float(summary.get("dynamic_min_area_m2") or 0.0):
            continue
        summary["candidate_id"] = f"signal_roi_{round_idx:02d}_{len(records) + 1:02d}"
        records.append(summary)

    records = _merge_candidate_records(
        records,
        roi_cfg=roi_cfg,
        grid_crs=grid_crs,
        grid_transform=grid_transform,
        score_map=score_map,
        texture_map=texture_map,
        shadow_map=shadow_map,
        terrain_map=terrain_map,
        boundary_map=boundary_map,
        canopy_map=canopy_mask,
        prior_metric_gdf=prior_metric_gdf,
        prior_id_field=prior_id_field,
        round_idx=round_idx,
    )
    records.sort(
        key=lambda item: (
            float(item.get("score") or 0.0)
            + 0.10 * float(item.get("prior_overlap_ratio") or 0.0)
            + 0.06 * float(item.get("boundary_score_mean") or 0.0)
            + 0.04 * float(item.get("terrain_score_mean") or 0.0)
        ),
        reverse=True,
    )
    max_keep = int(roi_cfg.get("signal_candidate_max_keep", max(top_k * 4, 8)))
    selected = records[: max(max_keep, 0)]
    selected_ids = {item["candidate_id"] for item in selected}

    if records:
        from shapely import wkt

        candidate_geoms = [wkt.loads(str(item["geometry_wkt"])) for item in records]
        candidate_gdf = gpd.GeoDataFrame(records, geometry=candidate_geoms, crs=grid_crs)
        candidate_gdf["selected"] = candidate_gdf["candidate_id"].isin(selected_ids)
        candidate_gdf.to_file(out_prefix.with_suffix(".gpkg"))
    else:
        candidate_gdf = gpd.GeoDataFrame({"candidate_id": []}, geometry=[], crs=grid_crs)

    summary = {
        "candidate_count": len(records),
        "selected_count": len(selected),
        "selection_mode": "signal_driven",
        "candidate_threshold": candidate_threshold,
        "signal_quantiles_used": quantiles_used,
        "signal_support_quantile": support_quantile,
        "signal_seed_grow_radius_px": grow_radius_px,
        "signal_fill_radius_px": fill_radius_px,
        "prior_scene_profile": prior_scene_profile,
        "grid_shape": [int(height), int(width)],
        "grid_crs": str(grid_crs),
        "signal_candidates": records,
        "selected_candidates": selected,
        "candidates_gpkg": str(out_prefix.with_suffix(".gpkg")) if records else None,
    }
    summary["summary_json"] = _write_json(summary, out_prefix.with_name(out_prefix.name + "_summary.json"))
    return summary


def make_bad_roi_gdf(
    *,
    xiaoban_shp: str,
    xiaoban_id_field: str,
    bad_ids: list[str],
    buffer_m: float = 5.0,
) -> gpd.GeoDataFrame:
    xgdf = gpd.read_file(xiaoban_shp)
    if xgdf.crs is None:
        raise ValueError("xiaoban shapefile has no CRS.")

    xgdf[xiaoban_id_field] = xgdf[xiaoban_id_field].astype(str)
    bad = xgdf[xgdf[xiaoban_id_field].isin([str(x) for x in bad_ids])].copy()
    if bad.empty:
        raise ValueError(f"No bad xiaoban found in shp. bad_ids={bad_ids}")

    if not bad.crs.is_projected:
        bad = bad.to_crs(bad.estimate_utm_crs())

    roi_union = unary_union(bad.geometry.tolist())
    return gpd.GeoDataFrame(
        {"roi_id": ["bad_roi"]},
        geometry=[roi_union.buffer(buffer_m)],
        crs=bad.crs,
    )


def make_bad_reference_unit_roi_gdf(
    *,
    reference_vector_path: str,
    reference_id_field: str,
    bad_reference_unit_ids: list[str],
    buffer_m: float = 5.0,
) -> gpd.GeoDataFrame:
    return make_bad_roi_gdf(
        xiaoban_shp=reference_vector_path,
        xiaoban_id_field=reference_id_field,
        bad_ids=bad_reference_unit_ids,
        buffer_m=buffer_m,
    )


def clip_xiaoban_to_geometry_with_fields(
    *,
    src_vector: str,
    geom_gdf: gpd.GeoDataFrame,
    out_vector: str,
    xiaoban_id_field: str,
    allowed_ids: list[str] | None = None,
    tree_count_field: str | None = None,
    crown_field: str | None = None,
    closure_field: str | None = None,
    area_ha_field: str | None = None,
    density_field: str | None = None,
) -> str:
    gdf = gpd.read_file(src_vector)
    if gdf.crs is None:
        raise ValueError(f"Vector has no CRS: {src_vector}")

    if allowed_ids:
        gdf[xiaoban_id_field] = gdf[xiaoban_id_field].astype(str)
        gdf = gdf[gdf[xiaoban_id_field].isin(_normalize_ids(allowed_ids))].copy()
        if gdf.empty:
            raise ValueError(f"No target xiaoban found after allowed_ids filter: {allowed_ids}")

    geom = geom_gdf.to_crs(gdf.crs)
    clipped = gpd.overlay(gdf, geom, how="intersection")
    clipped = clipped[clipped.geometry.notnull() & (~clipped.geometry.is_empty)].copy()
    if clipped.empty:
        raise ValueError(f"Clipped xiaoban is empty: {src_vector}")

    clipped = enrich_xiaoban_clip_fields(
        clipped_gdf=clipped,
        source_gdf=gdf,
        xiaoban_id_field=xiaoban_id_field,
        tree_count_field=tree_count_field,
        crown_field=crown_field,
        closure_field=closure_field,
        area_ha_field=area_ha_field,
        density_field=density_field,
    )

    out_path = _ensure_parent(out_vector)
    clipped.to_file(out_path)
    return str(out_path)


def clip_reference_vector_to_geometry_with_fields(
    *,
    src_vector: str,
    geom_gdf: gpd.GeoDataFrame,
    out_vector: str,
    reference_id_field: str,
    allowed_ids: list[str] | None = None,
    tree_count_field: str | None = None,
    crown_field: str | None = None,
    closure_field: str | None = None,
    area_ha_field: str | None = None,
    density_field: str | None = None,
) -> str:
    return clip_xiaoban_to_geometry_with_fields(
        src_vector=src_vector,
        geom_gdf=geom_gdf,
        out_vector=out_vector,
        xiaoban_id_field=reference_id_field,
        allowed_ids=allowed_ids,
        tree_count_field=tree_count_field,
        crown_field=crown_field,
        closure_field=closure_field,
        area_ha_field=area_ha_field,
        density_field=density_field,
    )


def crop_roi_terrain_bundle(
    *,
    roi_geom_gdf: gpd.GeoDataFrame,
    roi_dir: str | Path,
    dem_tif: str | None = None,
    slope_tif: str | None = None,
    aspect_tif: str | None = None,
    landform_tif: str | None = None,
    slope_position_tif: str | None = None,
) -> dict[str, str | None]:
    roi_dir = _ensure_dir(roi_dir)
    out = {
        "roi_dem_tif": None,
        "roi_slope_tif": None,
        "roi_aspect_tif": None,
        "roi_landform_tif": None,
        "roi_slope_position_tif": None,
    }
    if dem_tif:
        out["roi_dem_tif"] = crop_raster_to_geometry(dem_tif, roi_geom_gdf, roi_dir / "roi_dem.tif")
    if slope_tif:
        out["roi_slope_tif"] = crop_raster_to_geometry(slope_tif, roi_geom_gdf, roi_dir / "roi_slope.tif")
    if aspect_tif:
        out["roi_aspect_tif"] = crop_raster_to_geometry(aspect_tif, roi_geom_gdf, roi_dir / "roi_aspect.tif")
    if landform_tif:
        out["roi_landform_tif"] = crop_raster_to_geometry(landform_tif, roi_geom_gdf, roi_dir / "roi_landform.tif")
    if slope_position_tif:
        out["roi_slope_position_tif"] = crop_raster_to_geometry(slope_position_tif, roi_geom_gdf, roi_dir / "roi_slope_position.tif")
    return out


def prepare_roi_refinement_inputs(
    *,
    base_cfg: dict[str, Any],
    xiaoban_ids: list[str] | None,
    buffer_m: float,
    group_name: str,
    terrain_info: dict[str, Any],
    roi_geometry_wkt: str | None = None,
    roi_geometry_crs: str | None = None,
    roi_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    roi_cache_root = _ensure_dir(Path(base_cfg["output_dir"]) / "data_processing" / "roi_cache" / group_name)
    target_ids = _normalize_ids(xiaoban_ids)
    if roi_geometry_wkt:
        from shapely import wkt

        if not roi_geometry_crs:
            raise ValueError("roi_geometry_crs is required when roi_geometry_wkt is provided.")
        roi_gdf = gpd.GeoDataFrame(
            {"roi_id": [group_name]},
            geometry=[wkt.loads(roi_geometry_wkt)],
            crs=roi_geometry_crs,
        )
    else:
        roi_gdf = make_bad_roi_gdf(
            xiaoban_shp=base_cfg["xiaoban_shp"],
            xiaoban_id_field=base_cfg["xiaoban_id_field"],
            bad_ids=target_ids,
            buffer_m=buffer_m,
        )

    roi_geom_path = roi_cache_root / "roi_extent.gpkg"
    roi_gdf.to_file(roi_geom_path)

    roi_image = crop_raster_to_geometry(base_cfg["input_image"], roi_gdf, roi_cache_root / "roi_image.tif")
    roi_xiaoban = None
    if base_cfg.get("xiaoban_shp") and Path(str(base_cfg["xiaoban_shp"])).exists():
        roi_xiaoban = clip_xiaoban_to_geometry_with_fields(
            src_vector=base_cfg["xiaoban_shp"],
            geom_gdf=roi_gdf,
            out_vector=roi_cache_root / "roi_xiaoban.gpkg",
            xiaoban_id_field=base_cfg["xiaoban_id_field"],
            allowed_ids=target_ids if target_ids else None,
            tree_count_field=base_cfg.get("tree_count_field"),
            crown_field=base_cfg.get("crown_field"),
            closure_field=base_cfg.get("closure_field"),
            area_ha_field=base_cfg.get("area_ha_field"),
            density_field=base_cfg.get("density_field"),
        )
    terrain_roi_outputs = crop_roi_terrain_bundle(
        roi_geom_gdf=roi_gdf,
        roi_dir=roi_cache_root,
        dem_tif=terrain_info.get("dem_tif"),
        slope_tif=terrain_info.get("slope_tif"),
        aspect_tif=terrain_info.get("aspect_tif"),
        landform_tif=terrain_info.get("landform_tif"),
        slope_position_tif=terrain_info.get("slope_position_tif"),
    )
    metric_crs = _metric_crs(roi_gdf)
    roi_metric = roi_gdf.to_crs(metric_crs)
    roi_union = roi_metric.geometry.union_all()
    target_union = None
    output_ids: list[str] = []
    if roi_xiaoban:
        roi_xiaoban_gdf = gpd.read_file(roi_xiaoban).to_crs(metric_crs)
        output_ids = sorted(roi_xiaoban_gdf[base_cfg["xiaoban_id_field"]].astype(str).unique().tolist())
    if target_ids and base_cfg.get("xiaoban_shp") and Path(str(base_cfg["xiaoban_shp"])).exists():
        source_xiaoban = gpd.read_file(base_cfg["xiaoban_shp"])
        if source_xiaoban.crs is None:
            raise ValueError(f"Vector has no CRS: {base_cfg['xiaoban_shp']}")
        source_xiaoban[base_cfg["xiaoban_id_field"]] = source_xiaoban[base_cfg["xiaoban_id_field"]].astype(str)
        target_xiaoban = source_xiaoban[source_xiaoban[base_cfg["xiaoban_id_field"]].isin(target_ids)].copy()
        if not target_xiaoban.empty:
            target_union = target_xiaoban.to_crs(metric_crs).geometry.union_all()

    summary = {
        "group_name": group_name,
        "reference_id_field": base_cfg["xiaoban_id_field"],
        "target_reference_unit_ids": target_ids,
        "roi_reference_unit_ids_in_output": output_ids,
        "roi_reference_vector_gpkg": roi_xiaoban,
        "target_reference_unit_area_m2": float(target_union.area) if target_union is not None else None,
        "xiaoban_id_field": base_cfg["xiaoban_id_field"],
        "target_xiaoban_ids": target_ids,
        "roi_xiaoban_ids_in_output": output_ids,
        "buffer_m": float(buffer_m),
        "roi_source": "signal_driven_geometry" if roi_geometry_wkt else "inventory_buffer",
        "terrain_source": "dom_context_inherited",
        "terrain_layer_policy": terrain_info.get("terrain_layer_policy") or {},
        "roi_cache_root": str(roi_cache_root),
        "roi_extent_gpkg": str(roi_geom_path),
        "roi_image_tif": roi_image,
        "roi_xiaoban_gpkg": roi_xiaoban,
        "target_xiaoban_area_m2": float(target_union.area) if target_union is not None else None,
        "roi_extent_area_m2": float(roi_union.area),
        "roi_extra_area_m2": float(max(roi_union.area - target_union.area, 0.0)) if target_union is not None else None,
        "roi_extra_area_ratio": float(max(roi_union.area - target_union.area, 0.0) / target_union.area) if target_union is not None and target_union.area else None,
        "roi_boundary_offset_m": float(roi_union.boundary.distance(target_union)) if target_union is not None else None,
        "roi_metadata": roi_metadata or {},
        **terrain_roi_outputs,
    }
    summary["summary_json"] = _write_json(summary, roi_cache_root / "roi_extraction_summary.json")
    return summary
