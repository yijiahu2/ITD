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

from ITD_agent.data_processing.inventory.normalizer import crop_raster_to_geometry, enrich_xiaoban_clip_fields


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
    prior_gdf: gpd.GeoDataFrame | None,
    prior_id_field: str | None,
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
    prior_overlap_ratio = 0.0
    prior_ids: list[str] = []
    if prior_gdf is not None and prior_id_field:
        metric_prior = prior_gdf.to_crs(metric_crs)
        inter = metric_prior[metric_prior.geometry.intersects(geom_metric.geometry.iloc[0])].copy()
        if not inter.empty:
            inter["overlap_area_m2"] = inter.geometry.intersection(geom_metric.geometry.iloc[0]).area
            inter = inter[inter["overlap_area_m2"] > 0]
            if not inter.empty:
                prior_ids = sorted(inter[prior_id_field].astype(str).tolist())
                prior_overlap_ratio = float(inter["overlap_area_m2"].sum() / max(area_m2, 1e-6))
    return {
        "area_m2": area_m2,
        "bounds": [float(v) for v in geom.bounds],
        "score": float(np.nanmean(score_vals)) if score_vals.size else 0.0,
        "texture_score_mean": float(np.nanmean(texture_vals)) if texture_vals.size else 0.0,
        "shadow_score_mean": float(np.nanmean(shadow_vals)) if shadow_vals.size else 0.0,
        "terrain_score_mean": float(np.nanmean(terrain_vals)) if terrain_vals.size else 0.0,
        "boundary_score_mean": float(np.nanmean(boundary_vals)) if boundary_vals.size else 0.0,
        "canopy_fraction": float(np.nanmean(canopy_vals)) if canopy_vals.size else 0.0,
        "prior_overlap_ratio": prior_overlap_ratio,
        "prior_xiaoban_ids": prior_ids,
        "geometry_wkt": geom.wkt,
        "geometry_crs": str(grid_crs),
    }


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
    min_area_m2 = float(roi_cfg.get("signal_min_area_m2", 150.0))
    buffer_m = float(roi_cfg.get("signal_buffer_m", roi_cfg.get("buffer_m", 5.0)))
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
    candidate_threshold = 1.0
    geoms = []
    for q in [quantile, 0.90, 0.88, 0.85, 0.80]:
        if not np.any(valid):
            break
        candidate_threshold = float(np.nanquantile(score_map[valid], q))
        candidate_mask = ((score_map >= candidate_threshold) & valid).astype(np.uint8)
        candidate_mask = _keep_top_connected_components(candidate_mask, max_components=max(top_k * 6, 12))
        geoms = _mask_to_geometries(candidate_mask, transform=grid_transform)
        if geoms:
            break

    prior_gdf = None
    prior_id_field = None
    if base_cfg.get("xiaoban_shp") and Path(str(base_cfg["xiaoban_shp"])).exists() and base_cfg.get("xiaoban_id_field"):
        prior_gdf = gpd.read_file(str(base_cfg["xiaoban_shp"]))
        if prior_gdf.crs is not None:
            prior_id_field = str(base_cfg["xiaoban_id_field"])
            prior_gdf[prior_id_field] = prior_gdf[prior_id_field].astype(str)

    records: list[dict[str, Any]] = []
    for idx, geom in enumerate(geoms, 1):
        geom_gdf = gpd.GeoDataFrame({"candidate_id": [idx]}, geometry=[geom], crs=grid_crs)
        metric_crs = _metric_crs(geom_gdf)
        geom_metric = geom_gdf.to_crs(metric_crs)
        buffered = geom_metric.geometry.iloc[0].buffer(buffer_m)
        if float(buffered.area) < min_area_m2:
            continue
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
            prior_gdf=prior_gdf,
            prior_id_field=prior_id_field,
        )
        summary["candidate_id"] = f"signal_roi_{round_idx:02d}_{len(records) + 1:02d}"
        records.append(summary)

    records.sort(
        key=lambda item: (
            float(item.get("score") or 0.0)
            + 0.08 * float(item.get("prior_overlap_ratio") or 0.0)
        ),
        reverse=True,
    )
    selected = records[: max(int(top_k), 0)]
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
