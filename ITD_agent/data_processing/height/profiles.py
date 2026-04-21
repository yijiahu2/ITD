from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import box

from input_layer.contracts import InputManifest

from ITD_agent.data_processing.contracts import HeightRasterProfile, ImagePriorProfile


def _height_summary(path: str | Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        valid = np.isfinite(data)
        nodata = src.nodata
        if nodata is not None and np.isfinite(float(nodata)):
            valid &= ~np.isclose(data, float(nodata))
        vals = data[valid]
        if vals.size == 0:
            return {"valid_pixel_count": 0, "available": False}
        gy, gx = np.gradient(np.where(valid, data, np.nanmean(vals)))
        edge = np.sqrt(gx * gx + gy * gy)
        support = valid & (data > 1.0)
        return {
            "available": True,
            "valid_pixel_count": int(vals.size),
            "valid_ratio": float(valid.mean()),
            "height_mean": float(np.mean(vals)),
            "height_p50": float(np.percentile(vals, 50)),
            "height_p95": float(np.percentile(vals, 95)),
            "height_max": float(np.max(vals)),
            "height_std": float(np.std(vals)),
            "height_edge_strength": float(np.nanmean(edge)),
            "canopy_support_ratio_gt_1m": float(support.mean()),
        }


def _alignment_with_image(src, image_profile: ImagePriorProfile | None) -> dict[str, Any]:
    if image_profile is None:
        return {"status": "no_reference_image", "recommended_action": "profile_only"}
    same_crs = str(src.crs) == str(image_profile.crs)
    res_x = abs(float(src.transform.a))
    res_y = abs(float(src.transform.e))
    same_resolution = False
    if image_profile.resolution_x_m is not None and image_profile.resolution_y_m is not None:
        same_resolution = abs(res_x - float(image_profile.resolution_x_m)) < 1e-6 and abs(res_y - float(image_profile.resolution_y_m)) < 1e-6
    img_bounds = (image_profile.quality_summary or {}).get("bounds") or {}
    overlap_status = "unknown"
    if img_bounds:
        geom = box(float(img_bounds["left"]), float(img_bounds["bottom"]), float(img_bounds["right"]), float(img_bounds["top"]))
        geom_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=image_profile.crs)
        try:
            if src.crs is not None:
                geom_in_src = geom_gdf.to_crs(src.crs)
                overlap = geom_in_src.geometry.iloc[0].intersects(box(*src.bounds))
            else:
                overlap = False
        except Exception:
            overlap = False
        overlap_status = "overlap" if overlap else "disjoint"
    return {
        "same_crs": same_crs,
        "same_resolution": same_resolution,
        "overlap_status": overlap_status,
        "recommended_action": (
            "crop_to_dom_extent"
            if same_crs and overlap_status == "overlap"
            else "reproject_then_crop_to_dom_extent"
            if overlap_status == "overlap"
            else "profile_only"
        ),
    }


def _crop_to_image_extent(
    *,
    src_path: str,
    image_profile: ImagePriorProfile,
    out_path: Path,
) -> str | None:
    bounds = (image_profile.quality_summary or {}).get("bounds") or {}
    if not bounds:
        return None
    with rasterio.open(src_path) as src:
        if src.crs is None:
            return None
        geom = box(float(bounds["left"]), float(bounds["bottom"]), float(bounds["right"]), float(bounds["top"]))
        geom_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=image_profile.crs).to_crs(src.crs)
        raster_bounds = box(*src.bounds)
        if not geom_gdf.geometry.iloc[0].intersects(raster_bounds):
            return None
        data, transform = mask(src, [geom_gdf.geometry.iloc[0].__geo_interface__], crop=True, filled=True)
        profile = src.profile.copy()
        profile.update(height=data.shape[1], width=data.shape[2], transform=transform, compress="LZW")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
    return str(out_path)


def build_height_raster_profiles(
    manifest: InputManifest,
    image_profiles: list[ImagePriorProfile],
    storage_layout: dict[str, str],
) -> list[HeightRasterProfile]:
    first_image = image_profiles[0] if image_profiles else None
    raster_cache = Path(storage_layout["raster_cache"])
    profiles: list[HeightRasterProfile] = []
    sources = [(item, "chm") for item in manifest.canopy_height] + [(item, "dsm") for item in manifest.surface_models]
    for item, role in sources:
        path = Path(item.path)
        if not path.exists():
            profiles.append(HeightRasterProfile(source_id=item.id, path=item.path, role=role, metadata={"status": "missing"}))
            continue
        with rasterio.open(path) as src:
            res_x = abs(float(src.transform.a))
            res_y = abs(float(src.transform.e))
            area_ha = float(src.width * src.height * res_x * res_y / 10000.0)
            alignment = _alignment_with_image(src, first_image)
        cropped_path = None
        if first_image is not None and alignment.get("overlap_status") == "overlap":
            cropped_path = _crop_to_image_extent(
                src_path=str(path),
                image_profile=first_image,
                out_path=raster_cache / f"{path.stem}_{role}_dom_extent.tif",
            )
        normalized_path = cropped_path or str(path)
        with rasterio.open(path) as src:
            profile = HeightRasterProfile(
                source_id=item.id,
                path=item.path,
                role=role,
                crs=str(src.crs) if src.crs else item.crs,
                width=int(src.width),
                height=int(src.height),
                resolution_x_m=res_x,
                resolution_y_m=res_y,
                area_ha=area_ha,
                alignment_with_image=alignment,
                dom_cropped_path=cropped_path,
                height_summary=_height_summary(normalized_path),
                normalization={
                    "status": "dom_extent_cropped" if cropped_path else "profile_only",
                    "normalized_path": normalized_path,
                    "target_grid": "dom_extent",
                    "resampled_to_dom_resolution": False,
                },
                metadata={"required": item.required, "vertical_unit": item.vertical_unit},
            )
        profiles.append(profile)
    return profiles
