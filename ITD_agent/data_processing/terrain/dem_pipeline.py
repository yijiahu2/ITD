from __future__ import annotations

from pathlib import Path
from typing import Any

import rasterio

from input_layer.contracts import InputManifest

from ITD_agent.data_processing.contracts import DEMProcessingProfile, ImagePriorProfile
from ITD_agent.data_processing.terrain.constraints import (
    LANDFORM_CODE,
    SLOPE_POSITION_CODE,
    TerrainRuleConfig,
    classify_aspect_class,
    classify_aspect_class_cn,
    classify_landform_type,
    classify_landform_type_cn,
    classify_slope_class,
    classify_slope_class_cn,
    classify_slope_position_class,
    classify_slope_position_class_cn,
    circular_mean_deg,
    dominant_class,
    encode_class_to_int,
    normalize_aspect_deg,
    safe_float,
    summarize_terrain_classes,
)
from ITD_agent.data_processing.terrain.features import generate_terrain_products


def _alignment_with_image(dem_src, image_profile: ImagePriorProfile | None) -> dict[str, Any]:
    if image_profile is None:
        return {"status": "no_reference_image"}
    same_crs = str(dem_src.crs) == str(image_profile.crs)
    res_x = abs(float(dem_src.transform.a))
    res_y = abs(float(dem_src.transform.e))
    same_resolution = False
    if image_profile.resolution_x_m is not None and image_profile.resolution_y_m is not None:
        same_resolution = abs(res_x - float(image_profile.resolution_x_m)) < 1e-6 and abs(res_y - float(image_profile.resolution_y_m)) < 1e-6
    dem_bounds = dem_src.bounds
    img_bounds = (image_profile.quality_summary or {}).get("bounds") or {}
    overlap_status = "unknown"
    if img_bounds:
        overlap = not (
            dem_bounds.right <= float(img_bounds["left"])
            or dem_bounds.left >= float(img_bounds["right"])
            or dem_bounds.top <= float(img_bounds["bottom"])
            or dem_bounds.bottom >= float(img_bounds["top"])
        )
        overlap_status = "overlap" if overlap else "disjoint"
    return {
        "same_crs": same_crs,
        "same_resolution": same_resolution,
        "overlap_status": overlap_status,
        "recommended_action": (
            "reproject_and_crop_to_image_extent"
            if not same_crs or not same_resolution or overlap_status != "overlap"
            else "crop_to_image_extent"
        ),
    }


def build_dem_profiles(
    manifest: InputManifest,
    image_profiles: list[ImagePriorProfile],
    terrain_info: dict[str, Any],
) -> list[DEMProcessingProfile]:
    first_image = image_profiles[0] if image_profiles else None
    profiles: list[DEMProcessingProfile] = []
    for item in manifest.terrain_dem:
        path = Path(item.path)
        if not path.exists():
            profiles.append(DEMProcessingProfile(source_id=item.id, path=item.path, metadata={"status": "missing"}))
            continue
        with rasterio.open(path) as src:
            res_x = abs(float(src.transform.a))
            res_y = abs(float(src.transform.e))
            area_ha = float(src.width * src.height * res_x * res_y / 10000.0)
            profiles.append(
                DEMProcessingProfile(
                    source_id=item.id,
                    path=item.path,
                    crs=str(src.crs) if src.crs else item.crs,
                    width=int(src.width),
                    height=int(src.height),
                    resolution_x_m=res_x,
                    resolution_y_m=res_y,
                    area_ha=area_ha,
                    alignment_with_image=_alignment_with_image(src, first_image),
                    terrain_products={
                        "dem_tif": terrain_info.get("dem_tif"),
                        "slope_tif": terrain_info.get("slope_tif"),
                        "aspect_tif": terrain_info.get("aspect_tif"),
                        "landform_tif": terrain_info.get("landform_tif"),
                        "slope_position_tif": terrain_info.get("slope_position_tif"),
                        "global_dem_tif": terrain_info.get("global_dem_tif"),
                        "global_slope_tif": terrain_info.get("global_slope_tif"),
                        "global_aspect_tif": terrain_info.get("global_aspect_tif"),
                        "global_landform_tif": terrain_info.get("global_landform_tif"),
                        "global_slope_position_tif": terrain_info.get("global_slope_position_tif"),
                        "global_terrain_background": terrain_info.get("global_terrain_background") or {},
                        "dom_terrain_context": terrain_info.get("dom_terrain_context") or {},
                        "terrain_layer_policy": terrain_info.get("terrain_layer_policy") or {},
                        "terrain_generated": terrain_info.get("terrain_generated", False),
                    },
                    crop_strategy={
                        "mode": "align_with_image_and_crop",
                        "shared_tile_index": True,
                    },
                    metadata={
                        "vertical_unit": item.vertical_unit,
                        "required": item.required,
                    },
                )
            )
    return profiles


__all__ = [
    "LANDFORM_CODE",
    "SLOPE_POSITION_CODE",
    "TerrainRuleConfig",
    "build_dem_profiles",
    "classify_aspect_class",
    "classify_aspect_class_cn",
    "classify_landform_type",
    "classify_landform_type_cn",
    "classify_slope_class",
    "classify_slope_class_cn",
    "classify_slope_position_class",
    "classify_slope_position_class_cn",
    "circular_mean_deg",
    "dominant_class",
    "encode_class_to_int",
    "generate_terrain_products",
    "normalize_aspect_deg",
    "safe_float",
    "summarize_terrain_classes",
]
