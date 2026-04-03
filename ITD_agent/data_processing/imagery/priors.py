from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from input_layer.contracts import InputManifest

from ITD_agent.data_processing.contracts import ImagePriorProfile
from ITD_agent.data_processing.imagery.quality import estimate_quality, prepare_rgb_preview
from ITD_agent.data_processing.imagery.texture import estimate_texture
from ITD_agent.data_processing.imagery.tile_plan import build_tile_plan


def build_image_profiles(
    manifest: InputManifest,
    runtime_cfg: dict[str, object],
) -> list[ImagePriorProfile]:
    profiles: list[ImagePriorProfile] = []
    for item in manifest.remote_sensing:
        path = Path(item.path)
        if not path.exists():
            profiles.append(ImagePriorProfile(source_id=item.id, path=item.path, metadata={"status": "missing"}))
            continue
        with rasterio.open(path) as src:
            profile = src.profile.copy()
            bounds = src.bounds
            crs = str(src.crs) if src.crs else item.crs
            width = int(src.width)
            height = int(src.height)
            res_x = abs(float(src.transform.a))
            res_y = abs(float(src.transform.e))
            area_ha = float(width * height * res_x * res_y / 10000.0)
            nodata = src.nodata
            gray_preview = src.read(1, out_shape=(min(src.height, 512), min(src.width, 512))).astype(np.float32)
            gray_valid_mask = np.isfinite(gray_preview)
            if nodata is not None:
                gray_valid_mask &= ~np.isclose(gray_preview, nodata)
            rgb_preview, rgb_valid_mask = prepare_rgb_preview(src, nodata=nodata, max_dim=512)
            quality_summary = {
                "nodata": nodata,
                "bounds": {
                    "left": float(bounds.left),
                    "bottom": float(bounds.bottom),
                    "right": float(bounds.right),
                    "top": float(bounds.top),
                },
                "valid_pixel_ratio_estimate": float(np.mean(gray_valid_mask)) if gray_preview.size else 0.0,
                "quality_metrics": estimate_quality(rgb_preview, rgb_valid_mask),
            }
            texture_summary = estimate_texture(gray_preview, nodata=nodata)
            profiles.append(
                ImagePriorProfile(
                    source_id=item.id,
                    path=item.path,
                    width=width,
                    height=height,
                    crs=crs,
                    resolution_x_m=res_x,
                    resolution_y_m=res_y,
                    area_ha=area_ha,
                    band_count=int(src.count),
                    dtype=str(profile.get("dtype")),
                    quality_summary=quality_summary,
                    texture_summary=texture_summary,
                    tile_plan=build_tile_plan(width=width, height=height, res_x=res_x, res_y=res_y, runtime_cfg=runtime_cfg),
                    metadata={
                        "sensor": item.sensor,
                        "bands": item.bands,
                        "required": item.required,
                    },
                )
            )
    return profiles


__all__ = ["build_image_profiles"]
