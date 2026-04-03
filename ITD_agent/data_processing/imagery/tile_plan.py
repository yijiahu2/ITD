from __future__ import annotations

from typing import Any

from ITD_agent.data_processing.contracts import RasterTilePlan
from ITD_agent.data_processing.imagery.common import safe_float


def build_tile_plan(*, width: int, height: int, res_x: float | None, res_y: float | None, runtime_cfg: dict[str, Any]) -> RasterTilePlan:
    dp_cfg = ((runtime_cfg.get("ITD_agent") or {}).get("data_processing") or {})
    image_policy = dp_cfg.get("image_policy") or {}
    pixel_count = int(width * height)
    tile_size = int(runtime_cfg.get("tile", 2048) or 2048)
    overlap = int(runtime_cfg.get("overlap", 512) or 512)
    tile_overlap = safe_float(runtime_cfg.get("tile_overlap"))
    area_m2 = None
    if res_x and res_y:
        area_m2 = float(width * height * abs(res_x) * abs(res_y))

    max_direct_pixels = int(image_policy.get("max_direct_pixels", 30_000_000))
    max_direct_area_ha = safe_float(image_policy.get("max_direct_area_ha"))
    if max_direct_area_ha is None:
        max_direct_area_ha = 25.0
    requires_sliding = pixel_count > max_direct_pixels
    requires_crop = area_m2 is not None and area_m2 > float(max_direct_area_ha) * 10000.0
    if requires_crop and requires_sliding:
        mode = "geometry_crop_then_sliding_window"
        reason = "影像面积和像素规模都较大，先裁剪再滑窗。"
    elif requires_crop:
        mode = "geometry_crop"
        reason = "影像范围较大，优先按 ROI/小班几何裁剪。"
    elif requires_sliding:
        mode = "sliding_window"
        reason = "影像像素规模较大，适合直接滑窗。"
    else:
        mode = "direct"
        reason = "影像规模适中，可直接处理。"

    estimated_tile_count = None
    if requires_sliding and tile_size > 0:
        step = max(tile_size - overlap, 1)
        nx = max(((width - overlap - 1) // step) + 1, 1)
        ny = max(((height - overlap - 1) // step) + 1, 1)
        estimated_tile_count = int(nx * ny)
    return RasterTilePlan(
        mode=mode,
        requires_geometry_crop=requires_crop,
        requires_sliding_window=requires_sliding,
        reason=reason,
        tile_size=tile_size,
        overlap=overlap,
        tile_overlap_ratio=tile_overlap,
        estimated_tile_count=estimated_tile_count,
    )
