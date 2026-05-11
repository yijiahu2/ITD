from __future__ import annotations

from pathlib import Path
from typing import Any

from rasterio.windows import Window


def materialize_expert_tile(
    *,
    image_path: str | Path,
    tile_window_px: list[float],
    output_dir: str | Path,
    image_id: str,
) -> dict[str, Any]:
    src_path = Path(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = _window_xyxy(tile_window_px)
    tile_path = out_dir / f"expert_tile_{image_id}_{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}.tif"
    try:
        import rasterio

        with rasterio.open(src_path) as src:
            width = max(1, int(round(x2 - x1)))
            height = max(1, int(round(y2 - y1)))
            window = Window(col_off=int(round(x1)), row_off=int(round(y1)), width=width, height=height)
            data = src.read(window=window)
            profile = src.profile.copy()
            profile.update(
                {
                    "height": data.shape[1],
                    "width": data.shape[2],
                    "transform": src.window_transform(window),
                }
            )
            with rasterio.open(tile_path, "w", **profile) as dst:
                dst.write(data)
        return {
            "status": "materialized",
            "tile_image_path": str(tile_path),
            "offset_xy": [x1, y1],
            "tile_window_px": [x1, y1, x2, y2],
            "width": width,
            "height": height,
        }
    except Exception as exc:
        return {
            "status": "fallback_full_image",
            "reason": str(exc),
            "tile_image_path": str(src_path),
            "offset_xy": [0.0, 0.0],
            "tile_window_px": [x1, y1, x2, y2],
        }


def offset_instances_to_full_image(instances: list[dict[str, Any]], offset_xy: list[float]) -> list[dict[str, Any]]:
    offset_x = float(offset_xy[0] if offset_xy else 0.0)
    offset_y = float(offset_xy[1] if len(offset_xy) > 1 else 0.0)
    adjusted: list[dict[str, Any]] = []
    for instance in instances:
        item = dict(instance)
        bbox = list(item.get("bbox") or item.get("bbox_px") or [])
        if len(bbox) >= 4:
            bbox[0] = float(bbox[0]) + offset_x
            bbox[1] = float(bbox[1]) + offset_y
            item["bbox"] = bbox
        item["tile_offset_xy"] = [offset_x, offset_y]
        adjusted.append(item)
    return adjusted


def _window_xyxy(tile_window_px: list[float]) -> tuple[float, float, float, float]:
    if len(tile_window_px) < 4:
        return (0.0, 0.0, 1.0, 1.0)
    x1, y1, x2, y2 = [float(value) for value in tile_window_px[:4]]
    return (x1, y1, max(x1 + 1.0, x2), max(y1 + 1.0, y2))
