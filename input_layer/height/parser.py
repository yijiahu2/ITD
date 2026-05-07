from __future__ import annotations

from pathlib import Path
from typing import Any

from input_layer.common import as_list, first_non_empty, resolve_path, safe_float
from input_layer.contracts import DEMSource, HeightRasterSource


def parse_dem_sources(
    terrain_cfg: dict[str, Any],
    cfg: dict[str, Any],
    config_dir: Path | None,
) -> list[DEMSource]:
    sources: list[DEMSource] = []
    raw_dem = terrain_cfg.get("dem")
    raw_items = as_list(raw_dem if isinstance(raw_dem, list) else [raw_dem] if isinstance(raw_dem, dict) else raw_dem)
    single_dem = first_non_empty(terrain_cfg.get("dem_tif"), cfg.get("dem_tif"))
    if single_dem and not raw_items:
        raw_items = [single_dem]
    elif single_dem and all(not isinstance(item, str) or item != single_dem for item in raw_items):
        raw_items = [single_dem] + raw_items

    for idx, item in enumerate(raw_items, 1):
        if isinstance(item, dict):
            path = resolve_path(item.get("path") or item.get("dem"), config_dir)
            if not path:
                continue
            sources.append(
                DEMSource(
                    id=str(item.get("id") or f"dem_{idx:03d}"),
                    path=path,
                    resolution_m=safe_float(item.get("resolution_m")),
                    crs=item.get("crs"),
                    vertical_unit=item.get("vertical_unit"),
                    required=bool(item.get("required", False)),
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k not in {"id", "path", "dem", "resolution_m", "crs", "vertical_unit", "required"}
                    },
                )
            )
            continue
        path = resolve_path(item, config_dir)
        if not path:
            continue
        sources.append(DEMSource(id=f"dem_{idx:03d}", path=path))
    return sources


def parse_height_rasters(
    block_cfg: dict[str, Any],
    cfg: dict[str, Any],
    config_dir: Path | None,
    *,
    role: str,
    fallback_keys: tuple[str, ...],
) -> list[HeightRasterSource]:
    sources: list[HeightRasterSource] = []
    raw_items = as_list(block_cfg.get("rasters"))
    primary_value = block_cfg.get(fallback_keys[0]) if fallback_keys else None
    if isinstance(primary_value, (list, tuple, dict)):
        raw_items = as_list(primary_value)
    scalar_fallback_values = []
    for key in fallback_keys:
        value = block_cfg.get(key)
        if not isinstance(value, (list, tuple, dict)):
            scalar_fallback_values.append(value)
    for key in fallback_keys:
        value = cfg.get(key)
        if not isinstance(value, (list, tuple, dict)):
            scalar_fallback_values.append(value)
    fallback = first_non_empty(*scalar_fallback_values)
    if fallback and not raw_items:
        raw_items = [fallback]
    elif fallback and all(not isinstance(item, str) or item != fallback for item in raw_items):
        raw_items = [fallback] + raw_items

    for idx, item in enumerate(raw_items, 1):
        if isinstance(item, dict):
            path = resolve_path(item.get("path") or item.get("raster"), config_dir)
            if not path:
                continue
            sources.append(
                HeightRasterSource(
                    id=str(item.get("id") or f"{role}_{idx:03d}"),
                    path=path,
                    role=role,
                    resolution_m=safe_float(item.get("resolution_m")),
                    crs=item.get("crs"),
                    vertical_unit=item.get("vertical_unit"),
                    required=bool(item.get("required", False)),
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k not in {"id", "path", "raster", "resolution_m", "crs", "vertical_unit", "required"}
                    },
                )
            )
            continue
        path = resolve_path(item, config_dir)
        if not path:
            continue
        sources.append(HeightRasterSource(id=f"{role}_{idx:03d}", path=path, role=role))
    return sources
