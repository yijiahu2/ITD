from __future__ import annotations

from input_layer.common import RASTER_SUFFIXES, add_issue, path_exists, suffix
from input_layer.contracts import DEMSource, HeightRasterSource, ValidationIssue


def validate_dem_sources(items: list[DEMSource], issues: list[ValidationIssue]) -> None:
    for item in items:
        if not path_exists(item.path):
            level = "error" if item.required else "warning"
            add_issue(
                issues,
                level=level,
                code="missing_path",
                source_type="terrain_dem",
                source_id=item.id,
                message="DEM 路径不存在。",
                path=item.path,
            )
        if suffix(item.path) not in RASTER_SUFFIXES:
            add_issue(
                issues,
                level="warning",
                code="unexpected_suffix",
                source_type="terrain_dem",
                source_id=item.id,
                message="DEM 后缀不在常用栅格格式列表中。",
                path=item.path,
            )


def validate_height_rasters(items: list[HeightRasterSource], issues: list[ValidationIssue]) -> None:
    for item in items:
        source_type = "canopy_height" if item.role == "chm" else "surface_model"
        source_name = "CHM" if item.role == "chm" else "DSM"
        if not path_exists(item.path):
            level = "error" if item.required else "warning"
            add_issue(
                issues,
                level=level,
                code="missing_path",
                source_type=source_type,
                source_id=item.id,
                message=f"{source_name} 路径不存在。",
                path=item.path,
            )
        if suffix(item.path) not in RASTER_SUFFIXES:
            add_issue(
                issues,
                level="warning",
                code="unexpected_suffix",
                source_type=source_type,
                source_id=item.id,
                message=f"{source_name} 后缀不在常用栅格格式列表中。",
                path=item.path,
            )
