from __future__ import annotations

from input_layer.common import RASTER_SUFFIXES, add_issue, path_exists, suffix
from input_layer.contracts import RemoteSensingImageSource, ValidationIssue


def validate_remote_sensing_sources(
    items: list[RemoteSensingImageSource],
    issues: list[ValidationIssue],
) -> None:
    for item in items:
        if not path_exists(item.path):
            level = "error" if item.required else "warning"
            add_issue(
                issues,
                level=level,
                code="missing_path",
                source_type="remote_sensing",
                source_id=item.id,
                message="遥感影像路径不存在。",
                path=item.path,
            )
        if suffix(item.path) not in RASTER_SUFFIXES:
            add_issue(
                issues,
                level="warning",
                code="unexpected_suffix",
                source_type="remote_sensing",
                source_id=item.id,
                message="遥感影像后缀不在常用栅格格式列表中。",
                path=item.path,
            )
