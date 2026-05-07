from __future__ import annotations

from pathlib import Path
from typing import Any

from input_layer.common import as_list, as_string_list, first_non_empty, resolve_path, safe_float
from input_layer.contracts import RemoteSensingImageSource


def parse_remote_sensing_sources(
    remote_sensing_cfg: dict[str, Any],
    cfg: dict[str, Any],
    config_dir: Path | None,
) -> list[RemoteSensingImageSource]:
    sources: list[RemoteSensingImageSource] = []
    seen_paths: set[str] = set()
    raw_items = as_list(remote_sensing_cfg.get("images"))
    first_image = first_non_empty(
        remote_sensing_cfg.get("image"),
        remote_sensing_cfg.get("rgb_image"),
        cfg.get("input_image"),
    )
    if first_image and not raw_items:
        raw_items = [first_image]
    elif first_image:
        raw_items = [first_image] + raw_items

    for idx, item in enumerate(raw_items, 1):
        if isinstance(item, dict):
            path = resolve_path(item.get("path") or item.get("image"), config_dir)
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            sources.append(
                RemoteSensingImageSource(
                    id=str(item.get("id") or f"image_{idx:03d}"),
                    path=path,
                    sensor=item.get("sensor"),
                    resolution_m=safe_float(item.get("resolution_m")),
                    crs=item.get("crs"),
                    bands=as_string_list(item.get("bands")),
                    nodata=item.get("nodata"),
                    acquired_at=item.get("acquired_at"),
                    required=bool(item.get("required", True)),
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k
                        not in {
                            "id",
                            "path",
                            "image",
                            "sensor",
                            "resolution_m",
                            "crs",
                            "bands",
                            "nodata",
                            "acquired_at",
                            "required",
                        }
                    },
                )
            )
            continue
        path = resolve_path(item, config_dir)
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        sources.append(RemoteSensingImageSource(id=f"image_{idx:03d}", path=path))
    return sources
