from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from ITD_agent.data_processing.fusion.instance_ops import (
    dedupe_instances_by_overlap,
    merge_split_instances_by_proximity,
    suppress_small_boundary_fragments,
)


def fuse_instance_layers(
    *,
    instance_paths: list[str],
    output_path: str | Path,
    boundary_vector_path: str | None = None,
    overlap_ratio_thr: float = 0.6,
    boundary_band_m: float = 1.5,
    min_area_m2: float = 6.0,
) -> dict[str, Any]:
    valid_paths = [Path(path) for path in instance_paths if path and Path(path).exists()]
    if not valid_paths:
        return {"status": "no_instance_layers", "merged_instance_path": None}

    gdfs = [gpd.read_file(path) for path in valid_paths]
    base_crs = gdfs[0].crs
    merged = gpd.GeoDataFrame(
        pd.concat([gdf.to_crs(base_crs) if gdf.crs != base_crs else gdf for gdf in gdfs], ignore_index=True),
        geometry="geometry",
        crs=base_crs,
    )
    before_count = int(len(merged))
    if boundary_vector_path and Path(boundary_vector_path).exists():
        boundary_gdf = gpd.read_file(boundary_vector_path)
    else:
        boundary_gdf = None

    # Without a boundary vector, scene-wide proximity merging is too aggressive and
    # can collapse many adjacent crowns into a few large polygons. In no-xiaoban
    # mode, only overlap dedupe should run here; ROI/expert loops handle local
    # corrections later.
    if boundary_gdf is not None and not boundary_gdf.empty:
        merged = merge_split_instances_by_proximity(
            merged,
            boundary_gdf=boundary_gdf,
            boundary_band_m=boundary_band_m,
            merge_gap_m=0.9,
            centroid_distance_factor=1.4,
            max_centroid_distance_m=7.0,
        )
    deduped = dedupe_instances_by_overlap(merged, overlap_ratio_thr=overlap_ratio_thr)

    if boundary_gdf is not None:
        if not boundary_gdf.empty:
            deduped = suppress_small_boundary_fragments(
                deduped,
                boundary_gdf,
                boundary_band_m=boundary_band_m,
                min_area_m2=min_area_m2,
            )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_file(out_path)
    after_count = int(len(deduped))
    return {
        "status": "ok",
        "merged_instance_path": str(out_path),
        "removed_duplicates": int(max(before_count - after_count, 0)),
        "input_count": before_count,
        "output_count": after_count,
    }
