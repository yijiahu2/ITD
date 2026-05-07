from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.data_processing.fusion.postprocess import fuse_instance_layers


def test_fuse_instance_layers_skips_global_proximity_merge_without_boundary(tmp_path: Path) -> None:
    src = tmp_path / "instances.gpkg"
    out = tmp_path / "merged.gpkg"
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[box(0, 0, 2, 2), box(2.3, 0, 4.3, 2)],
        crs="EPSG:4547",
    )
    gdf.to_file(src, driver="GPKG")

    result = fuse_instance_layers(
        instance_paths=[str(src)],
        output_path=out,
        boundary_vector_path=None,
        overlap_ratio_thr=0.5,
    )

    merged = gpd.read_file(out)
    assert result["input_count"] == 2
    assert result["output_count"] == 2
    assert len(merged) == 2
