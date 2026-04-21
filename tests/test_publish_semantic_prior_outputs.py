from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from output_layer.publisher import publish_segmentation_deliverables


def _write_raster(path: Path, data: np.ndarray, dtype: str = "uint8") -> None:
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": "EPSG:4547",
        "transform": from_origin(0, 10, 1, 1),
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_publish_segmentation_deliverables_keeps_semantic_prior_assets(tmp_path: Path) -> None:
    inst_path = tmp_path / "Y_inst.gpkg"
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(1, 1, 3, 3)], crs="EPSG:4547")
    gdf.to_file(inst_path, driver="GPKG")

    m_sem_tif = tmp_path / "M_sem.tif"
    _write_raster(m_sem_tif, np.ones((10, 10), dtype=np.uint8))

    m_sem_png = tmp_path / "M_sem.png"
    m_sem_png.write_bytes(b"png")

    publish_root = tmp_path / "final_outputs"
    result = publish_segmentation_deliverables(
        inst_shp=str(inst_path),
        publish_root=publish_root,
        semantic_prior_tif=str(m_sem_tif),
        semantic_prior_png=str(m_sem_png),
        background_raster=str(m_sem_tif),
    )

    assert result["semantic_prior_tif"] == str(publish_root / "M_sem.tif")
    assert result["semantic_prior_png"] == str(publish_root / "M_sem.png")
    assert (publish_root / "M_sem.tif").exists()
    assert (publish_root / "M_sem.png").exists()
