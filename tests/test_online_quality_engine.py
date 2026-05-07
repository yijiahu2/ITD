from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.evaluation_analysis.online_quality_engine import evaluate_online_quality


def _write_raster(path: Path, data: np.ndarray, resolution: float = 1.0) -> None:
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": str(data.dtype),
        "crs": "EPSG:4547",
        "transform": from_origin(0, 10, resolution, resolution),
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_evaluate_online_quality_with_semantic_and_chm(tmp_path: Path) -> None:
    inst_path = tmp_path / "inst.gpkg"
    gdf = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[box(1, 1, 3, 3), box(6, 6, 8, 8)], crs="EPSG:4547")
    gdf.to_file(inst_path, driver="GPKG")

    sem = np.zeros((10, 10), dtype=np.uint8)
    sem[7:9, 1:3] = 1
    sem[2:4, 6:8] = 1
    sem_path = tmp_path / "M_sem.tif"
    _write_raster(sem_path, sem)

    chm = np.zeros((10, 10), dtype=np.float32)
    chm[7:9, 1:3] = 12
    chm[2:4, 6:8] = 15
    chm_path = tmp_path / "chm.tif"
    _write_raster(chm_path, chm)

    result = evaluate_online_quality(
        inst_shp=str(inst_path),
        m_sem_tif=str(sem_path),
        chm_tif=str(chm_path),
        patch_raster=str(sem_path),
    )

    semantic = result["metrics"]["semantic_instance_consistency"]
    geometry = result["metrics"]["geometry_plausibility"]
    geometry_diag = result["metrics"]["geometry_diagnostics"]

    assert semantic["available"] is True
    assert semantic["coverage_ratio"] == pytest.approx(1.0)
    assert semantic["semantic_cover_ratio"] == pytest.approx(0.08)
    assert semantic["instance_cover_ratio"] == pytest.approx(0.08)
    assert semantic["cover_ratio_delta_abs"] == pytest.approx(0.0)
    assert semantic["overlap_iou"] == pytest.approx(1.0)
    assert result["metrics"]["height_consistency"]["available"] is True
    assert geometry["instance_count"] == 2
    assert geometry["sum_to_union_ratio"] == pytest.approx(1.0)
    assert geometry["max_instance_area_share"] == pytest.approx(0.5)
    assert geometry["top5_instance_area_share"] == pytest.approx(1.0)
    assert geometry["small_fragment_ratio_lt_4m2"] == pytest.approx(0.0)
    assert geometry_diag["pred_instance_count"] == 2
    assert geometry_diag["valid_instance_ratio"] == pytest.approx(1.0)
    assert geometry_diag["small_fragment_ratio"] == pytest.approx(0.0)
    assert geometry_diag["semantic_instance_consistency"] == pytest.approx(1.0)
    assert geometry_diag["semantic_instance_conflict_flag"] is False
    assert result["quality_score"] is not None
