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

from ITD_agent.evaluation_analysis.final_assessment import evaluate_reference_quality_result
from output_layer.reporting.experiment_report import build_experiment_report


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


def test_reference_quality_result_includes_area_and_geometry_diagnostics(tmp_path: Path) -> None:
    inst_path = tmp_path / "tree_crowns.gpkg"
    gdf = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[box(1, 1, 3, 3), box(6, 6, 8, 8)], crs="EPSG:4547")
    gdf.to_file(inst_path, driver="GPKG")

    sem = np.zeros((10, 10), dtype=np.uint8)
    sem[7:9, 1:3] = 1
    sem[2:4, 6:8] = 1
    sem_path = tmp_path / "M_sem.tif"
    _write_raster(sem_path, sem)

    summary = {
        "run_name": "demo_run",
        "tree_crowns_shp": str(inst_path),
        "run_meta": {"input_image": str(sem_path)},
        "data_processing": {"m_sem_tif": str(sem_path)},
    }

    result = evaluate_reference_quality_result(summary, runtime_cfg={"input_image": str(sem_path)})

    assert result["evaluation_mode"] == "reference_quality"
    assert result["area_consistency"]["available"] is True
    assert result["geometry_diagnostics"]["available"] is True
    assert result["area_consistency"]["overlap_iou"] == 1.0
    assert result["geometry_diagnostics"]["instance_count"] == 2


def test_experiment_report_renders_area_diagnostics_sections(tmp_path: Path) -> None:
    report_path = tmp_path / "final_report.md"
    summary = {
        "run_name": "demo_run",
        "tree_crowns_shp": str(tmp_path / "tree_crowns.shp"),
        "final_evaluation": {
            "evaluation_mode": "reference_quality",
            "selected_metrics": {},
            "area_consistency": {
                "available": True,
                "patch_area_m2": 100.0,
                "semantic_area": 60.0,
                "instance_union_area": 58.0,
                "semantic_cover_ratio": 0.60,
                "instance_cover_ratio": 0.58,
                "cover_ratio_delta_abs": 0.02,
                "overlap_iou": 0.90,
                "coverage_ratio": 0.9667,
                "semantic_recall": 0.95,
                "instance_leakage": 0.03,
                "semantic_gap": 0.05,
            },
            "geometry_diagnostics": {
                "available": True,
                "instance_count": 42,
                "union_area_m2": 58.0,
                "sum_to_union_ratio": 1.0,
                "mean_area_m2": 1.38,
                "median_area_m2": 1.12,
                "mean_equivalent_crown_width_m": 1.25,
                "small_fragment_ratio_lt_4m2": 0.30,
                "small_fragment_ratio_lt_6m2": 0.40,
                "max_instance_area_share": 0.07,
                "top5_instance_area_share": 0.22,
                "edge_touch_ratio": 0.12,
                "overlap_pair_count": 0,
            },
        },
    }

    build_experiment_report(summary, report_path)
    content = report_path.read_text(encoding="utf-8")

    assert "## 冠层面积一致性" in content
    assert "## 几何健康度" in content
    assert "| 实例数 | 42 |" in content
