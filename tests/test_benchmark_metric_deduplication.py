from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.evaluation_analysis.benchmark_engine import evaluate_benchmark_vector_result


def test_benchmark_result_omits_duplicate_precision_recall_aliases(tmp_path: Path) -> None:
    pred_path = tmp_path / "pred.gpkg"
    gt_path = tmp_path / "gt.gpkg"
    gdf = gpd.GeoDataFrame({"score": [0.9]}, geometry=[box(0, 0, 2, 2)], crs="EPSG:4547")
    gdf.to_file(pred_path, driver="GPKG")
    gdf.to_file(gt_path, driver="GPKG")

    result = evaluate_benchmark_vector_result(pred_shp=str(pred_path), gt_shp=str(gt_path))

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["ap50"] == 1.0
    assert result["ap75"] == 1.0
    assert result["f1_score50"] == 1.0
    assert result["mean_iou_matched"] == 1.0
    assert result["error_decomposition"]["miss_detection_score"] == 0.0
    assert result["error_decomposition"]["false_detection_score"] == 0.0
    for duplicate_key in [
        "precision_percent",
        "recall_percent",
        "precision50",
        "recall50",
        "precision50_percent",
        "recall50_percent",
        "precision75",
        "recall75",
        "precision75_percent",
        "recall75_percent",
    ]:
        assert duplicate_key not in result
