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
    online_metrics = result["online_quality"]["metrics"]
    assert online_metrics["semantic_instance_consistency"]["available"] is True
    assert online_metrics["geometry_plausibility"]["available"] is True
    assert online_metrics["semantic_instance_consistency"]["overlap_iou"] == 1.0
    assert online_metrics["geometry_plausibility"]["instance_count"] == 2
    assert "area_consistency" not in result
    assert "geometry_diagnostics" not in result


def test_experiment_report_renders_area_diagnostics_sections(tmp_path: Path) -> None:
    report_path = tmp_path / "final_report.md"
    summary = {
        "run_name": "demo_run",
        "tree_crowns_shp": str(tmp_path / "tree_crowns.shp"),
        "final_evaluation": {
            "evaluation_mode": "reference_quality",
            "selected_metrics": {},
            "decision_flags": {
                "overall_score": 0.82,
                "quality_pass_flag": True,
                "need_local_refine_flag": False,
                "need_param_search_flag": False,
                "need_finetune_flag": False,
                "need_manual_review_flag": False,
                "accepted_improvement_flag": None,
                "regression_flag": None,
            },
            "online_quality": {
                "quality_score": 0.1,
                "metrics": {
                    "semantic_instance_consistency": {
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
                    "geometry_plausibility": {
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
                    "geometry_diagnostics": {
                        "pred_instance_count": 42,
                        "empty_output_flag": False,
                        "pred_cover_ratio": 0.58,
                        "valid_instance_ratio": 1.0,
                        "shape_anomaly_ratio": 0.04,
                        "small_fragment_ratio": 0.30,
                        "large_blob_ratio": 0.07,
                        "duplicate_overlap_ratio": 0.0,
                        "edge_artifact_score": 0.12,
                        "fragmentation_score": 0.21,
                        "merge_blob_score": 0.11,
                        "semantic_instance_conflict_flag": False,
                    },
                },
            },
        },
    }

    build_experiment_report(summary, report_path)
    content = report_path.read_text(encoding="utf-8")

    assert "## 冠层面积一致性" in content
    assert "## 几何健康度" in content
    assert "| 预测实例数 | 42 |" in content
    assert "## 决策 Flags" in content
    assert "| overall_score | 0.8200 |" in content


def test_experiment_report_renders_benchmark_error_decomposition(tmp_path: Path) -> None:
    report_path = tmp_path / "benchmark_report.md"
    summary = {
        "run_name": "benchmark_demo",
        "final_evaluation": {
            "evaluation_mode": "benchmark",
            "precision": 0.8,
            "recall": 0.75,
            "ap50": 0.7,
            "ap75": 0.55,
            "f1_score50": 0.7742,
            "mean_iou_matched": 0.68,
            "mae": 1.2,
            "rmse": 1.5,
            "rmse_percent": 15.0,
            "r2": 0.72,
            "iou_0_75": {"precision": 0.6, "recall": 0.55},
            "num_predictions": 100,
            "num_ground_truth": 95,
            "score_field": "score",
            "ground_truth_file": "/tmp/gt.gpkg",
            "crown_area_iou_0_50": {"num_matched_crowns": 71},
            "error_decomposition": {
                "under_segmentation_score": 0.12,
                "over_segmentation_score": 0.08,
                "miss_detection_score": 0.10,
                "false_detection_score": 0.09,
                "failure_severity": 0.12,
                "failure_pattern_confidence": 0.03,
            },
            "decision_flags": {
                "overall_score": 0.66,
                "quality_pass_flag": False,
                "need_param_search_flag": False,
                "need_finetune_flag": False,
                "need_manual_review_flag": True,
            },
        },
    }

    build_experiment_report(summary, report_path)
    content = report_path.read_text(encoding="utf-8")

    assert "## 错误分解" in content
    assert "| under_segmentation_score | 0.1200 |" in content
    assert "| failure_severity | 0.1200 |" in content
    assert "| failure_pattern_confidence | 0.0300 |" in content
    assert "| F1@0.50 | 0.7742 |" in content
