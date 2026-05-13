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

from output_layer.contracts import FinalTreeCrownResult
from output_layer.publisher import publish_final_tree_crown_outputs, publish_segmentation_deliverables


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
    assert result["semantic_mask_tif"] == str(publish_root / "semantic_mask.tif")
    assert result["semantic_mask_png"] == str(publish_root / "semantic_mask.png")
    assert result["tree_crowns_shp"] == str(publish_root / "tree_crowns.shp")
    assert result["tree_points_shp"] == str(publish_root / "tree_points.shp")
    assert (publish_root / "M_sem.tif").exists()
    assert (publish_root / "M_sem.png").exists()
    assert (publish_root / "tree_crowns.shp").exists()
    assert (publish_root / "tree_points.shp").exists()
    assert (publish_root / "segmentation_visualization.png").exists()
    assert (publish_root / "final_report.md").exists()
    assert result["scenario"] == "dom_without_gt"
    assert result["results_tree_crowns_shp"] == str(publish_root / "results" / "tree_crowns.shp")
    assert result["masks_semantic_mask_tif"] == str(publish_root / "masks" / "semantic_mask.tif")
    assert result["masks_instance_mask_tif"] == str(publish_root / "masks" / "instance_mask.tif")
    assert (publish_root / "results" / "tree_crowns.shp").exists()
    assert (publish_root / "results" / "tree_points.shp").exists()
    assert (publish_root / "masks" / "semantic_mask.tif").exists()
    assert (publish_root / "masks" / "instance_mask.tif").exists()
    assert (publish_root / "visualization" / "pred_overlay.png").exists()
    assert (publish_root / "visualization" / "risk_map.png").exists()
    assert (publish_root / "report" / "inference_report.md").exists()


def test_publish_final_tree_crown_outputs_accepts_coco_bbox_instances(tmp_path: Path) -> None:
    dom_path = tmp_path / "dom.tif"
    _write_raster(dom_path, np.ones((20, 20), dtype=np.uint8))
    publish_root = tmp_path / "coco_final_outputs"

    result = publish_final_tree_crown_outputs(
        result=FinalTreeCrownResult(
            run_id="run_coco",
            output_dir=str(publish_root),
            input_dom_path=str(dom_path),
            instances=[
                {
                    "id": 1,
                    "image_id": 7,
                    "category_id": 1,
                    "bbox": [2, 3, 5, 6],
                    "score": 0.9,
                }
            ],
            gt_metrics={
                "dataset": "Dataset_4 Validation_set sample",
                "image_count": 1,
                "gt_instance_count": 1,
                "pred_instance_count": 1,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "bbox_ap50": 1.0,
                "bbox_ap75": 1.0,
                "bbox_ap": 1.0,
                "mask_ap50": 1.0,
                "mask_ap75": 1.0,
                "mask_ap": 1.0,
                "miou": 1.0,
                "fp": 0,
                "fn": 0,
            },
            metadata={
                "source_adapter": "test_coco",
                "write_legacy_compat_outputs": True,
                "annotation_path": "/tmp/validation_gt.json",
                "image_root": "/tmp/images",
                "dataset_type": "coco_instance_segmentation_with_gt_no_coordinates",
            },
        ),
        publish_root=publish_root,
    )

    assert result["status"] == "published"
    assert result["scenario"] == "coco_gt"
    assert result["coco_predictions_json"] == str(publish_root / "results" / "coco_predictions.json")
    assert result["crown_geometry_source"] == "bbox_fallback"
    assert Path(result["tree_crowns_shp"]).exists()
    assert Path(result["tree_points_shp"]).exists()
    assert Path(result["semantic_mask_tif"]).exists()
    assert Path(result["semantic_mask_png"]).exists()
    assert Path(result["segmentation_visualization_png"]).exists()
    assert Path(result["final_report_md"]).exists()
    assert (publish_root / "results" / "coco_predictions.json").exists()
    assert (publish_root / "masks" / "instance_masks" / "7_instance_mask.png").exists()
    assert (publish_root / "report" / "evaluation_report.md").exists()
    report_text = (publish_root / "report" / "evaluation_report.md").read_text(encoding="utf-8")
    for section in [
        "## 1. 数据集基本信息",
        "## 2. 模型与推理配置",
        "## 3. COCO 指标结果表",
        "## 4. 检测指标结果表",
        "## 5. 掩码分割指标结果表",
        "## 6. 错误类型统计",
        "## 7. 典型失败案例",
        "## 8. 总体结论",
    ]:
        assert section in report_text
    assert "| Dataset_4 Validation_set sample | 1 | 1 | 1 | 1.0000 | 1.0000 | 1.0000 |" in report_text


def test_publish_final_tree_crown_outputs_writes_dom_with_gt_layout(tmp_path: Path) -> None:
    dom_path = tmp_path / "dom.tif"
    _write_raster(dom_path, np.ones((20, 20), dtype=np.uint8))
    publish_root = tmp_path / "dom_gt_final_outputs"

    result = publish_final_tree_crown_outputs(
        result=FinalTreeCrownResult(
            run_id="run_dom_gt",
            output_dir=str(publish_root),
            input_type="dom_with_gt",
            input_dom_path=str(dom_path),
            instances=[
                {
                    "id": 1,
                    "image_id": "tile_1",
                    "bbox": [2, 3, 5, 6],
                    "score": 0.9,
                    "gt_id": "gt_1",
                    "iou_gt": 0.72,
                }
            ],
            gt_metrics={"precision": 1.0, "recall": 1.0},
            gt_matches=[{"pred_id": 1, "gt_id": "gt_1", "iou_gt": 0.72}],
        ),
        publish_root=publish_root,
    )

    assert result["scenario"] == "dom_with_gt"
    assert Path(result["results_tree_crowns_shp"]).exists()
    assert Path(result["results_tree_points_shp"]).exists()
    assert Path(result["masks_semantic_mask_tif"]).exists()
    assert Path(result["masks_instance_mask_tif"]).exists()
    assert (publish_root / "visualization" / "evaluation_map.png").exists()
    assert (publish_root / "report" / "evaluation_report.md").exists()
