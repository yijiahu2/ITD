from __future__ import annotations

import json
from pathlib import Path

from ITD_agent.evolution.config_preflight import preflight_runtime_config


def test_preflight_materializes_default_templates_and_checks_real_assets(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "dataset"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    annotation_json = annotation_dir / "annotations.json"
    annotation_json.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "image_1.tif", "width": 100, "height": 100}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400}],
                "categories": [{"id": 1, "name": "tree_crown"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (annotation_dir / "image_1.tif").write_bytes(b"placeholder")
    base_runtime = tmp_path / "base_runtime.json"
    base_runtime.write_text(json.dumps({"work_dir": str(tmp_path)}), encoding="utf-8")
    htc_config = tmp_path / "htc_config.py"
    htc_ckpt = tmp_path / "htc_best.pth"
    htc_config.write_text("# config", encoding="utf-8")
    htc_ckpt.write_bytes(b"checkpoint")

    cfg_path = tmp_path / "adaptive_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "mode": "adaptive_inference",
                "input": {
                    "annotation_json": str(annotation_json),
                    "image_root": str(annotation_dir),
                    "max_images": 1,
                },
                "runtime": {"base_config": str(base_runtime)},
                "main_model_runtime_config": {
                    "model_id": "legacy_cellpose_sam",
                    "segmentation_algorithm": "legacy_cellpose_sam",
                    "execution_mode": "real",
                    "runtime": {"prediction_score_mode": "semantic_prior_mean_prob"},
                    "stage1_semantic_prior": {"semantic_prior_script": "stage1_segformer_sliding.py"},
                    "stage2_instance": {"segmentation_script": "stage2_cellpose_sam_sliding_v2.py", "diam_list": "96,192,320"},
                },
                "expert_models": {
                    "execution_mode": "real",
                    "default_templates": {
                        "htc": {
                            "expert_model": {
                                "name": "htc",
                                "framework": "mmdetection",
                                "config_file": str(htc_config),
                                "checkpoint_file": str(htc_ckpt),
                                "device": "cuda:0",
                            },
                            "input": {"tile_size": 1024, "tile_overlap": 384},
                            "inference": {"score_thr": 0.25, "batch_size": 1},
                            "postprocess": {"min_area_px": 20, "merge_tile_iou_thr": 0.45},
                        }
                    },
                },
                "output_dir": str(tmp_path / "adaptive_out"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = preflight_runtime_config(str(cfg_path))

    assert result["ok"] is True
    assert result["runtime"]["base_config"]["exists"] is True
    assert result["main_model"]["execution_mode"] == "real"
    assert result["expert_models"]["execution_mode"] == "real"
    assert result["model_checks"]["htc"]["segmentation_algorithm"] == "mmdet_htc"
    assert result["model_checks"]["htc"]["config_file"]["exists"] is True
    assert result["model_checks"]["htc"]["checkpoint"]["exists"] is True
