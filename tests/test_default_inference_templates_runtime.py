from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from input_layer.adapters import normalize_agent_runtime_config


def _touch(path: Path, content: str = "stub") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_main_workflow_normalization_materializes_default_templates_into_runtime_keys(tmp_path: Path) -> None:
    dom = _touch(tmp_path / "dom.tif")
    cfg = {
        "runtime": {
            "run_name": "template_runtime",
            "work_dir": str(tmp_path),
            "conda_sh": "/home/xth/anaconda3/etc/profile.d/conda.sh",
            "conda_env": "tcd",
        },
        "inputs": {
            "remote_sensing": {"images": [{"id": "dom", "path": dom, "required": True}]},
        },
        "main_model_runtime_config": {
            "model_id": "legacy_cellpose_sam",
            "model_role": "main_model",
            "segmentation_algorithm": "legacy_cellpose_sam",
            "execution_mode": "real",
            "runtime": {
                "work_dir": str(tmp_path),
                "conda_env": "tcd",
                "device": "cuda:0",
                "prediction_score_mode": "semantic_prior_mean_prob",
            },
            "stage1_semantic_prior": {
                "semantic_prior_script": "/home/xth/forest_agent_project/runtime_entrypoints/semantic_prior_segformer.py",
            },
            "stage2_instance": {
                "segmentation_script": "/home/xth/forest_agent_project/runtime_entrypoints/segmentation_legacy_cellpose_sam.py",
                "diam_list": "96,192,320",
                "tile": 1024,
                "overlap": 0,
                "tile_overlap": 0.35,
                "bsize": 256,
                "augment": True,
                "iou_merge_thr": 0.24,
            },
        },
        "expert_models": {
            "execution_mode": "real",
            "default_templates": {
                "maskdino": {
                    "expert_model": {
                        "name": "maskdino",
                        "framework": "maskdino_detectron2",
                        "config_file": "/home/xth/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml",
                        "checkpoint_file": "/home/xth/MaskDINO/weights/maskdino_r50.pth",
                        "device": "cuda:0",
                    },
                    "input": {"tile_size": 1024, "tile_overlap": 0},
                    "inference": {"instance_score_thr": 0.2, "batch_size": 1, "max_instances": 500},
                    "postprocess": {"min_area_px": 20, "mode": "missed_instance_recall"},
                }
            },
        },
        "outputs": {"root_base_dir": str(tmp_path / "outputs")},
    }

    runtime_cfg, _ = normalize_agent_runtime_config(cfg)

    assert runtime_cfg["semantic_prior_script"] == "/home/xth/forest_agent_project/runtime_entrypoints/semantic_prior_segformer.py"
    assert runtime_cfg["segmentation_script"] == "/home/xth/forest_agent_project/runtime_entrypoints/segmentation_legacy_cellpose_sam.py"
    assert runtime_cfg["segmentation_algorithm"] == "legacy_cellpose_sam"
    assert runtime_cfg["diam_list"] == "96,192,320"
    assert runtime_cfg["tile"] == 1024
    assert runtime_cfg["tile_overlap"] == 0.35
    assert runtime_cfg["iou_merge_thr"] == 0.24
    assert runtime_cfg["_default_inference_templates"]["main_model"]["execution_mode"] == "real"
    assert runtime_cfg["_default_inference_templates"]["expert_models"]["execution_mode"] == "real"
    assert runtime_cfg["_default_inference_templates"]["model_configs"]["maskdino"]["segmentation_algorithm"] == "maskdino_official"


def test_main_workflow_normalization_rejects_non_real_default_template_modes(tmp_path: Path) -> None:
    dom = _touch(tmp_path / "dom.tif")
    cfg = {
        "runtime": {"run_name": "bad_template", "work_dir": str(tmp_path)},
        "inputs": {
            "remote_sensing": {"images": [{"id": "dom", "path": dom, "required": True}]},
        },
        "main_model_runtime_config": {
            "model_id": "legacy_cellpose_sam",
            "segmentation_algorithm": "legacy_cellpose_sam",
            "execution_mode": "prediction_json",
            "runtime": {"work_dir": str(tmp_path)},
            "stage1_semantic_prior": {"semantic_prior_script": "/tmp/semantic.py"},
            "stage2_instance": {"segmentation_script": "/tmp/seg.py", "diam_list": "96,192,320", "tile": 1024, "overlap": 0, "tile_overlap": 0.35, "bsize": 256, "augment": True, "iou_merge_thr": 0.24},
        },
        "expert_models": {"execution_mode": "real", "default_templates": {"maskdino": {"expert_model": {"name": "maskdino", "framework": "maskdino_detectron2", "config_file": "/tmp/maskdino.yaml", "checkpoint_file": "/tmp/maskdino.pth"}}}},
        "outputs": {"root_base_dir": str(tmp_path / "outputs")},
    }

    with pytest.raises(ValueError, match="main_model_runtime_config.execution_mode must be 'real'"):
        normalize_agent_runtime_config(cfg)
