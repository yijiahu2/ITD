from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.evolution.adaptive_inference import run_adaptive_inference_stage
from ITD_agent.learning_gate.event_builder import build_learning_events_from_run_result


def test_adaptive_inference_stage_has_formal_entrypoint() -> None:
    assert callable(run_adaptive_inference_stage)


def test_adaptive_inference_writes_foreground_outputs_and_background_refs(tmp_path: Path) -> None:
    annotation_json = tmp_path / "annotations.json"
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
    (tmp_path / "image_1.tif").write_bytes(b"placeholder")
    base_runtime = tmp_path / "base_runtime.json"
    base_runtime.write_text(
        json.dumps(
            {
                "work_dir": str(tmp_path),
                "conda_sh": "/tmp/conda.sh",
                "conda_env": "tcd",
                "semantic_prior_script": "stage1_default.py",
                "segmentation_script": "stage2_default.py",
                "diam_list": "64,128,256",
                "tile": 512,
                "overlap": 0,
                "tile_overlap": 0.2,
                "bsize": 64,
                "augment": False,
                "iou_merge_thr": 0.1,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cfg_path = tmp_path / "adaptive_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "mode": "adaptive_inference",
                "input": {
                    "annotation_json": str(annotation_json),
                    "image_root": str(tmp_path),
                    "max_images": 1,
                },
                "runtime": {"base_config": str(base_runtime)},
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
                        "semantic_prior_script": "stage1_segformer_sliding.py",
                    },
                    "stage2_instance": {
                        "segmentation_script": "stage2_cellpose_sam_sliding_v2.py",
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
                        "htc": {
                            "expert_model": {
                                "name": "htc",
                                "role": "under_segmentation_correction",
                                "framework": "mmdetection",
                                "config_file": str(tmp_path / "htc_config.py"),
                                "checkpoint_file": str(tmp_path / "htc_best.pth"),
                                "device": "cuda:0",
                            },
                            "input": {"tile_size": 1024, "tile_overlap": 384, "keep_georeference": True},
                            "inference": {"score_thr": 0.25, "batch_size": 1, "max_per_img": 500, "nms_iou_thr": 0.55},
                            "postprocess": {"min_area_px": 20, "merge_tile_iou_thr": 0.45, "mode": "split_merged_crowns"},
                        }
                    },
                },
                "experience_retrieval": {"enabled": False},
                "expert_routing_policy": {"expert_map": {"under_segmentation": {"primary_expert": "htc"}}},
                "output_dir": str(tmp_path / "adaptive_out"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    from unittest import mock

    with mock.patch("ITD_agent.evolution.adaptive_inference.run_real_segmentation_for_sample") as real_run:
        real_run.return_value = {
            "status": "completed",
            "model_id": "legacy_cellpose_sam",
            "runtime_cfg": {"segmentation_algorithm": "legacy_cellpose_sam"},
            "artifacts": {"y_inst_tif": str(tmp_path / "legacy_cellpose_sam.tif")},
            "instances": [{"id": "main_1", "image_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "score": 0.95}],
        }
        summary = run_adaptive_inference_stage(str(cfg_path))

    final_outputs = summary["final_outputs"]
    assert summary["foreground_goal"] == "single_tree_crown_detection_and_extraction"
    assert summary["input_layer"]["mainline_profile"] == "A_DOM_ONLY"
    assert summary["input_layer"]["validation"]["status"] == "ok"
    assert summary["data_processing"]["public_dataset_summary"]["gt_visibility_policy"] == "evaluation_analysis_only"
    assert summary["planning_context"]["gt_leakage_guard"]["main_model_plan_uses_gt"] is False
    assert Path(final_outputs["final_prediction_json"]).exists()
    assert Path(final_outputs["final_result_bundle_json"]).exists()
    assert Path(final_outputs["final_report_json"]).exists()
    assert Path(final_outputs["final_report_md"]).exists()
    assert final_outputs["output_layer"]["status"] == "published"
    assert Path(final_outputs["output_layer"]["tree_crowns_shp"]).exists()
    assert Path(final_outputs["output_layer"]["tree_points_shp"]).exists()
    assert Path(final_outputs["output_layer"]["semantic_mask_tif"]).exists()
    assert Path(final_outputs["output_layer"]["final_report_md"]).exists()
    assert Path(summary["background_evolution"]["state_db"]).exists()

    final_prediction = json.loads(Path(final_outputs["final_prediction_json"]).read_text(encoding="utf-8"))
    assert len(final_prediction["annotations"]) == 1
    assert final_prediction["annotations"][0]["score"] == 0.95

    trajectory = json.loads(Path(summary["trajectories"][0]["trajectory_path"]).read_text(encoding="utf-8"))
    assert trajectory["main_decision_stage"]["decision"] == "accept_main"
    assert trajectory["final_evaluation_stage"]["coco_metrics"]["matched_count"] == 1
    assert trajectory["main_model_stage"]["prediction_artifacts"]["standardized_instance_format"] == "coco_xywh_instance_v1"
    assert "routing_update_candidates" in trajectory["pending_review_candidates"]
    assert trajectory["pending_review_candidates"]["auto_update_policy"]["start_training"] is False

    events = build_learning_events_from_run_result(summary)
    assert events[0]["final_prediction_json"] == final_outputs["final_prediction_json"]
    assert events[0]["state_db"] == summary["background_evolution"]["state_db"]


def test_adaptive_inference_rejects_non_real_main_mode(tmp_path: Path) -> None:
    annotation_json = tmp_path / "annotations.json"
    annotation_json.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "image_1.png", "width": 100, "height": 100}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400}],
                "categories": [{"id": 1, "name": "tree_crown"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "image_1.png").write_bytes(b"placeholder")
    cfg_path = tmp_path / "adaptive_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "mode": "adaptive_inference",
                "input": {
                    "annotation_json": str(annotation_json),
                    "image_root": str(tmp_path),
                    "max_images": 1,
                },
                "main_model": {"execution_mode": "prediction_json", "model_id": "fixture_main"},
                "expert_models": {"execution_mode": "real"},
                "output_dir": str(tmp_path / "adaptive_out"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    try:
        run_adaptive_inference_stage(str(cfg_path))
    except ValueError as exc:
        assert "main_model.execution_mode must be 'real'" in str(exc)
    else:
        raise AssertionError("adaptive_inference should reject non-real main model mode")


def test_adaptive_inference_accepts_default_runtime_templates_and_maps_to_real_runtime(tmp_path: Path, monkeypatch) -> None:
    annotation_json = tmp_path / "annotations.json"
    annotation_json.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "image_1.tif", "width": 100, "height": 100}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400},
                    {"id": 2, "image_id": 1, "category_id": 1, "bbox": [40, 10, 20, 20], "area": 400},
                ],
                "categories": [{"id": 1, "name": "tree_crown"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "image_1.tif").write_bytes(b"placeholder")
    base_runtime = tmp_path / "base_runtime.json"
    base_runtime.write_text(
        json.dumps(
            {
                "work_dir": str(tmp_path),
                "conda_sh": "/tmp/conda.sh",
                "conda_env": "tcd",
                "semantic_prior_script": "stage1_default.py",
                "segmentation_script": "stage2_default.py",
                "diam_list": "64,128,256",
                "tile": 512,
                "overlap": 0,
                "tile_overlap": 0.2,
                "bsize": 64,
                "augment": False,
                "iou_merge_thr": 0.1,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calls: list[dict] = []

    def fake_real_segmentation(**kwargs):
        calls.append(kwargs)
        model_id = kwargs["model_cfg"].get("model_id")
        if model_id == "legacy_cellpose_sam":
            instances = [{"id": "main_1", "image_id": 1, "bbox": [8, 8, 54, 24], "area": 1296, "score": 0.75}]
        else:
            instances = [
                {"id": "expert_1", "image_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "score": 0.99},
                {"id": "expert_2", "image_id": 1, "bbox": [40, 10, 20, 20], "area": 400, "score": 0.98},
            ]
        return {
            "status": "completed",
            "model_id": model_id,
            "runtime_cfg": kwargs["model_cfg"],
            "artifacts": {"y_inst_tif": str(tmp_path / f"{model_id}.tif")},
            "instances": instances,
        }

    monkeypatch.setattr("ITD_agent.evolution.adaptive_inference.run_real_segmentation_for_sample", fake_real_segmentation)

    cfg_path = tmp_path / "adaptive_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "mode": "adaptive_inference",
                "input": {
                    "annotation_json": str(annotation_json),
                    "image_root": str(tmp_path),
                    "max_images": 1,
                },
                "runtime": {"base_config": str(base_runtime)},
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
                        "semantic_prior_script": "stage1_segformer_sliding.py",
                        "semantic_prior_ckpt": "/tmp/semantic.ckpt",
                        "save_semantic_prior_probability_tif": True,
                    },
                    "stage2_instance": {
                        "segmentation_script": "stage2_cellpose_sam_sliding_v2.py",
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
                        "tile_execution": True,
                        "default_templates": {
                            "htc": {
                            "expert_model": {
                                "name": "htc",
                                "role": "under_segmentation_correction",
                                "framework": "mmdetection",
                                "config_file": str(tmp_path / "htc_config.py"),
                                "checkpoint_file": str(tmp_path / "htc_best.pth"),
                                "device": "cuda:0",
                            },
                            "input": {"tile_size": 1024, "tile_overlap": 384, "keep_georeference": True},
                            "inference": {
                                "score_thr": 0.25,
                                "mask_thr_binary": 0.45,
                                "nms_iou_thr": 0.55,
                                "max_per_img": 500,
                                "batch_size": 1,
                            },
                            "postprocess": {
                                "min_area_px": 20,
                                "max_area_px": 300000,
                                "merge_tile_iou_thr": 0.45,
                                    "mode": "split_merged_crowns",
                                },
                            },
                            "cascade_mask_rcnn": {
                                "expert_model": {
                                    "name": "cascade_mask_rcnn",
                                    "role": "false_positive_cleanup",
                                    "framework": "mmdetection",
                                    "config_file": str(tmp_path / "cascade_config.py"),
                                    "checkpoint_file": str(tmp_path / "cascade_best.pth"),
                                    "device": "cuda:0",
                                },
                                "input": {"tile_size": 1024, "tile_overlap": 256, "keep_georeference": True},
                                "inference": {"score_thr": 0.50, "mask_thr_binary": 0.50, "nms_iou_thr": 0.45, "max_per_img": 300, "batch_size": 1},
                                "postprocess": {"min_area_px": 50, "max_area_px": 200000, "min_compactness": 0.08, "max_aspect_ratio": 4.0, "mode": "false_positive_filter"},
                            },
                            "mask2former": {
                                "expert_model": {
                                    "name": "mask2former",
                                    "role": "over_segmentation_correction",
                                    "framework": "detectron2_mask2former",
                                    "config_file": str(tmp_path / "mask2former_config.yaml"),
                                    "checkpoint_file": str(tmp_path / "mask2former_best.pth"),
                                    "device": "cuda:0",
                                },
                                "input": {"tile_size": 1024, "tile_overlap": 256, "keep_georeference": True},
                                "inference": {"instance_score_thr": 0.35, "object_mask_thr": 0.45, "overlap_thr": 0.80, "max_instances": 200, "batch_size": 1},
                                "postprocess": {"min_area_px": 50, "merge_fragment_iou_thr": 0.20, "merge_fragment_boundary_gap_px": 8, "merge_fragment_centroid_distance_px": 80, "mode": "fragment_merge"},
                            },
                            "maskdino": {
                                "expert_model": {
                                    "name": "maskdino",
                                    "role": "missed_crown_recall",
                                    "framework": "maskdino_detectron2",
                                    "config_file": str(tmp_path / "maskdino_config.yaml"),
                                    "checkpoint_file": str(tmp_path / "maskdino_best.pth"),
                                    "device": "cuda:0",
                                },
                                "input": {"tile_size": 1024, "tile_overlap": 384, "keep_georeference": True},
                                "inference": {"instance_score_thr": 0.20, "object_mask_thr": 0.40, "max_instances": 500, "topk_per_image": 500, "batch_size": 1},
                                "postprocess": {"min_area_px": 20, "max_area_px": 250000, "add_new_instance_iou_thr": 0.20, "suppress_overlap_with_main_iou_thr": 0.60, "mode": "missed_instance_recall"},
                            },
                        },
                    },
                    "expert_routing_policy": {
                        "expert_map": {
                            "under_segmentation": {"primary_expert": "htc"},
                            "over_segmentation": {"primary_expert": "mask2former"},
                            "false_positive": {"primary_expert": "cascade_mask_rcnn"},
                            "false_negative": {"primary_expert": "maskdino"},
                        }
                    },
                    "output_dir": str(tmp_path / "adaptive_out"),
                },
                ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = run_adaptive_inference_stage(str(cfg_path))

    assert summary["trajectory_count"] == 1
    assert len(calls) >= 1
    main_call = next(call for call in calls if call["model_cfg"].get("model_id") == "legacy_cellpose_sam")
    main_model_cfg = main_call["model_cfg"]
    assert main_model_cfg["model_id"] == "legacy_cellpose_sam"
    assert main_model_cfg["segmentation_algorithm"] == "legacy_cellpose_sam"
    assert main_model_cfg["runtime_overrides"]["semantic_prior_script"] == "stage1_segformer_sliding.py"
    assert main_model_cfg["runtime_overrides"]["semantic_prior_ckpt"] == "/tmp/semantic.ckpt"
    assert main_model_cfg["runtime_overrides"]["diam_list"] == "96,192,320"
    assert main_model_cfg["runtime_overrides"]["tile"] == 1024
    assert main_model_cfg["runtime_overrides"]["tile_overlap"] == 0.35
    expert_call = next(call for call in calls if call["model_cfg"].get("model_id") == "htc")
    expert_model_cfg = expert_call["model_cfg"]
    assert expert_model_cfg["segmentation_algorithm"] == "mmdet_htc"
    assert expert_model_cfg["segmentation_algorithm_cfg"]["config_file"] == str(tmp_path / "htc_config.py")
    assert expert_model_cfg["segmentation_algorithm_cfg"]["checkpoint"] == str(tmp_path / "htc_best.pth")
    assert expert_model_cfg["segmentation_algorithm_cfg"]["tile_size"] == 1024
    assert expert_model_cfg["segmentation_algorithm_cfg"]["tile_overlap"] == 384
