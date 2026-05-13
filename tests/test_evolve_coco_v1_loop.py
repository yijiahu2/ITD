from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.evolution.adaptive_inference import run_adaptive_inference_stage
from ITD_agent.evolution.expert.expert_task_builder import ExpertTask
from ITD_agent.evolution.expert.tile_image import offset_instances_to_full_image
from ITD_agent.evolution.fusion.local_roi_fusion import fuse_or_rollback
from ITD_agent.evolution.roi.roi_candidate_builder import build_roi_candidates
from ITD_agent.evolution.state.queries import list_pending_reviews, summarize_state
from ITD_agent.evaluation_analysis.coco_error_decomposition import decompose_coco_errors
from ITD_agent.evaluation_analysis.expert_result_comparator import compare_expert_with_main
from ITD_agent.planning.scheduler.expert_routing_policy import route_expert_model


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _sample_coco_payload() -> dict:
    return {
        "images": [{"id": 1, "file_name": "tile_001.tif", "width": 1024, "height": 1024}],
        "annotations": [
            {"id": 101, "image_id": 1, "category_id": 1, "bbox": [100, 100, 80, 80], "area": 6400},
            {"id": 102, "image_id": 1, "category_id": 1, "bbox": [300, 100, 80, 80], "area": 6400},
            {"id": 103, "image_id": 1, "category_id": 1, "bbox": [500, 100, 80, 80], "area": 6400},
            {"id": 104, "image_id": 1, "category_id": 1, "bbox": [650, 100, 80, 80], "area": 6400},
        ],
        "categories": [{"id": 1, "name": "tree"}],
    }


def _main_predictions_payload() -> dict:
    return {
        "images": [{"id": 1, "file_name": "tile_001.tif", "width": 1024, "height": 1024}],
        "annotations": [
            {"id": 201, "image_id": 1, "category_id": 1, "bbox": [98, 98, 82, 82], "score": 0.91},
            {"id": 202, "image_id": 1, "category_id": 1, "bbox": [295, 95, 290, 90], "score": 0.88},
            {"id": 203, "image_id": 1, "category_id": 1, "bbox": [760, 100, 50, 50], "score": 0.55},
        ],
    }


def test_coco_error_decomposition_builds_four_supervised_error_classes() -> None:
    gt = _sample_coco_payload()["annotations"]
    pred = _main_predictions_payload()["annotations"]

    result = decompose_coco_errors(
        gt_instances=gt,
        pred_instances=pred,
        iou_threshold=0.5,
        weak_overlap_threshold=0.1,
    )

    assert result.metrics["false_negative_count"] == 3
    assert result.metrics["false_positive_count"] == 2
    assert result.metrics["under_segmentation_count"] == 1
    assert result.metrics["over_segmentation_count"] == 0
    assert {item["level1_error_type"] for item in result.errors} >= {
        "false_negative",
        "false_positive",
        "under_segmentation",
    }


def test_roi_candidate_builder_preserves_affected_ids_and_failure_family() -> None:
    error_decomp = decompose_coco_errors(
        gt_instances=_sample_coco_payload()["annotations"],
        pred_instances=_main_predictions_payload()["annotations"],
        iou_threshold=0.5,
        weak_overlap_threshold=0.1,
    )

    rois = build_roi_candidates(
        image_id="1",
        image_size=(1024, 1024),
        error_decomposition=error_decomp,
        geometry_review={"failure_tags": []},
    )

    assert rois
    false_negative = next(roi for roi in rois if roi.level1_error_type == "false_negative")
    assert false_negative.affected_gt_ids
    assert false_negative.bbox_px[2] > false_negative.bbox_px[0]
    assert false_negative.failure_family == "small_crown_recall"
    assert all(0.0 <= roi.severity_score <= 1.0 for roi in rois)


def test_local_fusion_keeps_main_instances_outside_accepted_roi() -> None:
    main_instances = [
        {"id": "main_keep", "bbox": [10, 10, 30, 30]},
        {"id": "main_replace", "bbox": [100, 100, 40, 40]},
    ]
    expert_instances = [{"id": "expert_roi", "bbox": [102, 102, 36, 36]}]

    result = fuse_or_rollback(
        main_instances=main_instances,
        expert_results=[
            {
                "expert_task_id": "task_1",
                "instances": expert_instances,
            }
        ],
        expert_reviews=[
            {
                "expert_task_id": "task_1",
                "decision": "accept",
                "accepted_roi_ids": ["roi_1"],
                "roi_bboxes": {"roi_1": [95, 95, 150, 150]},
            }
        ],
        min_improvement_epsilon=0.01,
    )

    fused_ids = {str(instance["id"]) for instance in result["instances"]}
    assert result["final_result_source"] == "expert_fused"
    assert "main_keep" in fused_ids
    assert "main_replace" not in fused_ids
    assert "expert_roi" in fused_ids


def test_expert_comparator_can_partial_accept_roi_subset() -> None:
    task = ExpertTask(
        expert_task_id="task_partial",
        trajectory_id="traj_1",
        image_id="1",
        expert_model="mock_expert",
        failure_family="mixed",
        level1_error_type="false_negative",
        roi_ids=["roi_improved", "roi_regressed"],
        fusion_bboxes={
            "roi_improved": [90, 90, 190, 190],
            "roi_regressed": [290, 90, 390, 190],
        },
    )
    gt_instances = [
        {"id": "gt_1", "bbox": [100, 100, 80, 80]},
        {"id": "gt_2", "bbox": [300, 100, 80, 80]},
    ]
    main_instances = [{"id": "main_2", "bbox": [300, 100, 80, 80]}]
    expert_results = [
        {
            "expert_task_id": "task_partial",
            "instances": [{"id": "expert_1", "bbox": [100, 100, 80, 80]}],
        }
    ]

    reviews = compare_expert_with_main(
        expert_tasks=[task],
        expert_results=expert_results,
        main_instances=main_instances,
        gt_instances=gt_instances,
    )

    assert reviews[0]["decision"] == "partial_accept"
    assert reviews[0]["accepted_roi_ids"] == ["roi_improved"]
    assert reviews[0]["rejected_roi_ids"] == ["roi_regressed"]


def test_expert_routing_policy_supports_primary_expert_map() -> None:
    policy = {
        "expert_map": {
            "under_segmentation": {"primary_expert": "htc"},
            "over_segmentation": {"primary_expert": "mask2former"},
            "false_positive": {"primary_expert": "cascade_mask_rcnn"},
            "false_negative": {"primary_expert": "maskdino"},
        }
    }

    assert route_expert_model("under_segmentation", policy)["expert_model"] == "htc"
    assert route_expert_model("over_segmentation", policy)["expert_model"] == "mask2former"
    assert route_expert_model("false_positive", policy)["expert_model"] == "cascade_mask_rcnn"
    assert route_expert_model("false_negative", policy)["expert_model"] == "maskdino"


def test_expert_tile_instances_are_offset_back_to_full_image() -> None:
    adjusted = offset_instances_to_full_image(
        [{"id": "tile_pred", "bbox": [10, 20, 30, 40], "score": 0.9}],
        [100, 200],
    )

    assert adjusted[0]["bbox"] == [110.0, 220.0, 30, 40]
    assert adjusted[0]["tile_offset_xy"] == [100.0, 200.0]


def test_evolve_infer_v1_runs_mock_expert_writes_trajectory_state_and_training_candidates(tmp_path: Path) -> None:
    gt_path = _write_json(tmp_path / "gt.json", _sample_coco_payload())
    main_pred_path = _write_json(tmp_path / "main_pred.json", _main_predictions_payload())
    output_dir = tmp_path / "evolve_out"
    config_path = _write_json(
        tmp_path / "config.json",
        {
            "mode": "adaptive_inference",
            "mainline_profile": "A_DOM_ONLY",
            "input": {
                "annotation_json": str(gt_path),
                "image_root": str(tmp_path),
                "prediction_json": str(main_pred_path),
            },
            "output_dir": str(output_dir),
            "main_model": {"model_id": "legacy_cellpose_sam", "execution_mode": "prediction_json"},
            "expert_models": {"execution_mode": "mock", "mock_strategy": "use_gt_or_perturbed_gt"},
            "evaluation": {"matching": {"iou_threshold": 0.5, "weak_overlap_threshold": 0.1}},
            "adaptive_inference": {"min_improvement_epsilon": 0.01},
            "roi_policy": {
                "expert_tile_size_px": 1024,
                "fusion_buffer_px": 64,
                "min_trigger_per_tile": {"min_failure_instances": 1},
            },
            "expert_routing_policy": {
                "route_map": {
                    "under_segmentation": "htc",
                    "over_segmentation": "mask2former",
                    "false_positive": "cascade_mask_rcnn",
                    "false_negative": "maskdino",
                }
            },
        },
    )

    summary = run_adaptive_inference_stage(str(config_path))

    assert summary["run_id"].startswith("run_")
    assert summary["trajectory_count"] == 1
    assert summary["totals"]["roi_candidates"] >= 1
    assert summary["totals"]["expert_tasks"] >= 1
    assert summary["totals"]["training_candidates"] >= 1
    assert summary["trajectories"][0]["final_result_source"] in {"expert_fused", "partial_expert_fused"}

    trajectory_path = Path(summary["trajectories"][0]["trajectory_path"])
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert trajectory["main_eval_stage"]["error_decomposition"]["metrics"]["false_negative_count"] == 3
    assert trajectory["expert_task_stage"]["expert_tasks"][0]["execution_mode"] == "mock"
    assert trajectory["expert_review_stage"]["expert_reviews"]
    assert trajectory["pending_review_candidates"]["training_candidates"]

    db_path = output_dir / "state.sqlite"
    with sqlite3.connect(db_path) as conn:
        table_counts = {
            name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            for name in [
                "runs",
                "trajectories",
                "roi_candidates",
                "expert_tasks",
                "expert_reviews",
                "fusion_events",
                "training_candidates",
                "artifacts",
            ]
        }
    assert table_counts["runs"] == 1
    assert table_counts["trajectories"] == 1
    assert table_counts["roi_candidates"] >= 1
    assert table_counts["expert_tasks"] >= 1
    assert table_counts["expert_reviews"] >= 1
    assert table_counts["fusion_events"] >= 1
    assert table_counts["training_candidates"] >= 1
    assert table_counts["artifacts"] >= 1

    state_summary = summarize_state(db_path)
    pending = list_pending_reviews(db_path)
    assert state_summary["counts"]["runs"] == 1
    assert pending["pending_training_candidates"]
    assert pending["pending_trajectories"]


def test_evolve_infer_v1_real_mode_derives_dataset_split_and_invokes_real_models(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "Dataset_4"
    split_dir = dataset_root / "Validation_set"
    annotation_path = _write_json(split_dir / "annotation" / "validation_gt.json", _sample_coco_payload())
    image_dir = split_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "tile_001.tif").write_bytes(b"placeholder")
    base_config = _write_json(tmp_path / "runtime.json", {"work_dir": str(tmp_path), "segmentation_script": "unused.py"})
    output_dir = tmp_path / "real_out"

    calls: list[dict] = []

    def fake_real_segmentation(**kwargs):
        calls.append(kwargs)
        model_id = kwargs["model_cfg"].get("model_id")
        if model_id == "legacy_cellpose_sam":
            instances = _main_predictions_payload()["annotations"]
        else:
            instances = [
                {**ann, "id": f"expert_{ann['id']}", "score": 0.99}
                for ann in _sample_coco_payload()["annotations"]
            ]
        return {
            "status": "completed",
            "model_id": model_id,
            "runtime_cfg": {"segmentation_algorithm": kwargs["model_cfg"].get("segmentation_algorithm", "legacy_cellpose_sam")},
            "artifacts": {"y_inst_tif": str(tmp_path / f"{model_id}.tif")},
            "instances": instances,
        }

    monkeypatch.setattr("ITD_agent.evolution.adaptive_inference.run_real_segmentation_for_sample", fake_real_segmentation)
    config_path = _write_json(
        tmp_path / "real_config.json",
        {
            "mode": "adaptive_inference",
            "mainline_profile": "A_DOM_ONLY",
            "input": {
                "dataset_root": str(dataset_root),
                "split": "validation",
                "max_images": 1,
            },
            "runtime": {"base_config": str(base_config)},
            "output_dir": str(output_dir),
            "main_model": {"model_id": "legacy_cellpose_sam", "execution_mode": "real"},
            "expert_models": {"execution_mode": "real"},
            "model_configs": {
                "legacy_cellpose_sam": {"model_id": "legacy_cellpose_sam"},
                "maskdino": {"segmentation_algorithm": "maskdino_official"},
                "mask2former": {"segmentation_algorithm": "mmdet_mask2former"},
                "cascade_mask_rcnn": {"segmentation_algorithm": "mmdet_cascade_mask_rcnn"},
                "htc": {"segmentation_algorithm": "mmdet_htc"},
            },
            "expert_routing_policy": {
                "expert_map": {
                    "under_segmentation": {"primary_expert": "htc"},
                    "over_segmentation": {"primary_expert": "mask2former"},
                    "false_positive": {"primary_expert": "cascade_mask_rcnn"},
                    "false_negative": {"primary_expert": "maskdino"},
                }
            },
            "evaluation": {"matching": {"iou_threshold": 0.5, "weak_overlap_threshold": 0.1}},
            "adaptive_inference": {"min_improvement_epsilon": 0.01},
            "roi_policy": {
                "expert_tile_size_px": 1024,
                "min_trigger_per_tile": {"min_failure_instances": 1},
            },
        },
    )

    summary = run_adaptive_inference_stage(str(config_path))

    assert summary["trajectory_count"] == 1
    assert summary["totals"]["expert_tasks"] >= 1
    assert len(calls) >= 2
    assert calls[0]["model_cfg"]["model_id"] == "legacy_cellpose_sam"
    assert any(call["model_cfg"].get("model_id") != "legacy_cellpose_sam" for call in calls[1:])
    trajectory = json.loads(Path(summary["trajectories"][0]["trajectory_path"]).read_text(encoding="utf-8"))
    assert trajectory["input_snapshot"]["annotation_json"] == str(annotation_path)
    assert trajectory["main_model_stage"]["execution_result"]["execution_mode"] == "real"
    assert trajectory["expert_review_stage"]["expert_results"][0]["execution_mode"] == "real"
    routed_models = {task["expert_model"] for task in trajectory["expert_task_stage"]["expert_tasks"]}
    assert routed_models
    assert routed_models.issubset({"htc", "mask2former", "cascade_mask_rcnn", "maskdino"})
