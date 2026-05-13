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
    prediction_json = tmp_path / "predictions.json"
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
    prediction_json.write_text(
        json.dumps(
            {
                "annotations": [
                    {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "score": 0.95}
                ]
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
                    "prediction_json": str(prediction_json),
                    "max_images": 1,
                },
                "main_model": {"execution_mode": "prediction_json", "model_id": "fixture_main"},
                "experience_retrieval": {"enabled": False},
                "output_dir": str(tmp_path / "adaptive_out"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

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
