from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.evolution.evolve_infer_runner import run_evolve_infer_v1
from ITD_agent.evolution.review.review_guardrails import V2WriteAction, assert_v2_guardrails, check_write_action
from ITD_agent.evolution.review.review_runner import run_review_v2


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


def _run_v1_fixture(tmp_path: Path) -> dict:
    gt_path = _write_json(tmp_path / "gt.json", _sample_coco_payload())
    main_pred_path = _write_json(tmp_path / "main_pred.json", _main_predictions_payload())
    (tmp_path / "tile_001.tif").write_bytes(b"placeholder-image")
    output_dir = tmp_path / "evolve_out"
    config_path = _write_json(
        tmp_path / "v1_config.json",
        {
            "mode": "supervised_coco_evolve_v1",
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
        },
    )
    return run_evolve_infer_v1(str(config_path))


def test_review_v2_consolidates_v1_run_assets_and_blocks_v3_actions(tmp_path: Path) -> None:
    v1_summary = _run_v1_fixture(tmp_path)
    v1_output_dir = Path(v1_summary["output_dir"])
    v2_config = _write_json(
        tmp_path / "v2_config.json",
        {
            "version": "v2",
            "mode": "trajectory_review_coco_v2",
            "mainline_profile": "A_DOM_ONLY",
            "source": {
                "run_id": v1_summary["run_id"],
                "state_db_path": str(v1_output_dir / "state.sqlite"),
                "artifact_root": str(v1_output_dir),
            },
            "output": {"output_dir": str(v1_output_dir / "v2_review")},
            "trajectory_compression": {
                "enabled": True,
                "include_full_pending_candidates_in_context": False,
                "context_top_k_per_candidate_type": 2,
                "context_top_k_per_error_type": 1,
            },
            "memory_review": {"enabled": True, "min_quality_score": 0.6},
            "skill_review": {"enabled": True, "min_support_count": 1, "status_on_create": "draft"},
            "finetune_pool": {
                "enabled": True,
                "min_quality_score": 0.6,
                "max_samples_per_error_type_per_trajectory": 2,
                "export_coco_bundle": True,
            },
            "routing_review": {"enabled": True, "mark_only": True, "allow_routing_policy_update": False},
            "distillation_review": {"enabled": True, "mark_only": True, "allow_distillation_job": False, "min_quality_score": 0.6},
            "guardrails": {
                "allow_memory_write": True,
                "allow_skill_draft_write": True,
                "allow_finetune_sample_write": True,
                "allow_finetune_bundle_export": True,
                "allow_training_trigger": False,
                "allow_weight_update": False,
                "allow_model_promotion": False,
                "allow_active_skill_policy": False,
                "allow_routing_policy_update": False,
                "allow_expert_to_main_distillation": False,
            },
        },
    )

    report = run_review_v2(str(v2_config))

    assert report["trajectory_count"] == 1
    assert report["invalid_trajectories"] == 0
    assert report["asset_counts"]["memory_records"] >= 1
    assert report["asset_counts"]["skill_records"] >= 1
    assert report["asset_counts"]["finetune_samples"] >= 1
    assert report["asset_counts"]["routing_candidates"] >= 1
    assert report["asset_counts"]["distillation_candidates"] >= 1

    review_dir = Path(report["output_dir"])
    trajectory_id = v1_summary["trajectories"][0]["trajectory_id"]
    assert (review_dir / "integrity" / "integrity_report.json").exists()
    assert (review_dir / "compressed_trajectories" / f"{trajectory_id}.summary.json").exists()
    assert (review_dir / "compressed_trajectories" / f"{trajectory_id}.review_context.json").exists()
    assert (review_dir / "memory" / "memory_records.jsonl").exists()
    assert (review_dir / "skills" / "skill_records.jsonl").exists()
    assert (review_dir / "finetune_pool" / "manifest.csv").exists()
    assert (review_dir / "finetune_pool" / "coco_export_bundle" / "coco_export" / "annotations" / "instances_itd_v2_candidates.json").exists()
    assert (review_dir / "reports" / "review_summary.json").exists()

    compressed_context = json.loads((review_dir / "compressed_trajectories" / f"{trajectory_id}.review_context.json").read_text(encoding="utf-8"))
    protected = compressed_context["protected_stages"]
    assert "pending_review_candidates" not in protected
    assert protected["pending_candidate_summary"]["counts"]["training"] >= 1
    assert protected["pending_candidate_summary"]["full_pending_candidates_ref"]["json_pointer"] == "/pending_review_candidates"

    finetune_context = json.loads((review_dir / "review_contexts" / f"{trajectory_id}.finetune_context.json").read_text(encoding="utf-8"))
    assert finetune_context["context_compression"]["pending_candidates_embedded"] is False
    assert "training_candidates" not in finetune_context
    assert "roi_by_id" not in finetune_context
    assert finetune_context["training_candidate_summary"]["count"] >= 1
    assert len(finetune_context["selected_training_candidates"]) <= 2
    assert finetune_context["candidate_manifest_refs"]["full_pending_candidates_ref"]["json_pointer"] == "/pending_review_candidates"

    with sqlite3.connect(v1_output_dir / "state.sqlite") as conn:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in [
                "v2_review_runs",
                "memory_records",
                "skill_records",
                "finetune_samples",
                "routing_candidates",
                "distillation_candidates",
                "v2_review_events",
            ]
        }
        blocked_guardrails = conn.execute(
            """
            SELECT COUNT(*) FROM v2_review_events
            WHERE review_type = 'guardrail' AND decision = 'reject'
            """
        ).fetchone()[0]
    assert counts["v2_review_runs"] == 1
    assert counts["memory_records"] >= 1
    assert counts["skill_records"] >= 1
    assert counts["finetune_samples"] >= 1
    assert counts["routing_candidates"] >= 1
    assert counts["distillation_candidates"] >= 1
    assert blocked_guardrails >= 5


def test_review_v2_guardrails_reject_v3_flags_and_actions() -> None:
    bad_cfg = {"guardrails": {"allow_training_trigger": True}}

    try:
        assert_v2_guardrails(bad_cfg)
    except ValueError as exc:
        assert "cannot start training" in str(exc)
    else:
        raise AssertionError("V2 guardrails should reject training trigger enablement")

    good_cfg = {"guardrails": {"allow_training_trigger": False, "allow_routing_policy_update": False}}
    assert not check_write_action(V2WriteAction.START_TRAINING_JOB, good_cfg).allowed
    assert not check_write_action(V2WriteAction.UPDATE_ROUTING_POLICY, good_cfg).allowed
