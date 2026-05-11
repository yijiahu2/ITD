from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.training_loop.contracts import TrainingRunResult
from ITD_agent.training_loop.post_train_evaluator import run_post_train_evaluation
from ITD_agent.training_loop.replay_guard import evaluate_replay_guard
from ITD_agent.training_loop.training_runner import _should_run_formal_training
from ITD_agent.training_loop.training_runner import run_training_loop


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _append_jsonl(path: Path, payloads: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n", encoding="utf-8")


def test_training_loop_builds_controlled_pilot_from_review_assets(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    sample_dir = review_dir / "finetune_pool" / "samples" / "sample_traincand_1"
    image = sample_dir / "image.png"
    gt_mask = sample_dir / "gt_mask.json"
    main_pred = sample_dir / "main_pred_mask.json"
    expert_pred = sample_dir / "expert_pred_mask.json"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"fake-png")
    _write_json(gt_mask, {"mask": "gt"})
    _write_json(main_pred, {"mask": "main"})
    _write_json(expert_pred, {"mask": "expert"})
    _append_jsonl(
        review_dir / "finetune_pool" / "samples.jsonl",
        [
            {
                "sample_id": "sample_traincand_1",
                "created_at": "2026-05-10T00:00:00+00:00",
                "source_run_id": "run_fixture",
                "source_trajectory_id": "traj_1",
                "source_roi_id": "roi_1",
                "image_id": "1",
                "sample_type": "main_failure_sample",
                "target_model_role": "main_model",
                "target_error_type": "false_negative",
                "quality_score": 0.9,
                "review_status": "approved",
                "export_status": "exported",
                "image_crop_path": str(image),
                "gt_mask_path": str(gt_mask),
                "main_pred_path": str(main_pred),
                "expert_pred_path": str(expert_pred),
                "roi": {"roi_id": "roi_1", "bbox_px": [10, 20, 70, 90], "level1_error_type": "false_negative"},
            }
        ],
    )
    _append_jsonl(
        review_dir / "distillation" / "distillation_candidates.jsonl",
        [
            {
                "distillation_candidate_id": "distill_1",
                "source_run_id": "run_fixture",
                "source_trajectory_id": "traj_1",
                "source_roi_id": "roi_1",
                "expert_model": "maskdino",
                "quality_tier": "silver",
                "status": "candidate_only",
            }
        ],
    )
    _append_jsonl(review_dir / "routing" / "routing_candidates.jsonl", [])
    _write_json(review_dir / "reports" / "review_summary.json", {"review_run_id": "review_fixture", "source_run_id": "run_fixture"})

    cfg_path = _write_json(
        tmp_path / "v3_config.json",
        {
            "version": "v3",
            "mode": "controlled_training",
            "source": {"run_id": "run_fixture", "review_asset_dir": str(review_dir)},
            "target": {
                "target_model_role": "expert_model",
                "target_model_id": "maskdino_official",
                "target_expert_family": "dense_adhesion",
                "failure_category": "false_negative",
            },
            "expert_taxonomy": {"path": "configs/expert_taxonomy/expert_families.yaml"},
            "dataset": {"train_ratio": 0.7, "val_ratio": 0.15, "build_replay": True},
            "quality_gate": {
                "min_training_ready_samples": 1,
                "min_replay_samples": 0,
                "allow_manual_labels": True,
                "allow_pseudo_labels": True,
                "reject_missing_artifacts": True,
                "reject_invalid_masks": True,
                "reject_empty_annotations": True,
                "max_single_trajectory_ratio": 1.0,
            },
            "training": {"pilot": {"enabled": True, "override_epochs": 1, "build_only": True}},
            "runner": {"output_dir": str(tmp_path / "controlled_training"), "execute_training": False},
            "evaluation": {"run_replay_guard": True},
            "dom_only_geometry_guard": {"enabled": True},
            "capability_profile": {"enabled": True},
            "routing_candidate": {"enabled": True, "require_promotion_to_shadow": True, "require_replay_guard_pass": True, "require_geometry_guard_pass": True},
            "training_feedback": {"enabled": True},
            "promotion": {"register_candidate": True, "allow_promote_to_shadow": True, "allow_promote_to_active": False, "require_replay_guard_pass": True},
            "distillation": {"enabled": True, "pseudo_label_quality_min": "silver"},
            "guardrails": {
                "allow_weight_update": True,
                "allow_active_model_replace": False,
                "allow_active_routing_policy_update": False,
                "allow_active_skill_policy": False,
                "allow_llm_direct_training_decision": False,
                "allow_llm_direct_model_promotion": False,
            },
        },
    )

    summary = run_training_loop(str(cfg_path))

    out_dir = Path(summary["output_dir"])
    assert summary["trigger_decision"]["decision"] == "approve_pilot"
    assert summary["sample_quality_report"]["accepted_count"] == 1
    assert summary["family_config"]["algorithm_name"] == "maskdino_official"
    assert summary["training_plan"]["command"][-1] == "--build-only"
    assert summary["training_result"]["status"] == "skipped"
    command_path = next((out_dir / "training_jobs").glob("pilot_*/command.sh"))
    command_text = command_path.read_text(encoding="utf-8")
    assert "export PYTHONPATH=" in command_text
    assert "cd /home/xth/forest_agent_project" in command_text
    assert (out_dir / "finetune_bundle" / "finetune_dataset_bundle.json").exists()
    assert (out_dir / "dataset_bundle" / "annotations" / "instances_train.json").exists()
    assert (out_dir / "training_jobs").exists()
    assert (out_dir / "replay_guard" / "replay_guard_report.json").exists()
    assert (out_dir / "geometry_guard" / "dom_only_geometry_guard_report.json").exists()
    assert summary["dom_only_geometry_guard"]["status"] == "not_evaluated"
    assert summary["capability_profile"]["recommended_usage"]["allowed_status"] == "candidate"
    assert (out_dir / "promotion" / "promotion_decision.json").exists()
    assert (out_dir / "distillation" / "main_model_distillation_manifest.csv").exists()
    assert (out_dir / "routing" / "routing_update_candidate.json").exists()
    assert (out_dir / "feedback" / "memory_feedback_candidate.json").exists()
    assert (out_dir / "feedback" / "skill_feedback_candidate.json").exists()
    assert summary["promotion_decision"]["decision"] == "keep_candidate"


def test_post_train_eval_computes_metric_delta_and_replay_guard_blocks_missing_evidence(tmp_path: Path) -> None:
    baseline_metrics = _write_json(tmp_path / "baseline_metrics.json", {"ap_50_95": 0.42, "ap50": 0.68, "precision": 0.7, "recall": 0.61})
    candidate_metrics = _write_json(tmp_path / "candidate_metrics.json", {"ap_50_95": 0.45, "ap50": 0.7, "precision": 0.72, "recall": 0.63})
    result = TrainingRunResult(
        training_job_id="job_eval",
        training_mode="pilot",
        status="completed",
        returncode=0,
        command=["python", "-m", "ITD_agent.segmentation.model_training.train_mmdet_instance", "--config", str(tmp_path / "cfg.yaml")],
        stdout_log=str(tmp_path / "stdout.log"),
        stderr_log=str(tmp_path / "stderr.log"),
        best_checkpoint_path=str(tmp_path / "best.pth"),
        training_metrics_path=str(tmp_path / "training_metrics.json"),
    )

    evaluation = run_post_train_evaluation(
        cfg={"evaluation": {"baseline_metrics_json": str(baseline_metrics), "candidate_metrics_json": str(candidate_metrics)}},
        training_result=result,
        output_dir=tmp_path / "eval_out",
    )

    assert evaluation["delta"]["status"] == "computed"
    assert evaluation["delta"]["delta"]["ap_50_95"] > 0
    replay_pass = evaluate_replay_guard(
        evaluation=evaluation,
        cfg={"_v3_internal": {"replay_sample_count": 3}, "replay_guard": {"require_evaluated_delta": True, "require_replay_samples": True}},
        output_dir=tmp_path / "guard_ok",
    )
    assert replay_pass["passed"] is True

    replay_fail = evaluate_replay_guard(
        evaluation={"candidate": {"status": "completed"}, "delta": {"status": "not_computed"}},
        cfg={"_v3_internal": {"replay_sample_count": 0}, "replay_guard": {"require_evaluated_delta": True, "require_replay_samples": True}},
        output_dir=tmp_path / "guard_fail",
    )
    assert replay_fail["passed"] is False
    assert {item["check"] for item in replay_fail["failures"]} >= {"evaluated_delta", "replay_samples"}


def test_formal_training_gate_requires_pilot_and_guards() -> None:
    pilot_result = TrainingRunResult(
        training_job_id="job_pilot",
        training_mode="pilot",
        status="completed",
        returncode=0,
        command=[],
        stdout_log="",
        stderr_log="",
        best_checkpoint_path="/tmp/best.pth",
        training_metrics_path=None,
    )
    cfg = {"training": {"formal": {"enabled": True, "require_pilot_pass": True, "require_pilot_replay_guard_pass": True, "require_pilot_dom_geometry_guard_pass": True}}}

    assert _should_run_formal_training(
        cfg,
        pilot_result,
        {"passed": True},
        {"geometry_guard_passed": True},
    )
    assert not _should_run_formal_training(
        cfg,
        pilot_result,
        {"passed": True},
        {"geometry_guard_passed": False},
    )
