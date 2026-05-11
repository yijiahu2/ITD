from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.cli.main import build_parser
from ITD_agent.finetune_pool.review.review_context_builder import ReviewContext
from ITD_agent.finetune_pool.review.skill_reviewer import build_class_level_skill_records
from ITD_agent.learning_gate.dispatcher import dispatch_learning_events
from ITD_agent.learning_gate.event_builder import (
    build_learning_events_from_review_result,
    build_learning_events_from_run_result,
    build_learning_events_from_training_result,
)
from ITD_agent.orchestration import workflow
from ITD_agent.parameter_engine.expert_search_space import get_expert_model_search_space
from ITD_agent.parameter_engine.space_shrinker import shrink_search_space_with_skill
from ITD_agent.planning.scheduler.context_builder import build_scheduler_context
from ITD_agent.planning.scheduler.runtime_scheduler import (
    _build_expert_model_call_plan,
    build_expert_model_planning_runtime_cfg,
)
from ITD_agent.skill_store.matcher import match_skill_context
from ITD_agent.skill_store.query import load_skill_records
from ITD_agent.training_loop.data_synthesis import synthesize_training_samples
from ITD_agent.training_loop.error_type_evaluator import compute_error_type_delta
from ITD_agent.training_loop.evolution_feedback import write_training_evolution_feedback
from ITD_agent.training_loop.geometry_delta_evaluator import compute_geometry_delta


def _make_context(
    trajectory_id: str,
    *,
    by_failure_family: dict[str, int] | None = None,
    expert_model: str = "maskdino",
    expert_decision: str = "accept",
    review_gain: float = 0.05,
) -> ReviewContext:
    return ReviewContext(
        source_run_id="run_fixture",
        trajectory_id=trajectory_id,
        image_id="image_1",
        trajectory_summary={"roi_summary": {"by_failure_family": by_failure_family or {}}},
        artifact_refs={},
        memory_candidates=[],
        skill_candidates=[],
        training_candidates=[],
        routing_update_candidates=[
            {
                "candidate_id": f"route_{trajectory_id}",
                "trajectory_id": trajectory_id,
                "level1_error_type": "false_negative",
                "failure_family": next(iter((by_failure_family or {"small_crown_recall": 1}).keys())),
                "expert_model": expert_model,
                "expert_decision": expert_decision,
                "improvement": {"score_gain": review_gain},
                "safety": {"regression": False},
            }
        ],
        distillation_candidates=[],
        roi_by_id={},
        expert_task_by_id={},
    )


def test_skill_reviewer_builds_multiple_triggered_skill_types() -> None:
    contexts = [
        _make_context("traj_1", by_failure_family={"small_crown_recall": 2}),
        _make_context("traj_2", by_failure_family={"small_crown_recall": 1}),
        _make_context("traj_3", by_failure_family={"false_positive_cleanup": 1}, expert_decision="reject", review_gain=-0.02),
        _make_context("traj_4", by_failure_family={"false_positive_cleanup": 1}, expert_decision="reject", review_gain=-0.01),
    ]
    cfg = {
        "skill_review": {
            "min_support_count": 3,
            "status_on_create": "draft",
            "periodic_nudge": {"enabled": True, "every_n_trajectories": 4, "min_trajectory_count": 4},
            "expert_success": {"enabled": True, "min_support_count": 2, "min_score_improvement": 0.03},
            "fusion_guard": {"enabled": True, "min_rejected_refinement_count": 2},
        }
    }

    records = build_class_level_skill_records(
        review_run_id="review_fixture",
        source_run_id="run_fixture",
        contexts=contexts,
        cfg=cfg,
    )

    by_trigger = {str(record["trigger_conditions"].get("trigger_type")): record for record in records}
    assert "repeated_failure" in by_trigger
    assert "periodic_nudge" in by_trigger
    assert "expert_success" in by_trigger
    assert "fusion_guard" in by_trigger
    assert by_trigger["repeated_failure"]["skill_type"] == "training_sample_selection_skill"
    assert by_trigger["expert_success"]["skill_type"] == "expert_routing_skill"
    assert by_trigger["fusion_guard"]["skill_type"] == "fusion_guard_skill"


def test_scheduler_context_loads_skill_context_from_skill_store(tmp_path: Path) -> None:
    review_output_dir = tmp_path / "review_out"
    skill_records_path = review_output_dir / "skills" / "skill_records.jsonl"
    skill_records_path.parent.mkdir(parents=True, exist_ok=True)
    skill_records_path.write_text(
        json.dumps(
            {
                "skill_id": "skill_dense_recall",
                "skill_type": "expert_routing_skill",
                "name": "dense recall routing",
                "source_run_ids": ["run_fixture"],
                "source_trajectory_ids": ["traj_1", "traj_2"],
                "trigger_conditions": {"failure_family": "small_crown_recall", "trigger_type": "repeated_failure"},
                "recommended_action": {"mode": "readonly_suggestion", "description": "Prefer dense expert on small crown recall failures."},
                "evidence_summary": {"support_count": 2, "failure_family": "small_crown_recall"},
                "safety_constraints": {"requires_human_review_for_activation": True},
                "status": "draft",
                "version": "review.1",
                "created_at": "2026-05-11T12:00:00+00:00",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_cfg = {
        "mainline_profile": "A_DOM_ONLY",
        "_mainline_capabilities": {
            "allow_dem": False,
            "allow_chm": False,
            "allow_external_knowledge": False,
            "allow_public_datasets": True,
            "allow_memory_context": True,
            "allow_finetune_pool_context": True,
        },
        "review_output_dir": str(review_output_dir),
        "_roi_assessment": {"failure_family": "small_crown_recall"},
    }

    context = build_scheduler_context(runtime_cfg=runtime_cfg)

    assert "skill_context" in context
    assert context["skill_context"]["matched_skill_count"] == 1
    assert context["skill_context"]["matched_skills"][0]["skill_id"] == "skill_dense_recall"
    assert context["skill_context"]["matched_skills"][0]["evidence_summary"]["failure_family"] == "small_crown_recall"


def test_training_feedback_writer_outputs_controlled_feedback_file(tmp_path: Path) -> None:
    report = write_training_evolution_feedback(
        training_summary={
            "training_result": {"status": "completed"},
            "promotion_decision": {"decision": "promote_to_shadow"},
            "sample_quality_report": {"accepted_count": 8},
            "model_version": {"model_version_id": "model_1"},
        },
        output_dir=tmp_path / "training_out",
    )

    assert Path(report["feedback_path"]).exists()
    assert report["feedback"]["recommended_memory_write"] is True
    assert report["feedback"]["recommended_skill_update"] is True
    assert report["feedback"]["recommended_routing_review"] is True


def test_evolve_workflow_and_cli_chain_run_review_train(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def _run(config_path: str) -> dict[str, str]:
        calls.append(("run", config_path))
        return {"output_dir": "/tmp/run"}

    def _review(config_path: str) -> dict[str, str]:
        calls.append(("review", config_path))
        return {"output_dir": "/tmp/review"}

    def _train(config_path: str) -> dict[str, str]:
        calls.append(("train", config_path))
        return {"output_dir": "/tmp/train"}

    monkeypatch.setattr("ITD_agent.orchestration.workflow.run_full_workflow", _run)
    monkeypatch.setattr("ITD_agent.orchestration.workflow.run_review_workflow", _review)
    monkeypatch.setattr("ITD_agent.orchestration.workflow.run_training_workflow", _train)
    monkeypatch.setattr(
        "ITD_agent.learning_gate.dispatcher.dispatch_learning_events",
        lambda events, cfg, output_dir: {"event_count": len(events), "output_dir": str(output_dir)},
    )
    run_cfg = tmp_path / "run_cfg.json"
    review_cfg = tmp_path / "review_cfg.json"
    training_cfg = tmp_path / "training_cfg.json"
    run_cfg.write_text(json.dumps({"mode": "workflow"}, ensure_ascii=False), encoding="utf-8")
    review_cfg.write_text(json.dumps({"source": {}, "output": {}}, ensure_ascii=False), encoding="utf-8")
    training_cfg.write_text(json.dumps({"source": {}}, ensure_ascii=False), encoding="utf-8")
    evolve_cfg = tmp_path / "controlled_self_evolution.yaml"
    evolve_cfg.write_text(
        json.dumps(
            {
                "closed_loop": {"output_dir": str(tmp_path / "closed_loop"), "review_after_run": True, "train_after_review": True},
                "run_config": str(run_cfg),
                "review_config": str(review_cfg),
                "training_config": str(training_cfg),
                "learning_gate": {"enabled": True},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = workflow.evolve(str(evolve_cfg))

    assert [item[0] for item in calls] == ["run", "review", "train"]
    assert result["command"] == "evolve"
    assert result["run"]["result"]["output_dir"] == "/tmp/run"
    assert result["review"]["result"]["output_dir"] == "/tmp/review"
    assert result["training"]["result"]["output_dir"] == "/tmp/train"
    assert len(result["learning_events"]) == 3

    parser = build_parser()
    args = parser.parse_args(["evolve", "--config", str(evolve_cfg)])
    assert args.command == "evolve"


def test_skill_query_matcher_space_shrinker_and_synthesis_follow_acceptance_requirements(tmp_path: Path) -> None:
    review_output_dir = tmp_path / "review_output"
    skill_path = review_output_dir / "skills" / "skill_records.jsonl"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(
        json.dumps(
            {
                "skill_id": "skill_small_crown_recall",
                "skill_type": "parameter_adjustment_skill",
                "name": "small crown recall tuning",
                "status": "draft",
                "trigger_conditions": {"failure_family": "small_crown_recall", "trigger_type": "repeated_failure"},
                "recommended_action": {
                    "parameter_space": {
                        "shrink": {
                            "score_thr": {"direction": "decrease"},
                            "min_area_px": {"direction": "decrease"},
                        }
                    }
                },
                "evidence_summary": {"support_count": 5, "failure_family": "small_crown_recall"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    skills = load_skill_records(review_output_dir=review_output_dir, statuses=["draft", "shadow", "active"], limit=50)
    skill_context = match_skill_context(
        skills=skills,
        scene_profile={},
        evaluation_metrics={},
        roi_assessment={"failure_family": "small_crown_recall"},
        failure_pattern_context=[],
    )
    base_space = get_expert_model_search_space("mmdet_htc")
    shrunk = shrink_search_space_with_skill(base_space=base_space, skill_context=skill_context, failure_family="small_crown_recall")

    assert skill_context["matched_skill_count"] == 1
    assert shrunk["decision_params"]["score_thr"]["range"][1] <= base_space["decision_params"]["score_thr"]["range"][1]
    assert len(shrunk["decision_params"]["min_area_px"]["values"]) <= len(base_space["decision_params"]["min_area_px"]["values"])
    assert shrunk["_shrink_metadata"]["source"] == "skill_context"

    synthesis = synthesize_training_samples(
        accepted_samples=[
            {
                "sample_id": "sample_1",
                "failure_category": "false_negative",
                "metadata": {"failure_category": "false_negative"},
            },
            {
                "sample_id": "sample_2",
                "failure_category": "false_negative",
                "metadata": {"failure_category": "false_negative"},
            },
            {
                "sample_id": "sample_3",
                "failure_category": "false_negative",
                "metadata": {"failure_category": "false_negative"},
            },
            {
                "sample_id": "sample_4",
                "failure_category": "false_negative",
                "metadata": {"failure_category": "false_negative"},
            },
        ],
        cfg={"enabled": True, "max_synthetic_ratio": 0.3},
        output_dir=tmp_path / "dataset_bundle",
    )
    synthetic = synthesis["synthetic_samples"]
    assert len(synthetic) <= 1
    assert synthetic[0]["source_sample_id"] == "sample_1"
    assert synthetic[0]["label_status"] == "synthetic"
    assert synthetic[0]["split_policy"] == "train_only"


def test_error_geometry_delta_and_learning_dispatcher_produce_required_outputs(tmp_path: Path) -> None:
    error_delta = compute_error_type_delta(
        baseline_errors={"under_segmentation": 0.3, "false_negative": 0.2},
        candidate_errors={"under_segmentation": 0.2, "false_negative": 0.25},
    )
    geometry_delta = compute_geometry_delta(
        baseline_metrics={"tree_count_error_ratio": 0.2, "density_error_abs": 10.0},
        candidate_metrics={"tree_count_error_ratio": 0.15, "density_error_abs": 11.0},
    )
    run_events = build_learning_events_from_run_result({"result": {"run_id": "run_1", "output_dir": "/tmp/run", "score_before": 0.5, "score_after": 0.4, "scene_signature": {"forest": "dense"}, "parameter_signature": {"tile": 2048}}})
    review_events = build_learning_events_from_review_result({"result": {"output_dir": "/tmp/review", "asset_counts": {"skill_records": 1, "finetune_samples": 2}}})
    training_events = build_learning_events_from_training_result({"result": {"output_dir": "/tmp/training", "promotion_decision": {"decision": "keep_candidate"}, "training_result": {"status": "skipped"}}})
    dispatch = dispatch_learning_events(
        events=[*run_events, *review_events, *training_events],
        cfg={"allow_write_memory": True, "allow_write_skill_draft": True, "allow_write_finetune_pool": True},
        output_dir=tmp_path / "closed_loop",
    )

    assert error_delta["status"] == "computed"
    assert "under_segmentation_delta" in error_delta["delta"]
    assert geometry_delta["status"] == "computed"
    assert "tree_count_error_ratio_delta" in geometry_delta["delta"]
    assert dispatch["event_count"] == len(run_events) + len(review_events) + len(training_events)
    assert Path(dispatch["report_path"]).exists()


def test_expert_runtime_cfg_and_call_plan_include_skill_driven_search_space(tmp_path: Path) -> None:
    runtime_cfg = build_expert_model_planning_runtime_cfg(
        cfg={"output_dir": str(tmp_path / "run_out")},
        input_assessment={},
        input_manifest={},
        data_processing_summary={},
        roi_assessment={"failure_family": "small_crown_recall"},
        previous_round_summary={},
    )
    assert runtime_cfg["state_db_path"].endswith("state.sqlite")
    assert runtime_cfg["review_output_dir"].endswith("review")

    scheduler_context = {
        "skill_context": {
            "matched_skills": [
                {
                    "skill_id": "skill_small_crown_recall",
                    "recommended_action": {
                        "parameter_space": {
                            "shrink": {
                                "score_thr": {"direction": "decrease"},
                                "min_area_px": {"direction": "decrease"},
                            }
                        }
                    },
                }
            ]
        }
    }
    llm_result = {
        "expert_model_call_plan": {
            "preferred_expert_model": "mmdet_htc",
            "candidate_models": ["mmdet_htc"],
        }
    }

    plan = _build_expert_model_call_plan(
        runtime_cfg={
            "ITD_agent": {
                "planning": {"expert_model_routing": {}},
                "segmentation_models": {"expert_models": [{"name": "mmdet_htc", "algorithm": "mmdet_htc", "expert_family": "small_crown_recall", "failure_categories": ["small_crown_recall"]}]},
            }
        },
        scheduler_context=scheduler_context,
        llm_result=llm_result,
        planning_stage="expert_model",
    )

    assert "parameter_search_space" in plan
    assert plan["parameter_search_space"]["_shrink_metadata"]["source"] == "skill_context"
