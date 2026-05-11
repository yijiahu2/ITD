from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.finetune_pool.review.io_utils import write_json
from ITD_agent.training_loop.contracts import TrainingPlan, TrainingRunResult


def write_training_feedback_candidates(
    *,
    cfg: dict[str, Any],
    plan: TrainingPlan | None,
    training_result: TrainingRunResult | None,
    sample_quality_report: dict[str, Any],
    dataset_card: dict[str, Any],
    replay_guard_report: dict[str, Any],
    dom_geometry_guard_report: dict[str, Any],
    promotion_decision: dict[str, Any],
    capability_profile: dict[str, Any] | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    feedback_cfg = cfg.get("training_feedback") or {}
    root = Path(output_dir) / "feedback"
    if not bool(feedback_cfg.get("enabled", True)):
        report = {"enabled": False, "reason": "training_feedback_disabled"}
        write_json(root / "training_lesson_report.json", report)
        return report

    model_version_id = (capability_profile or {}).get("model_version_id")
    target_scope = _target_scope(plan)
    positive_lessons = _positive_lessons(training_result, replay_guard_report, dom_geometry_guard_report, promotion_decision)
    negative_lessons = _negative_lessons(sample_quality_report, replay_guard_report, dom_geometry_guard_report, promotion_decision)
    future_actions = _future_actions(sample_quality_report, dataset_card, replay_guard_report, dom_geometry_guard_report)
    memory_candidate = {
        "feedback_id": f"memory_feedback_{model_version_id or 'controlled_training'}",
        "source_stage": "controlled_training",
        "source_model_version_id": model_version_id,
        "memory_type": "training_lesson",
        "target_scope": target_scope,
        "positive_lessons": positive_lessons,
        "negative_lessons": negative_lessons,
        "recommended_future_actions": future_actions,
        "status": str(feedback_cfg.get("status") or "pending_review"),
    }
    skill_candidate = {
        "feedback_id": f"skill_feedback_{model_version_id or 'controlled_training'}",
        "source_stage": "controlled_training",
        "source_model_version_id": model_version_id,
        "skill_candidate_type": "readonly_analysis_skill",
        "suggested_skill_name": f"{target_scope}_training_case_review",
        "suggested_usage": {
            "can_affect_report": True,
            "can_affect_training_sample_selection": False,
            "can_affect_routing_policy": False,
            "can_affect_fusion_policy": False,
        },
        "content_summary": [*positive_lessons, *negative_lessons],
        "activation_requirement": "manual_review",
        "status": str(feedback_cfg.get("status") or "pending_review"),
    }
    if bool(feedback_cfg.get("write_memory_feedback_candidate", True)):
        write_json(root / "memory_feedback_candidate.json", memory_candidate)
    if bool(feedback_cfg.get("write_skill_feedback_candidate", True)):
        write_json(root / "skill_feedback_candidate.json", skill_candidate)
    report = {
        "enabled": True,
        "memory_feedback_candidate": str(root / "memory_feedback_candidate.json"),
        "skill_feedback_candidate": str(root / "skill_feedback_candidate.json"),
        "memory_feedback_summary": memory_candidate,
        "skill_feedback_summary": skill_candidate,
    }
    write_json(root / "training_lesson_report.json", report)
    return report


def _target_scope(plan: TrainingPlan | None) -> str:
    if not plan:
        return "controlled_training"
    return "_".join(str(item) for item in [plan.target_expert_family, plan.failure_category] if item)


def _positive_lessons(
    training_result: TrainingRunResult | None,
    replay_guard_report: dict[str, Any],
    dom_geometry_guard_report: dict[str, Any],
    promotion_decision: dict[str, Any],
) -> list[str]:
    lessons: list[str] = []
    if training_result and training_result.status in {"completed", "recovered_ckpt"}:
        lessons.append("pilot training produced a traceable checkpoint")
    if replay_guard_report.get("passed"):
        lessons.append("replay guard did not detect metric regression")
    if dom_geometry_guard_report.get("geometry_guard_passed"):
        lessons.append("DOM-only geometry guard did not detect geometry regression")
    if promotion_decision.get("decision") == "promote_to_shadow":
        lessons.append("candidate satisfied shadow gate")
    return lessons or ["training_loop generated auditable training evidence without active changes"]


def _negative_lessons(
    sample_quality_report: dict[str, Any],
    replay_guard_report: dict[str, Any],
    dom_geometry_guard_report: dict[str, Any],
    promotion_decision: dict[str, Any],
) -> list[str]:
    lessons: list[str] = []
    rejected = int(sample_quality_report.get("rejected_count") or 0)
    if rejected:
        lessons.append(f"{rejected} samples were rejected by quality gate")
    if not replay_guard_report.get("passed"):
        lessons.append("replay guard did not pass or was not evaluated")
    if not dom_geometry_guard_report.get("geometry_guard_passed"):
        lessons.append("DOM-only geometry guard did not pass or lacked evaluation inputs")
    if promotion_decision.get("decision") != "promote_to_shadow":
        lessons.append(f"promotion stayed at {promotion_decision.get('decision')}")
    return lessons


def _future_actions(
    sample_quality_report: dict[str, Any],
    dataset_card: dict[str, Any],
    replay_guard_report: dict[str, Any],
    dom_geometry_guard_report: dict[str, Any],
) -> list[str]:
    actions = ["review generated routing and feedback candidates manually"]
    if int((dataset_card.get("replay_counts") or {}).get("samples") or 0) == 0:
        actions.append("collect replay samples before formal promotion")
    if int(sample_quality_report.get("rejected_count") or 0) > 0:
        actions.append("inspect rejected_samples.csv and improve mask/bbox quality")
    if not replay_guard_report.get("passed"):
        actions.append("run replay evaluation before shadow promotion")
    if not dom_geometry_guard_report.get("geometry_guard_passed"):
        actions.append("provide DOM-only baseline/candidate geometry samples for guard evaluation")
    return actions
