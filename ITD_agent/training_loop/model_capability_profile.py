from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.evolution.review.io_utils import write_json
from ITD_agent.training_loop.contracts import TrainingPlan, TrainingRunResult


def build_model_capability_profile(
    *,
    plan: TrainingPlan,
    training_result: TrainingRunResult,
    evaluation: dict[str, Any],
    replay_guard_report: dict[str, Any],
    dom_geometry_guard_report: dict[str, Any],
    family_cfg: dict[str, Any],
    output_dir: str | Path,
    model_version_id: str,
) -> dict[str, Any]:
    delta = _metric_delta(evaluation)
    target_failure = plan.failure_category
    replay_passed = bool(replay_guard_report.get("passed"))
    geometry_passed = bool(dom_geometry_guard_report.get("geometry_guard_passed"))
    allowed_status = "shadow" if training_result.status in {"completed", "recovered_ckpt"} and replay_passed and geometry_passed else "candidate"
    profile = {
        "model_version_id": model_version_id,
        "model_id": plan.target_model_id,
        "model_role": plan.target_model_role,
        "target_expert_family": plan.target_expert_family,
        "target_failure_category": target_failure,
        "training_job_id": plan.training_job_id,
        "base_checkpoint": (plan.metadata.get("family_training_defaults") or {}).get("init_checkpoint"),
        "new_checkpoint": training_result.best_checkpoint_path,
        "training_objective": {
            "primary_goal": f"reduce_{target_failure or 'target_failure'}",
            "secondary_goals": _secondary_goals(plan, family_cfg),
        },
        "metric_delta_summary": {
            "target_error_improved": _target_error_improved(delta),
            "target_error_delta": delta.get("target_error_delta"),
            "coco_ap_delta": delta.get("ap_50_95"),
            "coco_ap50_delta": delta.get("ap50"),
            "precision_delta": delta.get("precision"),
            "recall_delta": delta.get("recall"),
            "geometry_anomaly_delta": _geometry_anomaly_delta(dom_geometry_guard_report),
            "replay_guard_passed": replay_passed,
            "dom_only_geometry_guard_passed": geometry_passed,
        },
        "strengths": [item for item in [plan.target_expert_family, target_failure, *(family_cfg.get("prior_axes") or [])] if item],
        "weaknesses": _weaknesses(replay_guard_report, dom_geometry_guard_report),
        "recommended_usage": {
            "allowed_status": allowed_status,
            "recommended_model_role": plan.target_model_role,
            "recommended_failure_categories": [target_failure] if target_failure else [],
            "recommended_expert_families": [plan.target_expert_family] if plan.target_expert_family else [],
            "not_recommended_for": ["general_main_model_replacement", "active_route_map_update_without_review"],
        },
        "routing_evidence": {
            "can_generate_routing_candidate": allowed_status == "shadow",
            "routing_scope": f"{target_failure}_only" if target_failure else "target_failure_only",
            "requires_human_review": True,
        },
        "traceability": {
            "training_plan_path": str(Path(plan.output_dir) / "training_plan.json"),
            "generated_config_path": plan.generated_config_path,
            "dataset_bundle_dir": plan.metadata.get("dataset_bundle_dir"),
        },
    }
    path = write_json(Path(output_dir) / "model_registry" / "capability_profiles" / f"{model_version_id}_capability_profile.json", profile)
    profile["profile_path"] = path
    return profile


def _metric_delta(evaluation: dict[str, Any]) -> dict[str, Any]:
    delta = evaluation.get("delta") or {}
    if isinstance(delta, dict):
        nested = delta.get("delta")
        if isinstance(nested, dict):
            return nested
    return {}


def _target_error_improved(delta: dict[str, Any]) -> bool:
    value = delta.get("target_error_delta")
    if value is None:
        return False
    try:
        return float(value) < 0.0
    except (TypeError, ValueError):
        return False


def _geometry_anomaly_delta(report: dict[str, Any]) -> float | None:
    values = []
    for value in (report.get("geometry_delta") or {}).values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def _secondary_goals(plan: TrainingPlan, family_cfg: dict[str, Any]) -> list[str]:
    goals = [f"improve_{axis}_cases" for axis in family_cfg.get("prior_axes") or []]
    if plan.target_expert_family:
        goals.append(f"improve_{plan.target_expert_family}_roi")
    return goals


def _weaknesses(replay_guard_report: dict[str, Any], dom_geometry_guard_report: dict[str, Any]) -> list[str]:
    weaknesses: list[str] = []
    if not replay_guard_report.get("passed"):
        weaknesses.append("replay_guard_not_passed")
    if not dom_geometry_guard_report.get("geometry_guard_passed"):
        weaknesses.append("dom_only_geometry_guard_not_passed")
    weaknesses.extend(str(item) for item in dom_geometry_guard_report.get("warning_items") or [])
    return weaknesses
