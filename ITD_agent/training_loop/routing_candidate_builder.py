from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.evolution.review.io_utils import write_json


def build_routing_update_candidate(
    *,
    cfg: dict[str, Any],
    capability_profile: dict[str, Any] | None,
    promotion_decision: dict[str, Any],
    replay_guard_report: dict[str, Any],
    dom_geometry_guard_report: dict[str, Any],
    family_cfg: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    routing_cfg = cfg.get("routing_candidate") or {}
    root = Path(output_dir) / "routing"
    if not bool(routing_cfg.get("enabled", True)) or not capability_profile:
        report = {"enabled": False, "reason": "routing_candidate_disabled_or_profile_missing"}
        write_json(root / "routing_candidate_report.json", report)
        return report

    target_failure = capability_profile.get("target_failure_category")
    model_version_id = capability_profile.get("model_version_id")
    requires_shadow = bool(routing_cfg.get("require_promotion_to_shadow", True))
    can_build = True
    reasons: list[str] = []
    if requires_shadow and promotion_decision.get("decision") != "promote_to_shadow":
        can_build = False
        reasons.append("promotion_to_shadow_required")
    if bool(routing_cfg.get("require_replay_guard_pass", True)) and not replay_guard_report.get("passed"):
        can_build = False
        reasons.append("replay_guard_not_passed")
    if bool(routing_cfg.get("require_geometry_guard_pass", True)) and not dom_geometry_guard_report.get("geometry_guard_passed"):
        can_build = False
        reasons.append("dom_only_geometry_guard_not_passed")

    candidate = {
        "candidate_id": f"routing_candidate_{target_failure or 'target'}_{model_version_id}",
        "source_model_version_id": model_version_id,
        "source_training_job_id": capability_profile.get("training_job_id"),
        "target_failure_category": target_failure,
        "target_expert_family": capability_profile.get("target_expert_family"),
        "recommended_route_update": {
            "failure_category": target_failure,
            "primary_expert_candidate": model_version_id,
            "fallback_expert_candidate": _fallback_expert(family_cfg),
            "activation_scope": "shadow_only",
        },
        "evidence": {
            "target_error_improved": (capability_profile.get("metric_delta_summary") or {}).get("target_error_improved"),
            "target_error_delta": (capability_profile.get("metric_delta_summary") or {}).get("target_error_delta"),
            "replay_guard_passed": replay_guard_report.get("passed"),
            "dom_only_geometry_guard_passed": dom_geometry_guard_report.get("geometry_guard_passed"),
            "promotion_decision": promotion_decision.get("decision"),
        },
        "risk_notes": [
            "not allowed to replace active route_map automatically",
            "requires human review before active routing update",
        ],
        "status": str(routing_cfg.get("status") or "pending_review") if can_build else "blocked",
        "blocked_reasons": reasons,
    }
    write_json(root / "routing_update_candidate.json", candidate)
    report = {"enabled": True, "candidate_built": can_build, "candidate_path": str(root / "routing_update_candidate.json"), "blocked_reasons": reasons}
    write_json(root / "routing_candidate_report.json", report)
    return report


def _fallback_expert(family_cfg: dict[str, Any]) -> str | None:
    priority = list(family_cfg.get("algorithms_priority") or [])
    return str(priority[1]) if len(priority) > 1 else (str(priority[0]) if priority else None)
