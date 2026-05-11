from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.finetune_pool.review.io_utils import write_json
from ITD_agent.training_loop.contracts import TrainingRunResult


def decide_model_promotion(
    *,
    cfg: dict[str, Any],
    training_result: TrainingRunResult,
    replay_guard_report: dict[str, Any],
    dom_geometry_guard_report: dict[str, Any],
    capability_profile: dict[str, Any] | None,
    evaluation: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    promotion_cfg = cfg.get("promotion") or {}
    allow_shadow = bool(promotion_cfg.get("allow_promote_to_shadow", True))
    allow_active = bool(promotion_cfg.get("allow_promote_to_active", False))
    if allow_active:
        decision = "reject"
        reason = "active_promotion_forbidden_in_v3"
    elif training_result.status not in {"completed", "recovered_ckpt"}:
        decision = "keep_candidate"
        reason = "training_not_completed"
    elif bool(promotion_cfg.get("require_replay_guard_pass", True)) and not replay_guard_report.get("passed"):
        decision = "rejected"
        reason = "replay_guard_failed"
    elif bool(promotion_cfg.get("require_dom_only_geometry_guard_pass", True)) and not dom_geometry_guard_report.get("geometry_guard_passed"):
        decision = "keep_candidate"
        reason = "dom_only_geometry_guard_not_passed"
    elif bool(promotion_cfg.get("require_capability_profile", True)) and not capability_profile:
        decision = "keep_candidate"
        reason = "capability_profile_missing"
    elif allow_shadow:
        decision = "promote_to_shadow"
        reason = "candidate_passed_v3_shadow_gate"
    else:
        decision = "keep_candidate"
        reason = "shadow_promotion_disabled"

    payload = {
        "decision": decision,
        "reason": reason,
        "allow_promote_to_active": False,
        "training_status": training_result.status,
        "replay_guard": replay_guard_report,
        "dom_only_geometry_guard": dom_geometry_guard_report,
        "capability_profile_path": (capability_profile or {}).get("profile_path"),
        "evaluation_delta": evaluation.get("delta") or {},
        "blocked_actions": ["promote_to_active", "replace_active_model", "update_active_route_map"],
        "next_recommended_actions": [
            "run shadow comparison in the next adaptive inference run",
            "review routing_update_candidate manually",
            "collect more replay and DOM-only geometry samples before active promotion",
        ],
    }
    root = Path(output_dir) / "promotion"
    write_json(root / "promotion_decision.json", payload)
    write_json(root / "promotion_report.json", payload)
    return payload


def status_for_registry(decision: dict[str, Any]) -> str:
    value = str(decision.get("decision") or "")
    if value == "promote_to_shadow":
        return "shadow"
    if value == "rejected":
        return "rejected"
    return "candidate"
