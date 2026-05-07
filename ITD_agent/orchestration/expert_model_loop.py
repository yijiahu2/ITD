from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def diagnose_residual_profile(
    *,
    roi_assessment: dict[str, Any],
    expert_eval_info: dict[str, Any],
    failure_modes: list[str] | None = None,
) -> dict[str, Any]:
    triggers = set(roi_assessment.get("trigger_metrics") or [])
    modes = list(failure_modes or [])
    residual_types: list[str] = []
    if "mean_crown_width_error_ratio" in triggers:
        residual_types.append("boundary_or_crown_size_residual")
    if "closure_error_abs" in triggers:
        residual_types.append("semantic_gap_or_cover_residual")
    if "tree_count_error_ratio" in triggers:
        residual_types.append("count_split_merge_residual")
    for mode in modes:
        text = str(mode)
        if "阴影" in text or "shadow" in text:
            residual_types.append("terrain_shadow_confused")
        if "高估" in text or "over" in text:
            residual_types.append("over_split")
        if "低估" in text or "under" in text:
            residual_types.append("under_split")
    if not residual_types:
        residual_types.append("uncertain_residual")
    return {
        "residual_types": list(dict.fromkeys(residual_types)),
        "trigger_metrics": list(triggers),
        "current_score": expert_eval_info.get("current_score"),
        "previous_score": expert_eval_info.get("previous_score"),
    }


def build_expert_model_loop_trace(
    *,
    round_idx: int,
    roi_assessment: dict[str, Any],
    expert_plan: dict[str, Any],
    refine_summary: dict[str, Any],
    expert_eval_info: dict[str, Any],
    roi_decision: dict[str, Any],
    accepted: bool,
    acceptance_reason: str,
    failure_modes: list[str] | None = None,
) -> dict[str, Any]:
    expert_call_plan = expert_plan.get("expert_model_call_plan") or {}
    residual_profile = diagnose_residual_profile(
        roi_assessment=roi_assessment,
        expert_eval_info=expert_eval_info,
        failure_modes=failure_modes,
    )
    return {
        "loop_name": "expert_model_loop",
        "round_idx": round_idx,
        "stages": [
            "roi_extraction",
            "roi_input_context",
            "residual_diagnosis",
            "coarse_expert_family_routing",
            "family_parameter_optimization",
            "expert_execution",
            "expert_evaluation",
            "accept_reject",
        ],
        "roi_candidates": roi_assessment.get("candidate_rois") or [],
        "residual_error_profile": residual_profile,
        "expert_route_plan": {
            "preferred_expert_family": expert_call_plan.get("preferred_expert_family"),
            "preferred_expert_model": expert_call_plan.get("preferred_expert_model"),
            "candidate_expert_families": expert_call_plan.get("candidate_expert_families") or [],
            "candidate_models": expert_call_plan.get("candidate_models") or [],
            "routing_mode": expert_call_plan.get("routing_mode"),
            "selection_reason": expert_call_plan.get("selection_reason"),
        },
        "family_parameter_optimization": {
            "parameter_updates": expert_plan.get("parameter_updates") or {},
            "runtime_plan": expert_plan.get("runtime_plan") or {},
        },
        "expert_execution": {
            "refine_summary": refine_summary,
        },
        "expert_evaluation": {
            "assessment_phase": expert_eval_info.get("assessment_phase"),
            "current_score": expert_eval_info.get("current_score"),
            "previous_score": expert_eval_info.get("previous_score"),
            "roi_decision": roi_decision,
        },
        "accept_reject": {
            "accepted": accepted,
            "acceptance_reason": acceptance_reason,
            "failure_modes": failure_modes or [],
        },
    }


def save_expert_model_loop_trace(trace: dict[str, Any], output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    return str(path)
