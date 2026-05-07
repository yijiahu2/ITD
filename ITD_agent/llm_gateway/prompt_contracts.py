from __future__ import annotations

from copy import deepcopy
from typing import Any


_PROMPT_CONTRACTS: dict[str, dict[str, Any]] = {
    "scene_profiler": {
        "allowed_inputs": ["online_scene_state"],
        "output_format": "json",
        "forbidden_behaviors": ["select_model", "write_training_plan", "freeform_parameters"],
        "required_outputs": ["scene_summary", "confidence", "missing_inputs"],
    },
    "parameter_planner": {
        "allowed_inputs": ["online_scene_state", "model_search_space", "public_dataset_prior_digest"],
        "output_format": "json",
        "forbidden_behaviors": ["invent_parameters_outside_search_space", "select_expert_without_gate"],
        "required_outputs": ["candidate_parameters", "selection_reason", "expected_tradeoffs"],
    },
    "expert_router": {
        "allowed_inputs": ["roi_scene_state", "residual_error_profile", "expert_model_search_space"],
        "output_format": "json",
        "forbidden_behaviors": ["route_without_residual_type", "skip_family_pre_route"],
        "required_outputs": ["preferred_expert_family", "preferred_expert_model", "rejection_reasons"],
    },
    "retrospective": {
        "allowed_inputs": ["learning_event", "trial_trace", "benchmark_result"],
        "output_format": "json",
        "forbidden_behaviors": ["write_success_without_metric_delta", "freeform_summary_only"],
        "required_outputs": ["learned_rule", "evidence_level", "write_decision"],
    },
}


def get_prompt_contract(name: str) -> dict[str, Any]:
    key = str(name or "").strip()
    if key not in _PROMPT_CONTRACTS:
        known = ", ".join(sorted(_PROMPT_CONTRACTS))
        raise KeyError(f"Unknown prompt contract: {name}. Known: {known}")
    return deepcopy(_PROMPT_CONTRACTS[key])


def list_prompt_contracts() -> dict[str, dict[str, Any]]:
    return deepcopy(_PROMPT_CONTRACTS)

