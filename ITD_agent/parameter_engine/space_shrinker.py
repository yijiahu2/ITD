from __future__ import annotations

from copy import deepcopy
from typing import Any


def shrink_search_space_with_skill(
    *,
    base_space: dict[str, Any],
    skill_context: dict[str, Any],
    failure_family: str | None = None,
) -> dict[str, Any]:
    shrunk = deepcopy(base_space)
    matched_skills = skill_context.get("matched_skills") or []

    for skill in matched_skills:
        action = skill.get("recommended_action") or {}
        parameter_space = action.get("parameter_space") or {}
        shrink_rules = parameter_space.get("shrink") or {}

        for param_name, rule in shrink_rules.items():
            _apply_shrink_rule(shrunk, param_name, rule)

    shrunk["_shrink_metadata"] = {
        "source": "skill_context",
        "matched_skill_ids": [s.get("skill_id") for s in matched_skills],
        "failure_family": failure_family,
    }
    return shrunk


def _apply_shrink_rule(space: dict[str, Any], param_name: str, rule: dict[str, Any]) -> None:
    for section_name in ["decision_params", "deployment_params"]:
        section = space.get(section_name) or {}
        if param_name not in section:
            continue

        spec = section[param_name]
        direction = rule.get("direction")

        if "range" in spec and isinstance(spec["range"], list) and len(spec["range"]) == 2:
            lo, hi = float(spec["range"][0]), float(spec["range"][1])
            mid = (lo + hi) / 2.0
            if direction == "decrease":
                spec["range"] = [lo, mid]
            elif direction == "increase":
                spec["range"] = [mid, hi]
            section[param_name] = spec

        elif "values" in spec and isinstance(spec["values"], list):
            values = spec["values"]
            if not values:
                continue
            if direction == "decrease":
                spec["values"] = values[: max(1, len(values) // 2)]
            elif direction == "increase":
                spec["values"] = values[len(values) // 2 :]
            section[param_name] = spec
