from __future__ import annotations

from typing import Any


DEFAULT_ROUTE_MAP = {
    "under_segmentation": "htc",
    "over_segmentation": "mask2former",
    "false_positive": "cascade_mask_rcnn",
    "false_negative": "maskdino",
}


def route_expert_model(level1_error_type: str, routing_policy: dict[str, Any] | None = None) -> dict[str, Any]:
    route_map = dict(DEFAULT_ROUTE_MAP)
    policy = routing_policy or {}
    expert_map = policy.get("expert_map") or {}
    for error_type, entry in expert_map.items():
        if isinstance(entry, dict) and entry.get("primary_expert"):
            route_map[str(error_type)] = str(entry["primary_expert"])
        elif isinstance(entry, str):
            route_map[str(error_type)] = entry
    route_map.update(policy.get("route_map") or {})
    expert_model = route_map.get(level1_error_type) or route_map.get("default") or "maskdino"
    return {
        "expert_model": expert_model,
        "routing_policy_version": policy.get("version", "v1_rule_based"),
        "routing_reason": f"rule_based_route_for_{level1_error_type}",
    }
