from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


def build_class_level_skill_records(*, review_run_id: str, source_run_id: str, contexts: list[Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    skill_cfg = cfg.get("skill_review") or {}
    min_support = int(skill_cfg.get("min_support_count", 3))
    status = str(skill_cfg.get("status_on_create") or "draft")
    if status == "active_hard_policy":
        status = "draft"
    by_family: dict[str, set[str]] = defaultdict(set)
    for context in contexts:
        for family, count in ((context.trajectory_summary.get("roi_summary") or {}).get("by_failure_family") or {}).items():
            if int(count) > 0:
                by_family[str(family)].add(context.trajectory_id)
    records: list[dict[str, Any]] = []
    for family, trajectory_ids in sorted(by_family.items()):
        if len(trajectory_ids) < min_support:
            continue
        skill_type = _skill_type_for_family(family)
        records.append(
            {
                "skill_id": f"skill_{review_run_id}_{family}",
                "skill_type": skill_type,
                "name": f"{family} review draft",
                "source_run_ids": [source_run_id],
                "source_trajectory_ids": sorted(trajectory_ids),
                "trigger_conditions": {"failure_family": family, "min_support_count": min_support},
                "recommended_action": {"mode": "readonly_suggestion", "description": f"Use review evidence to inspect {family} cases before policy activation."},
                "evidence_summary": {"support_count": len(trajectory_ids), "failure_family": family},
                "safety_constraints": {
                    "status_must_not_be": "active_hard_policy",
                    "review_does_not_modify_route_map": True,
                    "requires_human_review_for_activation": True,
                },
                "status": status,
                "version": "review.1",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return records


def _skill_type_for_family(family: str) -> str:
    return {
        "small_crown_recall": "training_sample_selection_skill",
        "false_positive_cleanup": "fusion_guard_skill",
        "crown_split": "expert_routing_skill",
        "crown_merge_cleanup": "expert_routing_skill",
    }.get(family, "geometry_failure_interpretation_skill")
