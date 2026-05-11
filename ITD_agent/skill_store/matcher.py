from __future__ import annotations

import json
from typing import Any


def match_skill_context(
    *,
    skills: list[dict[str, Any]],
    scene_profile: dict[str, Any],
    evaluation_metrics: dict[str, Any],
    roi_assessment: dict[str, Any],
    failure_pattern_context: list[dict[str, Any]],
    max_items: int = 5,
) -> dict[str, Any]:
    matched = []
    active_failure_families = _collect_failure_families(roi_assessment, failure_pattern_context)

    for skill in skills:
        trigger_conditions = _safe_json(skill.get("trigger_conditions_json") or skill.get("trigger_conditions") or {})
        evidence = _safe_json(skill.get("evidence_summary_json") or skill.get("evidence_summary") or {})
        recommended = _safe_json(skill.get("recommended_action_json") or skill.get("recommended_action") or {})

        failure_family = trigger_conditions.get("failure_family") or evidence.get("failure_family")
        score = 0.0
        reasons = []

        if failure_family and failure_family in active_failure_families:
            score += 0.6
            reasons.append(f"matched_failure_family:{failure_family}")

        if skill.get("status") == "active":
            score += 0.2
        elif skill.get("status") == "shadow":
            score += 0.1

        support_count = int(evidence.get("support_count") or 0)
        if support_count >= 3:
            score += 0.1

        if score > 0:
            matched.append(
                {
                    "skill_id": skill.get("skill_id"),
                    "skill_type": skill.get("skill_type"),
                    "status": skill.get("status"),
                    "score": round(score, 4),
                    "matched_reasons": reasons,
                    "recommended_action": recommended,
                    "evidence_summary": evidence,
                }
            )

    matched = sorted(matched, key=lambda x: x["score"], reverse=True)[:max_items]
    return {
        "matched_skill_count": len(matched),
        "matched_skills": matched,
        "application_mode": "context_only_readonly_suggestion",
    }


def _collect_failure_families(roi_assessment: dict[str, Any], failure_patterns: list[dict[str, Any]]) -> set[str]:
    out = set()
    for key in ["failure_family", "target_failure_family"]:
        if roi_assessment.get(key):
            out.add(str(roi_assessment[key]))
    for item in failure_patterns:
        if item.get("failure_family"):
            out.add(str(item["failure_family"]))
    return out


def _safe_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}
