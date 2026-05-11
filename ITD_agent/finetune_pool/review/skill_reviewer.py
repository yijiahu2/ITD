from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


def build_class_level_skill_records(*, review_run_id: str, source_run_id: str, contexts: list[Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    skill_cfg = cfg.get("skill_review") or {}
    status = str(skill_cfg.get("status_on_create") or "draft")
    if status == "active_hard_policy":
        status = "draft"
    records: list[dict[str, Any]] = []
    records.extend(_build_repeated_failure_skills(review_run_id=review_run_id, source_run_id=source_run_id, contexts=contexts, cfg=skill_cfg, status=status))
    records.extend(_build_periodic_nudge_skills(review_run_id=review_run_id, source_run_id=source_run_id, contexts=contexts, cfg=skill_cfg, status=status))
    records.extend(_build_expert_success_skills(review_run_id=review_run_id, source_run_id=source_run_id, contexts=contexts, cfg=skill_cfg, status=status))
    records.extend(_build_fusion_guard_skills(review_run_id=review_run_id, source_run_id=source_run_id, contexts=contexts, cfg=skill_cfg, status=status))
    return _dedupe_skill_records(records)


def _build_repeated_failure_skills(*, review_run_id: str, source_run_id: str, contexts: list[Any], cfg: dict[str, Any], status: str) -> list[dict[str, Any]]:
    min_support = int(cfg.get("min_support_count", 3))
    by_family: dict[str, set[str]] = defaultdict(set)
    support_by_family: dict[str, int] = defaultdict(int)
    for context in contexts:
        for family, count in ((context.trajectory_summary.get("roi_summary") or {}).get("by_failure_family") or {}).items():
            if int(count) > 0:
                by_family[str(family)].add(context.trajectory_id)
                support_by_family[str(family)] += int(count)
    records: list[dict[str, Any]] = []
    for family, trajectory_ids in sorted(by_family.items()):
        support_count = int(support_by_family.get(family) or 0)
        if support_count < min_support:
            continue
        records.append(_build_record(
            review_run_id=review_run_id,
            source_run_id=source_run_id,
            skill_suffix=family,
            skill_type=_skill_type_for_family(family),
            name=f"{family} review draft",
            source_trajectory_ids=sorted(trajectory_ids),
            trigger_conditions={"failure_family": family, "min_support_count": min_support, "trigger_type": "repeated_failure"},
            evidence_summary={"support_count": support_count, "trajectory_count": len(trajectory_ids), "failure_family": family},
            recommended_description=f"Use review evidence to inspect {family} cases before policy activation.",
            status=status,
        ))
    return records


def _build_periodic_nudge_skills(*, review_run_id: str, source_run_id: str, contexts: list[Any], cfg: dict[str, Any], status: str) -> list[dict[str, Any]]:
    periodic_cfg = cfg.get("periodic_nudge") or {}
    if not bool(periodic_cfg.get("enabled", False)):
        return []
    every_n = int(periodic_cfg.get("every_n_trajectories", 10) or 10)
    if len(contexts) < every_n:
        return []
    min_trajectory_count = int(periodic_cfg.get("min_trajectory_count", every_n) or every_n)
    if len(contexts) < min_trajectory_count:
        return []
    return [
        _build_record(
            review_run_id=review_run_id,
            source_run_id=source_run_id,
            skill_suffix="periodic_nudge",
            skill_type="failure_interpretation_skill",
            name="periodic review nudge",
            source_trajectory_ids=sorted(context.trajectory_id for context in contexts),
            trigger_conditions={"trigger_type": "periodic_nudge", "every_n_trajectories": every_n},
            evidence_summary={"support_count": len(contexts), "trajectory_count": len(contexts)},
            recommended_description="Inspect accumulated trajectories for new recurring patterns worth codifying into memory or skills.",
            status=status,
        )
    ]


def _build_expert_success_skills(*, review_run_id: str, source_run_id: str, contexts: list[Any], cfg: dict[str, Any], status: str) -> list[dict[str, Any]]:
    success_cfg = cfg.get("expert_success") or {}
    if not bool(success_cfg.get("enabled", False)):
        return []
    min_support = int(success_cfg.get("min_support_count", 2) or 2)
    min_gain = float(success_cfg.get("min_score_improvement", 0.03) or 0.03)
    by_model: dict[str, set[str]] = defaultdict(set)
    for context in contexts:
        for candidate in context.routing_update_candidates:
            gain = float((candidate.get("improvement") or {}).get("score_gain") or 0.0)
            if candidate.get("expert_decision") == "accept" and gain >= min_gain:
                by_model[str(candidate.get("expert_model") or "expert")].add(context.trajectory_id)
    records: list[dict[str, Any]] = []
    for model_name, trajectory_ids in sorted(by_model.items()):
        if len(trajectory_ids) < min_support:
            continue
        records.append(_build_record(
            review_run_id=review_run_id,
            source_run_id=source_run_id,
            skill_suffix=f"expert_success_{model_name}",
            skill_type="expert_routing_skill",
            name=f"{model_name} expert success pattern",
            source_trajectory_ids=sorted(trajectory_ids),
            trigger_conditions={"trigger_type": "expert_success", "expert_model": model_name, "min_score_improvement": min_gain},
            evidence_summary={"support_count": len(trajectory_ids), "expert_model": model_name},
            recommended_description=f"Prefer {model_name} when the same failure pattern recurs with positive review gains.",
            status=status,
        ))
    return records


def _build_fusion_guard_skills(*, review_run_id: str, source_run_id: str, contexts: list[Any], cfg: dict[str, Any], status: str) -> list[dict[str, Any]]:
    guard_cfg = cfg.get("fusion_guard") or {}
    if not bool(guard_cfg.get("enabled", False)):
        return []
    min_rejections = int(guard_cfg.get("min_rejected_refinement_count", 2) or 2)
    rejected_contexts: list[str] = []
    for context in contexts:
        rejected = any(candidate.get("expert_decision") == "reject" for candidate in context.routing_update_candidates)
        if rejected:
            rejected_contexts.append(context.trajectory_id)
    if len(rejected_contexts) < min_rejections:
        return []
    return [
        _build_record(
            review_run_id=review_run_id,
            source_run_id=source_run_id,
            skill_suffix="fusion_guard",
            skill_type="fusion_guard_skill",
            name="fusion rollback guard",
            source_trajectory_ids=sorted(rejected_contexts),
            trigger_conditions={"trigger_type": "fusion_guard", "min_rejected_refinement_count": min_rejections},
            evidence_summary={"support_count": len(rejected_contexts), "rejected_refinement_count": len(rejected_contexts)},
            recommended_description="Add a guardrail when expert refinement repeatedly gets rejected or rolled back.",
            status=status,
        )
    ]


def _build_record(
    *,
    review_run_id: str,
    source_run_id: str,
    skill_suffix: str,
    skill_type: str,
    name: str,
    source_trajectory_ids: list[str],
    trigger_conditions: dict[str, Any],
    evidence_summary: dict[str, Any],
    recommended_description: str,
    status: str,
) -> dict[str, Any]:
    return {
        "skill_id": f"skill_{review_run_id}_{skill_suffix}",
        "skill_type": skill_type,
        "name": name,
        "source_run_ids": [source_run_id],
        "source_trajectory_ids": source_trajectory_ids,
        "trigger_conditions": trigger_conditions,
        "recommended_action": {"mode": "readonly_suggestion", "description": recommended_description},
        "evidence_summary": evidence_summary,
        "safety_constraints": {
            "status_must_not_be": "active_hard_policy",
            "review_does_not_modify_route_map": True,
            "requires_human_review_for_activation": True,
        },
        "status": status,
        "version": "review.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _dedupe_skill_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        skill_id = str(record.get("skill_id") or "")
        if not skill_id or skill_id in seen:
            continue
        seen.add(skill_id)
        deduped.append(record)
    return deduped


def _skill_type_for_family(family: str) -> str:
    return {
        "small_crown_recall": "training_sample_selection_skill",
        "false_positive_cleanup": "fusion_guard_skill",
        "crown_split": "expert_routing_skill",
        "crown_merge_cleanup": "expert_routing_skill",
    }.get(family, "geometry_failure_interpretation_skill")
