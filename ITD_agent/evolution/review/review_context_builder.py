from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReviewContext:
    source_run_id: str
    trajectory_id: str
    image_id: str
    trajectory_summary: dict[str, Any]
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    memory_candidates: list[dict[str, Any]] = field(default_factory=list)
    skill_candidates: list[dict[str, Any]] = field(default_factory=list)
    training_candidates: list[dict[str, Any]] = field(default_factory=list)
    routing_update_candidates: list[dict[str, Any]] = field(default_factory=list)
    distillation_candidates: list[dict[str, Any]] = field(default_factory=list)
    roi_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    expert_task_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_llm_review_context(context: ReviewContext, reviewer_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    compression_cfg = cfg.get("trajectory_compression") or {}
    top_k = int(compression_cfg.get("context_top_k_per_candidate_type", 16) or 16)
    top_k_per_error = int(compression_cfg.get("context_top_k_per_error_type", 4) or 4)
    base = _base_context(context)
    base["pending_candidate_summary"] = build_pending_candidate_summary(context, top_k=top_k, top_k_per_error_type=top_k_per_error)
    if reviewer_name == "memory":
        return {
            **base,
            "reviewer": "memory",
            "memory_candidates": _select_by_severity(context.memory_candidates, context.roi_by_id, top_k),
            "roi_evidence_sample": _sample_roi_evidence(context.roi_by_id, top_k),
            "expert_outcome_summary": _expert_outcome_summary(context),
        }
    if reviewer_name == "skill":
        return {
            **base,
            "reviewer": "skill",
            "skill_candidates": _select_by_severity(context.skill_candidates, context.roi_by_id, top_k),
            "class_level_evidence": _class_level_evidence(context),
            "expert_outcome_summary": _expert_outcome_summary(context),
        }
    if reviewer_name == "finetune":
        return {
            **base,
            "reviewer": "finetune",
            "training_candidate_summary": _candidate_summary(context.training_candidates, context.roi_by_id),
            "selected_training_candidates": _select_by_error_type(context.training_candidates, context.roi_by_id, top_k_per_error, top_k),
        }
    if reviewer_name == "routing":
        return {
            **base,
            "reviewer": "routing",
            "routing_candidate_summary": _candidate_summary(context.routing_update_candidates, context.roi_by_id),
            "selected_routing_candidates": _select_by_severity(context.routing_update_candidates, context.roi_by_id, top_k),
            "expert_outcome_summary": _expert_outcome_summary(context),
        }
    if reviewer_name == "distillation":
        return {
            **base,
            "reviewer": "distillation",
            "distillation_candidate_summary": _candidate_summary(context.distillation_candidates, context.roi_by_id),
            "selected_distillation_candidates": _select_by_severity(context.distillation_candidates, context.roi_by_id, top_k),
        }
    return base


def build_pending_candidate_summary(context: ReviewContext, *, top_k: int = 16, top_k_per_error_type: int = 4) -> dict[str, Any]:
    return {
        "counts": {
            "memory": len(context.memory_candidates),
            "skill": len(context.skill_candidates),
            "training": len(context.training_candidates),
            "routing": len(context.routing_update_candidates),
            "distillation": len(context.distillation_candidates),
        },
        "training": {
            **_candidate_summary(context.training_candidates, context.roi_by_id),
            "selected_candidate_refs": _candidate_refs(_select_by_error_type(context.training_candidates, context.roi_by_id, top_k_per_error_type, top_k)),
        },
        "routing": {
            **_candidate_summary(context.routing_update_candidates, context.roi_by_id),
            "selected_candidate_refs": _candidate_refs(_select_by_severity(context.routing_update_candidates, context.roi_by_id, top_k)),
        },
        "distillation": {
            **_candidate_summary(context.distillation_candidates, context.roi_by_id),
            "selected_candidate_refs": _candidate_refs(_select_by_severity(context.distillation_candidates, context.roi_by_id, top_k)),
        },
        "full_pending_candidates_ref": {
            "source": "v1_trajectory_artifact",
            "path": (context.artifact_refs.get("trajectory_json") or {}).get("path"),
            "json_pointer": "/pending_review_candidates",
        },
    }


def build_review_context(
    *,
    trajectory: dict[str, Any],
    trajectory_summary: dict[str, Any],
    artifact_refs: dict[str, Any],
) -> ReviewContext:
    pending = trajectory.get("pending_review_candidates") or {}
    tasks = list((trajectory.get("expert_task_stage") or {}).get("expert_tasks") or [])
    reviews = list((trajectory.get("expert_review_stage") or {}).get("expert_reviews") or [])
    task_by_id = {str(task.get("expert_task_id")): task for task in tasks}
    routing_candidates: list[dict[str, Any]] = []
    for review in reviews:
        task = task_by_id.get(str(review.get("expert_task_id"))) or {}
        routing_candidates.append(
            {
                "candidate_id": f"routecand_{review.get('review_id')}",
                "trajectory_id": trajectory.get("trajectory_id"),
                "level1_error_type": task.get("level1_error_type"),
                "failure_family": task.get("failure_family"),
                "expert_model": task.get("expert_model"),
                "expert_decision": review.get("decision"),
                "improvement": review.get("improvement") or {},
                "safety": review.get("safety") or {},
            }
        )
    return ReviewContext(
        source_run_id=str(trajectory.get("run_id")),
        trajectory_id=str(trajectory.get("trajectory_id")),
        image_id=str(trajectory.get("image_id")),
        trajectory_summary=trajectory_summary,
        artifact_refs=artifact_refs,
        memory_candidates=list(pending.get("memory_candidates") or []),
        skill_candidates=list(pending.get("skill_candidates") or []),
        training_candidates=list(pending.get("training_candidates") or []),
        routing_update_candidates=routing_candidates,
        distillation_candidates=list(pending.get("distillation_candidates") or []),
        roi_by_id={str(roi.get("roi_id")): roi for roi in (trajectory.get("roi_stage") or {}).get("roi_candidates") or []},
        expert_task_by_id=task_by_id,
    )


def _base_context(context: ReviewContext) -> dict[str, Any]:
    return {
        "source_run_id": context.source_run_id,
        "trajectory_id": context.trajectory_id,
        "image_id": context.image_id,
        "trajectory_summary": context.trajectory_summary,
        "artifact_refs": context.artifact_refs,
        "candidate_manifest_refs": {
            "full_pending_candidates_ref": {
                "source": "v1_trajectory_artifact",
                "path": (context.artifact_refs.get("trajectory_json") or {}).get("path"),
                "json_pointer": "/pending_review_candidates",
            }
        },
        "context_compression": {
            "pending_candidates_embedded": False,
            "roi_map_embedded": False,
            "selection_policy": "top severity candidates with per-error coverage",
        },
    }


def _candidate_summary(candidates: list[dict[str, Any]], roi_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(candidates),
        "by_failure_category": _count_by(candidates, "failure_category", roi_by_id=roi_by_id, roi_key="level1_error_type"),
        "by_target_model_role": _count_by(candidates, "target_model_role"),
        "by_quality_status": _count_by(candidates, "quality_status"),
        "by_sample_type": _count_by(candidates, "sample_type"),
    }


def _count_by(
    candidates: list[dict[str, Any]],
    key: str,
    *,
    roi_by_id: dict[str, dict[str, Any]] | None = None,
    roi_key: str | None = None,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in candidates:
        value = item.get(key)
        if value is None and roi_by_id is not None and roi_key:
            value = (roi_by_id.get(str(item.get("roi_id"))) or {}).get(roi_key)
        counts[str(value or "unknown")] += 1
    return dict(counts)


def _select_by_error_type(
    candidates: list[dict[str, Any]],
    roi_by_id: dict[str, dict[str, Any]],
    per_error: int,
    max_total: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for item in _rank_by_severity(candidates, roi_by_id):
        error_type = _candidate_error_type(item, roi_by_id)
        if counts[error_type] >= per_error:
            continue
        counts[error_type] += 1
        selected.append(_candidate_brief(item, roi_by_id))
        if len(selected) >= max_total:
            break
    return selected


def _select_by_severity(candidates: list[dict[str, Any]], roi_by_id: dict[str, dict[str, Any]], max_total: int) -> list[dict[str, Any]]:
    return [_candidate_brief(item, roi_by_id) for item in _rank_by_severity(candidates, roi_by_id)[:max_total]]


def _rank_by_severity(candidates: list[dict[str, Any]], roi_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda item: float((roi_by_id.get(str(item.get("roi_id"))) or {}).get("severity_score") or 0.0), reverse=True)


def _candidate_brief(candidate: dict[str, Any], roi_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    roi = roi_by_id.get(str(candidate.get("roi_id"))) or {}
    return {
        "candidate_id": candidate.get("candidate_id") or _fallback_candidate_id(candidate),
        "trajectory_id": candidate.get("trajectory_id"),
        "roi_id": candidate.get("roi_id"),
        "failure_category": candidate.get("failure_category") or candidate.get("level1_error_type") or roi.get("level1_error_type"),
        "target_model_role": candidate.get("target_model_role"),
        "sample_type": candidate.get("sample_type"),
        "quality_status": candidate.get("quality_status"),
        "expert_model": candidate.get("expert_model"),
        "expert_decision": candidate.get("expert_decision"),
        "roi_evidence": _roi_brief(roi),
    }


def _fallback_candidate_id(candidate: dict[str, Any]) -> str:
    roi_id = candidate.get("roi_id") or "unknown_roi"
    trajectory_id = candidate.get("trajectory_id") or "unknown_trajectory"
    return f"candidate_{trajectory_id}_{roi_id}"


def _candidate_refs(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": item.get("candidate_id"),
            "roi_id": item.get("roi_id"),
            "failure_category": item.get("failure_category"),
            "severity_score": (item.get("roi_evidence") or {}).get("severity_score"),
        }
        for item in candidates
    ]


def _candidate_error_type(candidate: dict[str, Any], roi_by_id: dict[str, dict[str, Any]]) -> str:
    roi = roi_by_id.get(str(candidate.get("roi_id"))) or {}
    return str(candidate.get("failure_category") or roi.get("level1_error_type") or "unknown")


def _roi_brief(roi: dict[str, Any]) -> dict[str, Any]:
    if not roi:
        return {}
    return {
        "roi_id": roi.get("roi_id"),
        "level1_error_type": roi.get("level1_error_type"),
        "failure_family": roi.get("failure_family"),
        "severity_score": roi.get("severity_score"),
        "confidence_level": roi.get("confidence_level"),
        "bbox_px": roi.get("bbox_px"),
        "affected_gt_count": len(roi.get("affected_gt_ids") or []),
        "affected_pred_count": len(roi.get("affected_pred_ids") or []),
        "review_status": roi.get("review_status"),
        "expert_eligible": roi.get("expert_eligible"),
        "training_eligible": roi.get("training_eligible"),
        "distill_eligible": roi.get("distill_eligible"),
    }


def _sample_roi_evidence(roi_by_id: dict[str, dict[str, Any]], max_total: int) -> list[dict[str, Any]]:
    ranked = sorted(roi_by_id.values(), key=lambda roi: float(roi.get("severity_score") or 0.0), reverse=True)
    return [_roi_brief(roi) for roi in ranked[:max_total]]


def _expert_outcome_summary(context: ReviewContext) -> dict[str, Any]:
    return {
        "routing_candidate_count": len(context.routing_update_candidates),
        "by_expert_model": _count_by(context.routing_update_candidates, "expert_model"),
        "by_expert_decision": _count_by(context.routing_update_candidates, "expert_decision"),
        "by_error_type": _count_by(context.routing_update_candidates, "level1_error_type"),
    }


def _class_level_evidence(context: ReviewContext) -> dict[str, Any]:
    rois = list(context.roi_by_id.values())
    return {
        "roi_count": len(rois),
        "by_error_type": dict(Counter(str(roi.get("level1_error_type") or "unknown") for roi in rois)),
        "by_failure_family": dict(Counter(str(roi.get("failure_family") or "unknown") for roi in rois)),
        "high_severity_roi_refs": [
            {
                "roi_id": roi.get("roi_id"),
                "level1_error_type": roi.get("level1_error_type"),
                "failure_family": roi.get("failure_family"),
                "severity_score": roi.get("severity_score"),
            }
            for roi in sorted(rois, key=lambda item: float(item.get("severity_score") or 0.0), reverse=True)[:8]
        ],
    }
