from __future__ import annotations

from collections import Counter
from typing import Any


def _short_text(value: Any, limit: int = 120) -> str | None:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _compact_input_assessment_summary(run_summary: dict[str, Any]) -> dict[str, Any]:
    assessment = ((run_summary.get("evaluation_analysis") or {}).get("input_assessment") or {})
    scene_analysis = assessment.get("scene_analysis") or {}
    return {
        "readiness_score": assessment.get("readiness_score"),
        "modalities": assessment.get("modality_status") or {},
        "issues": list(assessment.get("issues") or [])[:5],
        "recommended_actions": list(assessment.get("recommended_actions") or [])[:5],
        "scene_analysis": {
            "forest_type": scene_analysis.get("forest_type"),
            "stand_condition": ((scene_analysis.get("stand_condition") or {}).get("labels") or []),
            "texture_labels": ((scene_analysis.get("image_texture_analysis") or {}).get("labels") or []),
        },
    }


def _compact_scene_profile(run_summary: dict[str, Any]) -> dict[str, Any]:
    input_assessment = ((run_summary.get("evaluation_analysis") or {}).get("input_assessment") or {})
    scene_analysis = input_assessment.get("scene_analysis") or {}
    texture = scene_analysis.get("image_texture_analysis") or {}
    quality = scene_analysis.get("image_quality_analysis") or {}
    run_meta = run_summary.get("run_meta") or {}
    terrain_info = run_meta.get("terrain_info") or {}
    data_processing = ((run_summary.get("data_processing") or {}).get("processing_summary") or {})
    image_profiles = data_processing.get("image_profiles") or []
    image_resolution = None
    if image_profiles:
        image_resolution = (image_profiles[0] or {}).get("resolution_x_m") or (image_profiles[0] or {}).get("resolution_y_m")
    return {
        "forest_type": scene_analysis.get("forest_type") or run_meta.get("forest_type"),
        "terrain_type": terrain_info.get("landform_type") or run_meta.get("terrain_type"),
        "image_resolution_m": image_resolution,
        "stand_condition_labels": ((scene_analysis.get("stand_condition") or {}).get("labels") or []),
        "texture_labels": texture.get("labels") or [],
        "image_texture_levels": texture.get("levels") or {},
        "quality_labels": quality.get("labels") or [],
        "image_quality_levels": quality.get("levels") or {},
    }


def _compact_metrics_summary(run_summary: dict[str, Any]) -> dict[str, Any]:
    metrics = (run_summary.get("metrics") or {}).copy()
    quality_score = None
    tree = _safe_float(metrics.get("tree_count_error_ratio"))
    crown = _safe_float(metrics.get("mean_crown_width_error_ratio"))
    closure = _safe_float(metrics.get("closure_error_abs"))
    density = _safe_float(metrics.get("density_error_abs"))
    if tree is not None and crown is not None and closure is not None:
        quality_score = tree + crown + closure + (density or 0.0) / 1000.0
    return {
        "tree_count_error_ratio": tree,
        "mean_crown_width_error_ratio": crown,
        "closure_error_abs": closure,
        "density_error_abs": density,
        "quality_score": quality_score,
    }


def _compact_top_problem_cases(run_summary: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    cases = ((run_summary.get("failure_analysis") or {}).get("top_problem_cases") or [])[:limit]
    compacted: list[dict[str, Any]] = []
    for case in cases:
        compacted.append(
            {
                "xiaoban_id": case.get("xiaoban_id"),
                "error_score": case.get("error_score"),
                "tree_count_error_abs": case.get("tree_count_error_abs"),
                "mean_crown_width_error_abs": case.get("mean_crown_width_error_abs"),
                "closure_error_abs": case.get("closure_error_abs"),
                "density_error_abs": case.get("density_error_abs"),
                "terrain_tags": [
                    item
                    for item in [
                        case.get("landform_type"),
                        case.get("slope_class"),
                        case.get("aspect_class"),
                        case.get("slope_position_class"),
                    ]
                    if item
                ],
            }
        )
    return compacted


def _compact_roi_round_summary(run_summary: dict[str, Any]) -> dict[str, Any]:
    planning = run_summary.get("planning_scheduler") or {}
    roi_rounds = planning.get("roi_rounds") or []
    preferred_models: list[str] = []
    for item in roi_rounds:
        preferred = (
            ((item.get("expert_model_call_plan") or item.get("child_model_call_plan") or {}).get("preferred_expert_model"))
            or ((item.get("expert_model_call_plan") or item.get("child_model_call_plan") or {}).get("preferred_child_model"))
            or ((item.get("roi_decision") or {}).get("preferred_expert_model"))
            or ((item.get("roi_decision") or {}).get("preferred_child_model"))
        )
        if preferred and preferred not in preferred_models:
            preferred_models.append(str(preferred))
    final_decision = ((run_summary.get("llm_gateway") or {}).get("roi_decision") or {})
    return {
        "roi_round_count": int(len(roi_rounds)),
        "stopped_by": _short_text(final_decision.get("reason"), 160),
        "decision_source": final_decision.get("decision_source"),
        "preferred_expert_models": preferred_models[:3],
    }


def _compact_memory_digest(memory_context: list[dict[str, Any]] | None, limit: int = 3) -> dict[str, Any]:
    rows = memory_context or []
    success_strategies: list[str] = []
    failure_modes: list[str] = []
    similar_success_count = 0
    similar_failure_count = 0
    for item in rows:
        memory_type = str(item.get("memory_type") or "")
        if memory_type == "successful_strategy":
            similar_success_count += 1
            strategy = _short_text(item.get("strategy_summary") or item.get("reason") or item.get("run_name"))
            if strategy and strategy not in success_strategies:
                success_strategies.append(strategy)
        elif memory_type == "failure_pattern":
            similar_failure_count += 1
            for mode in item.get("failure_modes") or []:
                short = _short_text(mode)
                if short and short not in failure_modes:
                    failure_modes.append(short)
    return {
        "similar_success_count": similar_success_count,
        "similar_failure_count": similar_failure_count,
        "top_success_strategies": success_strategies[:limit],
        "top_failure_modes": failure_modes[:limit],
    }


def _compact_finetune_pool_digest(finetune_context: list[dict[str, Any]] | None, limit: int = 3) -> dict[str, Any]:
    rows = finetune_context or []
    categories = [str(item.get("failure_category")) for item in rows if item.get("failure_category")]
    category_counts = Counter(categories)
    top_cases: list[dict[str, Any]] = []
    for item in rows[:limit]:
        top_cases.append(
            {
                "sample_id": item.get("sample_id"),
                "failure_category": item.get("failure_category"),
                "target_model_role": item.get("target_model_role"),
                "xiaoban_id": ((item.get("metadata") or {}).get("xiaoban_id")),
            }
        )
    return {
        "recent_failed_case_count": len(rows),
        "dominant_failure_categories": [name for name, _ in category_counts.most_common(limit)],
        "recent_case_briefs": top_cases,
    }


def build_run_retrospective_input(
    *,
    run_summary: dict[str, Any],
    memory_context: list[dict[str, Any]] | None,
    finetune_context: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return {
        "template_name": "run_retrospective",
        "template_version": "v1",
        "task_type": "retrospective",
        "run_name": run_summary.get("run_name"),
        "planning_stage": "retrospective",
        "context": {
            "scene_profile": _compact_scene_profile(run_summary),
            "input_assessment_summary": _compact_input_assessment_summary(run_summary),
            "final_metrics_summary": _compact_metrics_summary(run_summary),
            "roi_round_summary": _compact_roi_round_summary(run_summary),
            "top_problem_cases": _compact_top_problem_cases(run_summary, limit=5),
            "memory_digest": _compact_memory_digest(memory_context, limit=3),
            "finetune_pool_digest": _compact_finetune_pool_digest(finetune_context, limit=3),
        },
    }


def _build_run_retrospective_input(
    *,
    run_summary: dict[str, Any],
    memory_context: list[dict[str, Any]] | None,
    finetune_context: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return build_run_retrospective_input(
        run_summary=run_summary,
        memory_context=memory_context,
        finetune_context=finetune_context,
    )
