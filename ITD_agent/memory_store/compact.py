from __future__ import annotations

from copy import deepcopy
from typing import Any

from ITD_agent.common.values import safe_float


def _copy_selected(source: dict[str, Any] | None, keys: list[str]) -> dict[str, Any]:
    source = source or {}
    return {key: deepcopy(source[key]) for key in keys if key in source}


def _limit_list(values: list[Any] | None, limit: int = 5) -> list[Any]:
    if not isinstance(values, list):
        return []
    return deepcopy(values[:limit])


def compact_gateway_trace(trace: dict[str, Any] | None) -> dict[str, Any]:
    trace = trace or {}
    if not isinstance(trace, dict):
        return {}
    return _copy_selected(
        trace,
        [
            "task_type",
            "status",
            "provider",
            "model",
            "error",
            "fallback_used",
        ],
    )


def compact_scheduler_context(context: dict[str, Any] | None) -> dict[str, Any]:
    context = context or {}
    if not isinstance(context, dict):
        return {}

    input_assessment = context.get("input_assessment") or {}
    scene_analysis = input_assessment.get("scene_analysis") or {}
    terrain_analysis = scene_analysis.get("terrain_analysis") or {}
    recommendation = context.get("segmentation_parameter_recommendation") or {}

    return {
        "run_name": context.get("run_name"),
        "planning_stage": context.get("planning_stage"),
        "scene_profile": deepcopy(context.get("scene_profile") or {}),
        "current_parameters": deepcopy(context.get("current_parameters") or {}),
        "evaluation_metrics": deepcopy(context.get("evaluation_metrics") or {}),
        "summary_snapshot": deepcopy(context.get("summary_snapshot") or {}),
        "details_summary": {
            "top_problem_case_count": len((context.get("details_summary") or {}).get("top_k_reference_units") or []),
            "top_problem_cases": _limit_list((context.get("details_summary") or {}).get("top_k_reference_units") or [], limit=3),
        },
        "image_texture_analysis": deepcopy(context.get("image_texture_analysis") or {}),
        "image_quality_analysis": deepcopy(context.get("image_quality_analysis") or {}),
        "terrain_analysis": {
            "labels": deepcopy(terrain_analysis.get("labels") or []),
            "policy": deepcopy(terrain_analysis.get("policy") or {}),
            "global_background": _copy_selected(
                terrain_analysis.get("global_background") or {},
                ["landform_type", "slope_class", "aspect_class", "slope_position_class", "slope_mean_deg"],
            ),
            "dom_context": _copy_selected(
                terrain_analysis.get("dom_context") or {},
                ["landform_type", "slope_class", "aspect_class", "slope_position_class", "slope_mean_deg"],
            ),
        },
        "memory_context_digest": {
            "recent_success_count": len(context.get("memory_store_context") or []),
            "recent_failure_count": len(context.get("failure_pattern_context") or []),
            "recent_execution_count": len(context.get("execution_trace_context") or []),
            "similar_scene_count": len(context.get("scene_similar_memory_context") or []),
            "finetune_case_count": len(context.get("finetune_pool_recent_cases") or []),
        },
        "segmentation_parameter_recommendation": {
            "model_family": recommendation.get("model_family"),
            "parameter_updates": deepcopy(recommendation.get("parameter_updates") or {}),
            "confidence": recommendation.get("confidence"),
            "evidence": _copy_selected(
                recommendation.get("evidence") or {},
                [
                    "crown_width_m",
                    "crown_width_px",
                    "density_mean",
                    "closure_mean",
                    "resolution_m",
                    "scene_tags",
                    "texture_levels",
                    "quality_levels",
                    "terrain_labels",
                    "global_terrain_background",
                    "dom_terrain_context",
                ],
            ),
        },
    }


def compact_roi_refine_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    plan = plan or {}
    if not isinstance(plan, dict):
        return {}
    compacted = _copy_selected(
        plan,
        ["enabled", "use_llm", "max_rounds", "top_k", "buffer_m", "strategy_mode", "preferred_expert_model"],
    )
    compacted["candidate_expert_models"] = _limit_list(plan.get("candidate_expert_models") or [], limit=5)
    compacted["selection_rules"] = _limit_list(plan.get("selection_rules") or [], limit=4)
    compacted["stop_rules"] = _limit_list(plan.get("stop_rules") or [], limit=4)
    return compacted


def compact_expert_model_call_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    plan = plan or {}
    if not isinstance(plan, dict):
        return {}
    compacted = _copy_selected(plan, ["enabled", "planning_stage", "routing_mode", "preferred_expert_model", "selection_reason"])
    compacted["candidate_models"] = _limit_list(plan.get("candidate_models") or [], limit=5)
    compacted["routing_rules"] = _limit_list(plan.get("routing_rules") or [], limit=4)
    compacted["escalation_rules"] = _limit_list(plan.get("escalation_rules") or [], limit=4)
    compacted["candidate_profiles"] = _limit_list(plan.get("candidate_profiles") or [], limit=3)
    return compacted


def compact_knowledge_embedding_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    plan = plan or {}
    if not isinstance(plan, dict):
        return {}
    config_hints = plan.get("config_hints") or {}
    return {
        "enabled": bool(plan.get("enabled", False)),
        "knowledge_source_count": len(plan.get("knowledge_sources") or []),
        "backbone_rule_count": len(plan.get("backbone_rules") or []),
        "neck_rule_count": len(plan.get("neck_rules") or []),
        "initial_prediction_rule_count": len(plan.get("initial_prediction_rules") or []),
        "head_rule_count": len(plan.get("head_rules") or []),
        "config_hints": deepcopy(config_hints if isinstance(config_hints, dict) else {}),
    }


def compact_finetune_training_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    plan = plan or {}
    if not isinstance(plan, dict):
        return {}
    return _copy_selected(
        plan,
        [
            "should_prepare",
            "target_module",
            "trigger_mode",
            "generated_config_path",
            "train_mode",
            "freeze_backbone",
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "reason",
            "supervision_mode",
            "failure_category",
        ],
    )


def compact_runtime_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    plan = plan or {}
    if not isinstance(plan, dict):
        return {}
    return _copy_selected(
        plan,
        [
            "planning_stage",
            "enabled",
            "use_llm",
            "preferred_expert_model",
            "routing_mode",
            "reason",
        ],
    )


def compact_plan_snapshot(plan: dict[str, Any] | None) -> dict[str, Any]:
    plan = plan or {}
    if not isinstance(plan, dict):
        return {}
    return {
        "generated_config_path": plan.get("generated_config_path"),
        "parameter_updates": deepcopy(plan.get("parameter_updates") or {}),
        "llm_result": deepcopy(plan.get("llm_result") or {}),
        "llm_gateway_result": compact_gateway_trace(plan.get("llm_gateway_result") or {}),
        "runtime_plan": compact_runtime_plan(plan.get("runtime_plan") or {}),
        "roi_refine_plan": compact_roi_refine_plan(plan.get("roi_refine_plan") or {}),
        "expert_model_call_plan": compact_expert_model_call_plan(plan.get("expert_model_call_plan") or plan.get("expert_model_call_plan") or {}),
        "knowledge_embedding_plan": compact_knowledge_embedding_plan(plan.get("knowledge_embedding_plan") or {}),
        "finetune_training_plan": compact_finetune_training_plan(plan.get("finetune_training_plan") or {}),
        "scheduler_context": compact_scheduler_context(plan.get("scheduler_context") or {}),
    }


def compact_planning_summary(planning_summary: dict[str, Any] | None) -> dict[str, Any]:
    planning_summary = planning_summary or {}
    if not isinstance(planning_summary, dict):
        return {}
    roi_rounds = planning_summary.get("roi_rounds") or []
    compacted_rounds = []
    for item in roi_rounds[:3]:
        if not isinstance(item, dict):
            continue
        compacted_rounds.append(
            {
                "round_idx": item.get("round_idx"),
                "parameter_updates": deepcopy(item.get("parameter_updates") or {}),
                "runtime_plan": compact_runtime_plan(item.get("runtime_plan") or {}),
                "roi_refine_plan": compact_roi_refine_plan(item.get("roi_refine_plan") or {}),
                "expert_model_call_plan": compact_expert_model_call_plan(item.get("expert_model_call_plan") or item.get("expert_model_call_plan") or {}),
                "accepted": item.get("accepted"),
                "acceptance_reason": item.get("acceptance_reason"),
                "candidate_score": item.get("candidate_score"),
                "best_score_before_round": item.get("best_score_before_round"),
                "selected_score_after_round": item.get("selected_score_after_round"),
                "failure_modes": _limit_list(item.get("failure_modes") or [], limit=6),
            }
        )
    return {
        "main_model_plan": compact_plan_snapshot(planning_summary.get("main_model_plan") or {}),
        "roi_round_count": planning_summary.get("roi_round_count", len(roi_rounds)),
        "roi_rounds_preview": compacted_rounds,
        "refinement_review": deepcopy(planning_summary.get("refinement_review") or {}),
        "finetune_recommendation": _copy_selected(
            planning_summary.get("finetune_recommendation") or {},
            ["should_recommend", "target_module", "trigger_mode", "failure_category", "reason"],
        ),
        "finetune_training_plan": compact_finetune_training_plan(planning_summary.get("finetune_training_plan") or {}),
    }


def compact_segmentation_summary(segmentation_summary: dict[str, Any] | None) -> dict[str, Any]:
    segmentation_summary = segmentation_summary or {}
    if not isinstance(segmentation_summary, dict):
        return {}
    main_model = segmentation_summary.get("main_model") or {}
    execution_request = main_model.get("execution_request") or {}
    execution_result = main_model.get("execution_result") or {}
    runtime_cfg = execution_request.get("runtime_cfg") or {}
    return {
        "main_model": {
            "phase": execution_request.get("phase") or execution_result.get("phase"),
            "model_role": execution_request.get("model_role") or execution_result.get("model_role"),
            "algorithm_name": execution_request.get("algorithm_name") or execution_result.get("algorithm_name"),
            "selected_model_name": runtime_cfg.get("selected_model_name"),
            "segmentation_algorithm": runtime_cfg.get("segmentation_algorithm"),
            "segmentation_script": runtime_cfg.get("segmentation_script"),
            "status": execution_result.get("status"),
            "output_paths": deepcopy(execution_result.get("output_paths") or {}),
        },
        "roi_round_count": segmentation_summary.get("roi_round_count"),
        "y_inst_shp": segmentation_summary.get("y_inst_shp"),
        "tree_crowns_shp": segmentation_summary.get("tree_crowns_shp"),
        "tree_points_shp": segmentation_summary.get("tree_points_shp"),
    }


def compact_evaluation_summary(evaluation_summary: dict[str, Any] | None) -> dict[str, Any]:
    evaluation_summary = evaluation_summary or {}
    if not isinstance(evaluation_summary, dict):
        return {}
    final_evaluation = evaluation_summary.get("final_evaluation") or {}
    metrics = evaluation_summary.get("metrics") or {}
    return {
        "metrics": deepcopy(metrics),
        "score": safe_float(metrics.get("score")),
        "final_evaluation": _copy_selected(
            final_evaluation,
            ["status", "score", "ap50", "ap75", "mean_iou", "f1_score"],
        ),
        "failure_analysis": deepcopy(evaluation_summary.get("failure_analysis") or {}),
    }


def compact_run_retrospective_trace(trace: dict[str, Any] | None) -> dict[str, Any]:
    trace = trace or {}
    if not isinstance(trace, dict):
        return {}
    compacted = compact_gateway_trace(trace)
    compacted["parsed_result"] = deepcopy(trace.get("parsed_result") or {})
    return compacted


def compact_memory_record(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = deepcopy(payload or {})
    if not isinstance(payload, dict):
        return {}

    memory_type = str(payload.get("memory_type") or "")
    if memory_type == "execution_trace":
        payload["planning_summary"] = compact_planning_summary(payload.get("planning_summary") or {})
        payload["segmentation_summary"] = compact_segmentation_summary(payload.get("segmentation_summary") or {})
        payload["evaluation_summary"] = compact_evaluation_summary(payload.get("evaluation_summary") or {})
        return payload
    if memory_type == "successful_strategy":
        payload["strategy_summary"] = compact_planning_summary(payload.get("strategy_summary") or {})
        return payload
    if memory_type == "failure_pattern":
        failure_summary = payload.get("failure_summary") or {}
        if isinstance(failure_summary, dict):
            payload["failure_summary"] = {
                "metrics": deepcopy(failure_summary.get("metrics") or {}),
                "top_problem_cases": _limit_list(failure_summary.get("top_problem_cases") or [], limit=3),
                "refinement_review": deepcopy(failure_summary.get("refinement_review") or {}),
                "refinement_failure_modes": _limit_list(failure_summary.get("refinement_failure_modes") or [], limit=6),
            }
        payload["failure_modes"] = _limit_list(payload.get("failure_modes") or [], limit=8)
        payload["recommended_actions"] = _limit_list(payload.get("recommended_actions") or [], limit=6)
        return payload
    if memory_type == "run_retrospective":
        payload["llm_gateway_result"] = compact_run_retrospective_trace(payload.get("llm_gateway_result") or {})
        payload["parsed_result"] = deepcopy(payload.get("parsed_result") or {})
        return payload
    return payload
