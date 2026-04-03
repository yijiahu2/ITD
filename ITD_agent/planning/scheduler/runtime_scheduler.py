from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ITD_agent.config_adapter import load_raw_yaml, load_runtime_config, save_runtime_config
from ITD_agent.finetune_pool.policy import infer_failure_category
from ITD_agent.planning.contracts import (
    ChildModelCallPlan,
    FinetuneTrainingPlan,
    KnowledgeEmbeddingPlan,
    PlanningDecision,
    ROIRefinePlan,
)
from ITD_agent.planning.scheduler.adaptive_config_generator import generate_adaptive_config_from_template
from ITD_agent.planning.scheduler.template_manager import apply_parameter_updates


def _normalize_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def _get_itd_agent_block(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("ITD_agent") or {}


def _get_planning_block(cfg: dict[str, Any]) -> dict[str, Any]:
    return (_get_itd_agent_block(cfg).get("planning") or {})


def _get_roi_extraction_block(cfg: dict[str, Any]) -> dict[str, Any]:
    planning_cfg = _get_planning_block(cfg)
    roi_cfg = planning_cfg.get("roi_extraction")
    if isinstance(roi_cfg, dict):
        return roi_cfg
    roi_cfg = planning_cfg.get("roi_refine")
    if isinstance(roi_cfg, dict):
        return roi_cfg
    return {}


def _get_adaptive_generation_block(cfg: dict[str, Any]) -> dict[str, Any]:
    return (_get_planning_block(cfg).get("adaptive_generation") or {})


def _get_pipeline_block(cfg: dict[str, Any]) -> dict[str, Any]:
    pipeline_cfg = cfg.get("pipeline")
    return pipeline_cfg if isinstance(pipeline_cfg, dict) else {}


def _get_segmentation_models_block(cfg: dict[str, Any]) -> dict[str, Any]:
    return (_get_itd_agent_block(cfg).get("segmentation_models") or {})


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _normalize_tag(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _expand_tags(values: Any) -> set[str]:
    tags: set[str] = set()
    if values is None:
        return tags
    raw_values = values if isinstance(values, list) else _as_str_list(values)
    for item in raw_values:
        normalized = _normalize_tag(item)
        if not normalized:
            continue
        tags.add(normalized)
        parts = [part for part in re.split(r"[_/|]+", normalized) if part]
        tags.update(parts)
    return tags


def _get_model_entry_name(entry: dict[str, Any]) -> str | None:
    for key in ["name", "algorithm", "id", "script"]:
        value = entry.get(key)
        if value:
            return str(value)
    return None


def _extract_child_model_entries(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    seg_models = _get_segmentation_models_block(cfg)
    entries: list[dict[str, Any]] = []
    for key in ["child_models", "expert_models", "sub_models"]:
        block = seg_models.get(key)
        if isinstance(block, list):
            entries.extend(item for item in block if isinstance(item, dict))
        elif isinstance(block, dict):
            entries.extend(item for item in block.values() if isinstance(item, dict))
    return entries


def _get_candidate_child_models(cfg: dict[str, Any], scheduler_context: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for item in _extract_child_model_entries(cfg):
        name = _get_model_entry_name(item)
        if name:
            candidates.append(str(name))
    if not candidates:
        current_algorithm = cfg.get("segmentation_algorithm") or scheduler_context.get("current_parameters", {}).get("segmentation_algorithm")
        if current_algorithm:
            candidates.append(str(current_algorithm))
    deduped: list[str] = []
    for item in candidates:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _collect_scene_tags(scene_profile: dict[str, Any]) -> set[str]:
    tags = set()
    tags.update(_expand_tags(scene_profile.get("forest_type")))
    tags.update(_expand_tags(scene_profile.get("terrain_type")))
    tags.update(_expand_tags(scene_profile.get("knowledge_profile_types")))
    tags.update(_expand_tags(scene_profile.get("public_dataset_roles")))
    tags.update(_expand_tags(scene_profile.get("tags")))
    tags.update(_expand_tags(scene_profile.get("stand_condition_labels")))
    tags.update(_expand_tags(scene_profile.get("texture_labels")))
    return tags


def _collect_terrain_tags(top_cases: list[dict[str, Any]], scene_profile: dict[str, Any]) -> set[str]:
    tags = set()
    tags.update(_expand_tags(scene_profile.get("terrain_type")))
    for case in top_cases[:5]:
        for key in ["landform_type", "slope_class", "aspect_class", "slope_position_class"]:
            tags.update(_expand_tags(case.get(key)))
        mean_slope = _safe_float(case.get("mean_slope"))
        if mean_slope is not None:
            if mean_slope >= 25:
                tags.add("steep")
            elif mean_slope >= 12:
                tags.add("moderate")
            else:
                tags.add("gentle")
        aspect = _normalize_tag(case.get("aspect_class"))
        if aspect in {"north", "northeast", "northwest"}:
            tags.add("shadow")
            tags.add("north_shade")
    return tags


def _collect_error_patterns(top_cases: list[dict[str, Any]], metrics: dict[str, Any]) -> set[str]:
    patterns: set[str] = set()
    for case in top_cases[:5]:
        pred_tree = _safe_float(case.get("pred_tree_count"))
        expected_tree = _safe_float(case.get("expected_tree_count"))
        if pred_tree is not None and expected_tree is not None:
            patterns.add("count_under" if pred_tree < expected_tree else "count_over")

        pred_cover = _safe_float(case.get("pred_cover_ratio"))
        expected_cover = _safe_float(case.get("expected_closure"))
        if pred_cover is not None and expected_cover is not None:
            patterns.add("closure_low" if pred_cover < expected_cover else "closure_high")

        pred_density = _safe_float(case.get("pred_density_trees_per_ha"))
        expected_density = _safe_float(case.get("expected_density"))
        if pred_density is not None and expected_density is not None:
            patterns.add("density_low" if pred_density < expected_density else "density_high")

        if (_safe_float(case.get("mean_crown_width_error_abs")) or 0.0) > 0:
            patterns.add("crown")

    if float(metrics.get("tree_count_error_ratio") or 0.0) >= 0.18:
        patterns.add("count")
    if float(metrics.get("mean_crown_width_error_ratio") or 0.0) >= 0.22:
        patterns.add("crown")
    if float(metrics.get("closure_error_abs") or 0.0) >= 0.10:
        patterns.add("closure")
    return patterns


def _build_child_model_routing_context(scheduler_context: dict[str, Any]) -> dict[str, Any]:
    scene_profile = scheduler_context.get("scene_profile") or {}
    roi_assessment = scheduler_context.get("roi_assessment") or {}
    details_summary = (roi_assessment.get("details_summary") or scheduler_context.get("details_summary") or {})
    top_cases = details_summary.get("top_k_xiaoban") or []
    metrics = scheduler_context.get("evaluation_metrics") or {}
    failure_categories = [
        infer_failure_category(case)
        for case in top_cases[:5]
        if isinstance(case, dict)
    ]
    return {
        "scene_tags": sorted(_collect_scene_tags(scene_profile)),
        "terrain_tags": sorted(_collect_terrain_tags(top_cases, scene_profile)),
        "failure_categories": sorted(set(failure_categories)),
        "target_error_patterns": sorted(_collect_error_patterns(top_cases, metrics)),
        "top_problem_cases": top_cases[:5],
    }


def _normalize_child_model_profile(entry: dict[str, Any], routing_context: dict[str, Any]) -> dict[str, Any]:
    name = _get_model_entry_name(entry)
    scene_tags = _expand_tags(entry.get("scene_tags") or entry.get("scene_labels"))
    terrain_tags = _expand_tags(entry.get("terrain_tags"))
    failure_categories = _expand_tags(entry.get("failure_categories"))
    target_error_patterns = _expand_tags(entry.get("target_error_patterns"))
    selection_hints = _as_str_list(entry.get("selection_hints"))
    template_profile = bool(entry.get("template_profile"))
    if not template_profile:
        template_profile = not any(entry.get(key) for key in ["algorithm", "algorithm_module", "checkpoint", "config_file"])

    score = float(entry.get("routing_priority") or 0)
    reason_parts: list[str] = []

    matched_failures = sorted(failure_categories & set(routing_context.get("failure_categories") or []))
    if matched_failures:
        score += 80 + 8 * len(matched_failures)
        reason_parts.append(f"匹配失败类别: {', '.join(matched_failures)}")

    matched_errors = sorted(target_error_patterns & set(routing_context.get("target_error_patterns") or []))
    if matched_errors:
        score += 40 + 5 * len(matched_errors)
        reason_parts.append(f"匹配误差模式: {', '.join(matched_errors)}")

    matched_terrain = sorted(terrain_tags & set(routing_context.get("terrain_tags") or []))
    if matched_terrain:
        score += 24 + 3 * len(matched_terrain)
        reason_parts.append(f"匹配地形标签: {', '.join(matched_terrain)}")

    matched_scene = sorted(scene_tags & set(routing_context.get("scene_tags") or []))
    if matched_scene:
        score += 16 + 2 * len(matched_scene)
        reason_parts.append(f"匹配场景标签: {', '.join(matched_scene)}")

    if template_profile:
        score += 1
    if selection_hints:
        reason_parts.append(f"模板说明: {'; '.join(selection_hints[:2])}")
    if not reason_parts:
        reason_parts.append("未命中显式规则，按模板优先级回退。")

    return {
        "name": name,
        "description": str(entry.get("description") or ""),
        "template_profile": template_profile,
        "scene_tags": sorted(scene_tags),
        "terrain_tags": sorted(terrain_tags),
        "failure_categories": sorted(failure_categories),
        "target_error_patterns": sorted(target_error_patterns),
        "selection_hints": selection_hints,
        "routing_priority": int(entry.get("routing_priority") or 0),
        "algorithm": entry.get("algorithm"),
        "script": entry.get("script"),
        "checkpoint": entry.get("checkpoint"),
        "runtime_overrides": dict(entry.get("runtime_overrides") or {}),
        "score": score,
        "selection_reason": "；".join(reason_parts),
    }


def _rank_child_model_profiles(runtime_cfg: dict[str, Any], scheduler_context: dict[str, Any]) -> list[dict[str, Any]]:
    routing_context = _build_child_model_routing_context(scheduler_context)
    profiles = [
        _normalize_child_model_profile(entry, routing_context)
        for entry in _extract_child_model_entries(runtime_cfg)
        if _get_model_entry_name(entry)
    ]
    profiles.sort(key=lambda item: (item["score"], item["routing_priority"]), reverse=True)
    return profiles


def _resolve_preferred_child_model(
    *,
    runtime_cfg: dict[str, Any],
    scheduler_context: dict[str, Any],
    llm_result: dict[str, Any] | None,
) -> tuple[str | None, list[dict[str, Any]], str]:
    profiles = _rank_child_model_profiles(runtime_cfg, scheduler_context)
    candidate_names = [str(item["name"]) for item in profiles if item.get("name")]
    llm_preferred = str((((llm_result or {}).get("child_model_call_plan") or {}).get("preferred_child_model") or "")).strip()
    roi_preferred = str((((scheduler_context.get("roi_assessment") or {}).get("decision") or {}).get("preferred_child_model") or "")).strip()

    if llm_preferred and llm_preferred in candidate_names:
        return llm_preferred, profiles, f"LLM 指定子模型模板: {llm_preferred}"
    if roi_preferred and roi_preferred in candidate_names:
        return roi_preferred, profiles, f"ROI 决策指定子模型模板: {roi_preferred}"
    if profiles:
        return str(profiles[0]["name"]), profiles, str(profiles[0]["selection_reason"])

    fallback_name = None
    current_algorithm = runtime_cfg.get("segmentation_algorithm") or scheduler_context.get("current_parameters", {}).get("segmentation_algorithm")
    if current_algorithm:
        fallback_name = str(current_algorithm)
    return fallback_name, [], "未配置独立子模型模板，回退为当前分割引擎的 ROI 局部重跑。"


def _get_knowledge_profiles(scheduler_context: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = scheduler_context.get("knowledge_profiles")
    return profiles if isinstance(profiles, list) else []


def _build_roi_refine_plan(
    *,
    runtime_cfg: dict[str, Any],
    scheduler_context: dict[str, Any],
    llm_result: dict[str, Any] | None,
    planning_stage: str,
) -> dict[str, Any]:
    roi_cfg = _get_roi_extraction_block(runtime_cfg)
    llm_plan = (llm_result or {}).get("roi_refine_plan") or {}
    runtime_max_rounds = int(roi_cfg.get("max_rounds", 2))
    planned_max_rounds = int(llm_plan.get("max_rounds", runtime_max_rounds))
    effective_max_rounds = min(planned_max_rounds, runtime_max_rounds)
    preferred_child_model, ranked_profiles, selection_reason = _resolve_preferred_child_model(
        runtime_cfg=runtime_cfg,
        scheduler_context=scheduler_context,
        llm_result=llm_result,
    )
    candidates = [str(item["name"]) for item in ranked_profiles if item.get("name")] or _get_candidate_child_models(runtime_cfg, scheduler_context)
    preferred_child_model = llm_plan.get("preferred_child_model") or preferred_child_model
    selection_rules = _as_str_list(llm_plan.get("selection_rules")) or [
        "优先选择综合质量分数最低且多轮未收敛的 ROI。",
        "优先细化纹理复杂、地形起伏大、实例重叠冲突明显的 ROI。",
        "若存在独立子模型模板，优先选择与失败类别、地形标签和误差模式匹配度最高的模板。",
    ]
    stop_rules = _as_str_list(llm_plan.get("stop_rules")) or [
        "达到 ROI 质量阈值后停止。",
        "连续两轮提升不足时停止。",
        "超过最大 ROI 轮次后停止。",
    ]
    return ROIRefinePlan(
        enabled=_normalize_bool(roi_cfg.get("enabled", planning_stage == "child_model")),
        use_llm=_normalize_bool(roi_cfg.get("use_llm", True)),
        max_rounds=effective_max_rounds,
        top_k=int(llm_plan.get("top_k", roi_cfg.get("top_k", 3))),
        buffer_m=float(llm_plan.get("buffer_m", roi_cfg.get("buffer_m", 5.0))),
        strategy_mode=str(llm_plan.get("strategy_mode", roi_cfg.get("strategy_mode", "auto"))),
        preferred_child_model=str(preferred_child_model) if preferred_child_model else None,
        candidate_child_models=candidates,
        selection_rules=selection_rules + ([f"默认模板选择依据: {selection_reason}"] if selection_reason else []),
        stop_rules=stop_rules,
    ).to_dict()


def _build_child_model_call_plan(
    *,
    runtime_cfg: dict[str, Any],
    scheduler_context: dict[str, Any],
    llm_result: dict[str, Any] | None,
    planning_stage: str,
) -> dict[str, Any]:
    llm_plan = (llm_result or {}).get("child_model_call_plan") or {}
    preferred_child_model, ranked_profiles, selection_reason = _resolve_preferred_child_model(
        runtime_cfg=runtime_cfg,
        scheduler_context=scheduler_context,
        llm_result=llm_result,
    )
    candidates = [str(item["name"]) for item in ranked_profiles if item.get("name")] or _get_candidate_child_models(runtime_cfg, scheduler_context)
    preferred_child_model = llm_plan.get("preferred_child_model") or preferred_child_model
    routing_rules = _as_str_list(llm_plan.get("routing_rules")) or [
        "地形复杂和冠幅误差偏大的 ROI 优先路由到更强的子模型。",
        "实例粘连和边界碎裂明显的 ROI 优先启用边界更稳健的子模型。",
        "若未配置独立 checkpoint，则允许使用子模型模板复用主分割引擎并套用模板化运行参数。",
    ]
    escalation_rules = _as_str_list(llm_plan.get("escalation_rules")) or [
        "首选子模型连续一轮无提升时切换到候选列表中的下一个模型。",
        "无可用子模型或 ROI 信息不足时返回主模型结果并停止细化。",
    ]
    routing_mode = str(llm_plan.get("routing_mode") or ("template_profile_routing" if ranked_profiles else "roi_quality_driven"))
    plan = ChildModelCallPlan(
        enabled=planning_stage == "child_model",
        planning_stage=planning_stage,
        routing_mode=routing_mode,
        preferred_child_model=str(preferred_child_model) if preferred_child_model else None,
        candidate_models=_as_str_list(llm_plan.get("candidate_models")) or candidates,
        routing_rules=routing_rules,
        escalation_rules=escalation_rules,
    ).to_dict()
    plan["candidate_profiles"] = ranked_profiles
    plan["selection_reason"] = selection_reason
    return plan


def _default_knowledge_rule(component: str, description: str, parameter_hint: str, condition: str = "存在相关先验知识") -> dict[str, Any]:
    return {
        "component": component,
        "rule": description,
        "condition": condition,
        "parameter_hint": parameter_hint,
    }


def _build_knowledge_embedding_plan(
    *,
    runtime_cfg: dict[str, Any],
    scheduler_context: dict[str, Any],
    llm_result: dict[str, Any] | None,
) -> dict[str, Any]:
    llm_plan = (llm_result or {}).get("knowledge_embedding_plan") or {}
    knowledge_profiles = _get_knowledge_profiles(scheduler_context)
    sources = [str(item.get("source_id")) for item in knowledge_profiles if item.get("source_id")]
    if not knowledge_profiles:
        return KnowledgeEmbeddingPlan(enabled=False, knowledge_sources=[]).to_dict()

    raster_count = sum(1 for item in knowledge_profiles if item.get("normalized_type") == "raster_prior")
    rule_count = sum(1 for item in knowledge_profiles if item.get("normalized_type") in {"rule_knowledge", "strategy_knowledge"})
    backbone_rules = llm_plan.get("backbone_rules") or []
    neck_rules = llm_plan.get("neck_rules") or []
    initial_prediction_rules = llm_plan.get("initial_prediction_rules") or []
    head_rules = llm_plan.get("head_rules") or []
    if not backbone_rules:
        backbone_rules = [
            _default_knowledge_rule("backbone", "在子模型主干特征提取阶段引入场景先验标签或区域嵌入，用于调节特征提取侧重点。", "backbone.prior_conditioning=scene_embedding")
        ]
    if not neck_rules:
        neck_rules = [
            _default_knowledge_rule("neck", "在特征金字塔融合阶段叠加地形/林分类先验的多尺度引导特征。", "neck.prior_fusion=multi_scale_gating")
        ]
    if not initial_prediction_rules:
        initial_prediction_rules = [
            _default_knowledge_rule("initial_prediction", "用先验树高/冠幅或林型分布约束初始候选框/查询的尺度与密度。", "initial_prediction.prior_seed=crown_scale_query_bias")
        ]
    if not head_rules:
        head_rules = [
            _default_knowledge_rule("head", "在头部网络使用先验驱动的损失权重或后处理阈值，抑制不合理实例。", "head.prior_aware_loss=enabled")
        ]
    config_hints = llm_plan.get("config_hints") or {
        "knowledge_embedding": {
            "enabled": True,
            "raster_prior_count": raster_count,
            "rule_knowledge_count": rule_count,
            "apply_to": ["backbone", "neck", "initial_prediction", "head"],
        }
    }
    return KnowledgeEmbeddingPlan(
        enabled=True,
        knowledge_sources=sources,
        backbone_rules=backbone_rules,
        neck_rules=neck_rules,
        initial_prediction_rules=initial_prediction_rules,
        head_rules=head_rules,
        config_hints=config_hints,
    ).to_dict()


def _build_finetune_config_overrides(
    *,
    target_module: str,
    train_mode: str,
    freeze_backbone: bool,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    weight_decay: float,
    generated_output_dir: str,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "output_dir": generated_output_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "lr": lr,
        "weight_decay": weight_decay,
        "train_mode": train_mode,
        "freeze_backbone": freeze_backbone,
        "target_module": target_module,
    }
    if target_module == "segmentation_model":
        overrides.update(
            {
                "segmentation_train_epochs": epochs,
                "segmentation_train_batch_size": batch_size,
                "segmentation_train_num_workers": num_workers,
                "segmentation_train_lr": lr,
                "segmentation_train_weight_decay": weight_decay,
            }
        )
    return overrides


def build_finetune_training_plan(
    *,
    runtime_cfg: dict[str, Any],
    scheduler_context: dict[str, Any],
    llm_result: dict[str, Any] | None = None,
    finetune_recommendation: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    llm_plan = (llm_result or {}).get("finetune_training_plan") or {}
    recommendation = finetune_recommendation or {}
    metrics = scheduler_context.get("evaluation_metrics") or {}
    pipeline_cfg = _get_pipeline_block(runtime_cfg)
    template_path = pipeline_cfg.get("finetune_config")
    should_prepare = _normalize_bool(
        llm_plan.get(
            "should_prepare",
            recommendation.get("should_recommend", False),
        )
    )
    target_module = str(llm_plan.get("target_module") or recommendation.get("target_module") or "segmentation_model")
    max_error = max(
        float(metrics.get("tree_count_error_ratio") or 0.0),
        float(metrics.get("mean_crown_width_error_ratio") or 0.0),
        float(metrics.get("closure_error_abs") or 0.0),
    )
    train_mode = str(llm_plan.get("train_mode") or ("head_only" if target_module == "segmentation_model" else "full"))
    freeze_backbone = _normalize_bool(llm_plan.get("freeze_backbone", train_mode == "head_only"))
    epochs = int(llm_plan.get("epochs", 8 if max_error >= 0.25 else 4))
    batch_size = int(llm_plan.get("batch_size", 1))
    num_workers = int(llm_plan.get("num_workers", 4))
    lr = float(llm_plan.get("lr", 1.0e-4 if freeze_backbone else 5.0e-5))
    weight_decay = float(llm_plan.get("weight_decay", 1.0e-4))
    generated_output_dir = str(Path(runtime_cfg.get("output_dir") or ".").resolve().parent / "finetune" / str(runtime_cfg.get("run_name") or "itd_agent"))
    config_overrides = _build_finetune_config_overrides(
        target_module=target_module,
        train_mode=train_mode,
        freeze_backbone=freeze_backbone,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        lr=lr,
        weight_decay=weight_decay,
        generated_output_dir=generated_output_dir,
    )
    if isinstance(llm_plan.get("config_overrides"), dict):
        config_overrides.update(llm_plan["config_overrides"])

    generated_config_path: str | None = None
    if output_path is not None and template_path and Path(str(template_path)).exists():
        template_cfg = load_raw_yaml(template_path)
        generated_cfg = apply_parameter_updates(template_cfg, config_overrides)
        save_runtime_config(generated_cfg, output_path)
        generated_config_path = str(Path(output_path))

    return FinetuneTrainingPlan(
        should_prepare=should_prepare,
        target_module=target_module,
        trigger_mode=str(llm_plan.get("trigger_mode") or recommendation.get("trigger_mode") or "defer_until_pool_threshold"),
        template_config_path=str(template_path) if template_path else None,
        generated_config_path=generated_config_path,
        train_mode=train_mode,
        freeze_backbone=freeze_backbone,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        lr=lr,
        weight_decay=weight_decay,
        data_selection_rule=str(
            llm_plan.get("data_selection_rule")
            or recommendation.get("reason")
            or "按微调池中的同类失败样本聚类结果选择训练集。"
        ),
        supervision_mode=str(llm_plan.get("supervision_mode") or "hybrid"),
        dataset_bundle_path=None,
        dataset_selection_summary={},
        config_overrides=config_overrides,
        reason=str(llm_plan.get("reason") or recommendation.get("reason") or ""),
    ).to_dict()


def _build_runtime_plan(
    *,
    planning_stage: str,
    effective_runtime_cfg: dict[str, Any],
    parameter_updates: dict[str, Any],
) -> dict[str, Any]:
    return {
        "planning_stage": planning_stage,
        "segmentation_algorithm": effective_runtime_cfg.get("segmentation_algorithm"),
        "segmentation_parameters": extract_segmentation_params(effective_runtime_cfg),
        "parameter_updates": parameter_updates or {},
    }


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plan_runtime_config(
    *,
    template_path: str,
    output_path: str | Path,
    runtime_cfg: dict[str, Any],
    metrics_json: str | None = None,
    details_csv: str | None = None,
    summary_json: str | None = None,
) -> dict[str, Any]:
    adaptive_cfg = _get_adaptive_generation_block(runtime_cfg)
    planning_enabled = _normalize_bool(adaptive_cfg.get("enabled", False))
    use_llm = _normalize_bool(adaptive_cfg.get("use_llm", True))
    out_path = Path(output_path)
    ensure_parent(out_path)
    planning_stage = str(runtime_cfg.get("_planning_stage") or "main_model")

    if not planning_enabled or not Path(template_path).exists():
        save_runtime_config(runtime_cfg, out_path)
        planned_cfg, _ = load_runtime_config(out_path)
        scheduler_context: dict[str, Any] = {}
        runtime_plan = _build_runtime_plan(
            planning_stage=planning_stage,
            effective_runtime_cfg=planned_cfg,
            parameter_updates={},
        )
        roi_refine_plan = _build_roi_refine_plan(
            runtime_cfg=planned_cfg,
            scheduler_context=scheduler_context,
            llm_result=None,
            planning_stage=planning_stage,
        )
        child_model_call_plan = _build_child_model_call_plan(
            runtime_cfg=planned_cfg,
            scheduler_context=scheduler_context,
            llm_result=None,
            planning_stage=planning_stage,
        )
        knowledge_embedding_plan = _build_knowledge_embedding_plan(
            runtime_cfg=planned_cfg,
            scheduler_context=scheduler_context,
            llm_result=None,
        )
        return PlanningDecision(
            enabled=planning_enabled,
            use_llm=use_llm,
            planning_stage=planning_stage,
            template_path=str(template_path),
            generated_config_path=str(out_path),
            effective_runtime_cfg=planned_cfg,
            runtime_plan=runtime_plan,
            roi_refine_plan=roi_refine_plan,
            child_model_call_plan=child_model_call_plan,
            knowledge_embedding_plan=knowledge_embedding_plan,
        ).to_dict()

    planning_result = generate_adaptive_config_from_template(
        template_path=template_path,
        output_path=out_path,
        runtime_cfg=runtime_cfg,
        metrics_json=metrics_json,
        details_csv=details_csv,
        summary_json=summary_json,
        use_llm=use_llm,
    )
    planned_cfg, _ = load_runtime_config(out_path)
    scheduler_context = planning_result.get("scheduler_context") or {}
    parameter_updates = planning_result.get("parameter_updates") or {}
    llm_result = planning_result.get("llm_result")
    runtime_plan = _build_runtime_plan(
        planning_stage=planning_stage,
        effective_runtime_cfg=planned_cfg,
        parameter_updates=parameter_updates,
    )
    roi_refine_plan = _build_roi_refine_plan(
        runtime_cfg=planned_cfg,
        scheduler_context=scheduler_context,
        llm_result=llm_result,
        planning_stage=planning_stage,
    )
    child_model_call_plan = _build_child_model_call_plan(
        runtime_cfg=planned_cfg,
        scheduler_context=scheduler_context,
        llm_result=llm_result,
        planning_stage=planning_stage,
    )
    knowledge_embedding_plan = _build_knowledge_embedding_plan(
        runtime_cfg=planned_cfg,
        scheduler_context=scheduler_context,
        llm_result=llm_result,
    )
    return PlanningDecision(
        enabled=planning_enabled,
        use_llm=use_llm,
        planning_stage=planning_stage,
        template_path=str(template_path),
        generated_config_path=str(out_path),
        parameter_updates=parameter_updates,
        llm_result=llm_result,
        llm_gateway_result=planning_result.get("llm_gateway_result"),
        scheduler_context=scheduler_context,
        effective_runtime_cfg=planned_cfg,
        runtime_plan=runtime_plan,
        roi_refine_plan=roi_refine_plan,
        child_model_call_plan=child_model_call_plan,
        knowledge_embedding_plan=knowledge_embedding_plan,
    ).to_dict()


def extract_segmentation_params(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        key: cfg.get(key)
        for key in ["diam_list", "tile", "overlap", "tile_overlap", "augment", "iou_merge_thr", "bsize"]
        if key in cfg
    }


def build_main_model_planning_runtime_cfg(
    *,
    cfg: dict[str, Any],
    input_assessment: dict[str, Any],
    input_manifest: dict[str, Any],
    data_processing_summary: dict[str, Any],
) -> dict[str, Any]:
    plan_cfg = dict(cfg)
    plan_cfg["_planning_stage"] = "main_model"
    plan_cfg["_input_assessment"] = input_assessment
    plan_cfg["_input_manifest"] = input_manifest
    plan_cfg["_data_processing_summary"] = data_processing_summary
    return plan_cfg


def build_child_model_planning_runtime_cfg(
    *,
    cfg: dict[str, Any],
    input_assessment: dict[str, Any],
    input_manifest: dict[str, Any],
    data_processing_summary: dict[str, Any],
    roi_assessment: dict[str, Any],
    previous_round_summary: dict[str, Any],
) -> dict[str, Any]:
    plan_cfg = dict(cfg)
    plan_cfg["_planning_stage"] = "child_model"
    plan_cfg["_input_assessment"] = input_assessment
    plan_cfg["_input_manifest"] = input_manifest
    plan_cfg["_data_processing_summary"] = data_processing_summary
    plan_cfg["_roi_assessment"] = roi_assessment
    plan_cfg["_previous_round_summary"] = previous_round_summary
    return plan_cfg
