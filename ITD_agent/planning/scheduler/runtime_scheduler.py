from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ITD_agent.config_adapter import load_raw_yaml, load_runtime_config, save_runtime_config
from ITD_agent.finetune_pool.policy import infer_failure_category
from ITD_agent.model_roles import EXPERT_MODEL_ROLE, MAIN_MODEL_ROLE, normalize_model_role
from ITD_agent.planning.contracts import (
    ExpertModelCallPlan,
    FinetuneTrainingPlan,
    KnowledgeEmbeddingPlan,
    PlanningDecision,
    ROIRefinePlan,
)
from ITD_agent.planning.scheduler.adaptive_config_generator import generate_adaptive_config_from_template
from ITD_agent.planning.scheduler.expert_taxonomy import (
    build_expert_training_defaults,
    infer_expert_family_from_entry,
    load_expert_taxonomy,
    resolve_expert_template_path,
)
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


def _get_child_model_routing_block(cfg: dict[str, Any]) -> dict[str, Any]:
    planning_cfg = _get_planning_block(cfg)
    return (planning_cfg.get("expert_model_routing") or planning_cfg.get("child_model_routing") or {})


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


def _normalize_weight_map(raw: dict[str, Any], defaults: dict[str, float]) -> dict[str, float]:
    weights = {}
    for key, default_value in defaults.items():
        value = _safe_float(raw.get(key) if isinstance(raw, dict) else None)
        if value is None:
            value = float(default_value)
        weights[key] = max(float(value), 0.0)
    total = sum(weights.values())
    if total <= 0:
        return dict(defaults)
    return {key: value / total for key, value in weights.items()}


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
    for key in ["expert_models", "child_models", "sub_models"]:
        block = seg_models.get(key)
        if isinstance(block, list):
            entries.extend(item for item in block if isinstance(item, dict))
        elif isinstance(block, dict):
            entries.extend(item for item in block.values() if isinstance(item, dict))
    return entries


def _get_candidate_expert_models(cfg: dict[str, Any], scheduler_context: dict[str, Any]) -> list[str]:
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
    tags.update(_expand_tags(scene_profile.get("terrain_labels")))
    return tags


def _collect_forest_tags(scene_profile: dict[str, Any]) -> set[str]:
    tags = set()
    tags.update(_expand_tags(scene_profile.get("forest_type")))
    tags.update(_expand_tags(scene_profile.get("stand_condition_labels")))
    tags.update(_expand_tags(scene_profile.get("knowledge_profile_types")))
    return tags


def _collect_terrain_tags(
    top_cases: list[dict[str, Any]],
    scene_profile: dict[str, Any],
    terrain_analysis: dict[str, Any] | None = None,
) -> set[str]:
    tags = set()
    tags.update(_expand_tags(scene_profile.get("terrain_type")))
    terrain = terrain_analysis or {}
    tags.update(_expand_tags(terrain.get("labels")))
    tags.update(_expand_tags((terrain.get("dom_context") or {}).get("landform_type")))
    tags.update(_expand_tags((terrain.get("dom_context") or {}).get("slope_class")))
    tags.update(_expand_tags((terrain.get("dom_context") or {}).get("aspect_class")))
    tags.update(_expand_tags((terrain.get("dom_context") or {}).get("slope_position_class")))
    tags.update(_expand_tags((terrain.get("global_background") or {}).get("landform_type")))
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


def _collect_roi_signal_tags(roi_assessment: dict[str, Any]) -> set[str]:
    tags = set()
    for item in (roi_assessment.get("candidate_rois") or [])[:8]:
        if not isinstance(item, dict):
            continue
        tags.update(_expand_tags(item.get("signal_tags")))
        tags.update(_expand_tags(item.get("roi_signal_type")))
        tags.update(_expand_tags(item.get("prior_structure_tag")))
    return tags


def _get_routing_weight_spec(runtime_cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    block = _get_child_model_routing_block(runtime_cfg)
    family_defaults = {
        "failure_categories": 0.34,
        "error_patterns": 0.24,
        "terrain_tags": 0.18,
        "forest_tags": 0.12,
        "scene_tags": 0.06,
        "roi_signal_tags": 0.06,
    }
    model_defaults = {
        "failure_categories": 0.40,
        "error_patterns": 0.25,
        "terrain_tags": 0.15,
        "scene_tags": 0.10,
        "roi_signal_tags": 0.10,
    }
    return {
        "family": _normalize_weight_map(block.get("family_weights") or {}, family_defaults),
        "model": _normalize_weight_map(block.get("model_weights") or {}, model_defaults),
        "score_scale": {
            "family": float(_safe_float(block.get("family_score_scale")) or 100.0),
            "model": float(_safe_float(block.get("model_score_scale")) or 100.0),
        },
    }


def _match_ratio(expected_tags: set[str], observed_tags: set[str]) -> tuple[float, list[str]]:
    if not expected_tags or not observed_tags:
        return 0.0, []
    matched = sorted(expected_tags & observed_tags)
    if not matched:
        return 0.0, []
    return float(len(matched)) / max(float(len(expected_tags)), 1.0), matched


def _build_child_model_routing_context(scheduler_context: dict[str, Any]) -> dict[str, Any]:
    scene_profile = scheduler_context.get("scene_profile") or {}
    terrain_analysis = scheduler_context.get("terrain_analysis") or {}
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
        "forest_tags": sorted(_collect_forest_tags(scene_profile)),
        "scene_tags": sorted(_collect_scene_tags(scene_profile) | _collect_roi_signal_tags(roi_assessment)),
        "terrain_tags": sorted(_collect_terrain_tags(top_cases, scene_profile, terrain_analysis)),
        "failure_categories": sorted(set(failure_categories)),
        "target_error_patterns": sorted(_collect_error_patterns(top_cases, metrics)),
        "roi_signal_tags": sorted(_collect_roi_signal_tags(roi_assessment)),
        "terrain_analysis": terrain_analysis,
        "top_problem_cases": top_cases[:5],
    }


def _normalize_child_model_profile(runtime_cfg: dict[str, Any], entry: dict[str, Any], routing_context: dict[str, Any]) -> dict[str, Any]:
    name = _get_model_entry_name(entry)
    expert_family = infer_expert_family_from_entry(entry)
    scene_tags = _expand_tags(entry.get("scene_tags") or entry.get("scene_labels"))
    terrain_tags = _expand_tags(entry.get("terrain_tags"))
    failure_categories = _expand_tags(entry.get("failure_categories"))
    target_error_patterns = _expand_tags(entry.get("target_error_patterns"))
    selection_hints = _as_str_list(entry.get("selection_hints"))
    template_profile = bool(entry.get("template_profile"))
    if not template_profile:
        template_profile = not any(entry.get(key) for key in ["algorithm", "algorithm_module", "checkpoint", "config_file"])

    weight_spec = _get_routing_weight_spec(runtime_cfg)
    score_weights = weight_spec["model"]
    score_scale = float(weight_spec["score_scale"]["model"])
    score = float(entry.get("routing_priority") or 0)
    reason_parts: list[str] = []
    score_breakdown: dict[str, float] = {"routing_priority": score}

    matched_failures_ratio, matched_failures = _match_ratio(failure_categories, set(routing_context.get("failure_categories") or []))
    if matched_failures:
        component = score_scale * score_weights["failure_categories"] * matched_failures_ratio
        score += component
        score_breakdown["failure_categories"] = component
        reason_parts.append(f"匹配失败类别: {', '.join(matched_failures)}")

    matched_errors_ratio, matched_errors = _match_ratio(target_error_patterns, set(routing_context.get("target_error_patterns") or []))
    if matched_errors:
        component = score_scale * score_weights["error_patterns"] * matched_errors_ratio
        score += component
        score_breakdown["error_patterns"] = component
        reason_parts.append(f"匹配误差模式: {', '.join(matched_errors)}")

    matched_terrain_ratio, matched_terrain = _match_ratio(terrain_tags, set(routing_context.get("terrain_tags") or []))
    if matched_terrain:
        component = score_scale * score_weights["terrain_tags"] * matched_terrain_ratio
        score += component
        score_breakdown["terrain_tags"] = component
        reason_parts.append(f"匹配地形标签: {', '.join(matched_terrain)}")

    matched_scene_ratio, matched_scene = _match_ratio(scene_tags, set(routing_context.get("scene_tags") or []))
    if matched_scene:
        component = score_scale * score_weights["scene_tags"] * matched_scene_ratio
        score += component
        score_breakdown["scene_tags"] = component
        reason_parts.append(f"匹配场景标签: {', '.join(matched_scene)}")

    roi_signal_tags = _expand_tags(entry.get("roi_signal_tags") or entry.get("scene_tags") or entry.get("scene_labels"))
    matched_roi_signal_ratio, matched_roi_signal = _match_ratio(roi_signal_tags, set(routing_context.get("roi_signal_tags") or []))
    if matched_roi_signal:
        component = score_scale * score_weights["roi_signal_tags"] * matched_roi_signal_ratio
        score += component
        score_breakdown["roi_signal_tags"] = component
        reason_parts.append(f"匹配 ROI 信号标签: {', '.join(matched_roi_signal)}")

    if template_profile:
        score += 1
        score_breakdown["template_profile"] = 1.0
    if selection_hints:
        reason_parts.append(f"模板说明: {'; '.join(selection_hints[:2])}")
    if expert_family:
        reason_parts.append(f"专家家族: {expert_family}")
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
        "expert_family": expert_family,
        "algorithm": entry.get("algorithm"),
        "script": entry.get("script"),
        "checkpoint": entry.get("checkpoint"),
        "runtime_overrides": dict(entry.get("runtime_overrides") or {}),
        "score": score,
        "score_breakdown": score_breakdown,
        "selection_reason": "；".join(reason_parts),
    }


def _rank_child_model_profiles(runtime_cfg: dict[str, Any], scheduler_context: dict[str, Any]) -> list[dict[str, Any]]:
    routing_context = _build_child_model_routing_context(scheduler_context)
    profiles = [
        _normalize_child_model_profile(runtime_cfg, entry, routing_context)
        for entry in _extract_child_model_entries(runtime_cfg)
        if _get_model_entry_name(entry)
    ]
    profiles.sort(key=lambda item: (item["score"], item["routing_priority"]), reverse=True)
    return profiles


def _algorithms_priority_bonus(algorithm: Any, algorithms_priority: list[str]) -> int:
    normalized_algorithm = _normalize_tag(algorithm)
    ordered = [_normalize_tag(item) for item in algorithms_priority if _normalize_tag(item)]
    if not normalized_algorithm or normalized_algorithm not in ordered:
        return 0
    idx = ordered.index(normalized_algorithm)
    return max(0, (len(ordered) - idx) * 12)


def _build_expert_family_profiles(
    runtime_cfg: dict[str, Any],
    scheduler_context: dict[str, Any],
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    routing_context = _build_child_model_routing_context(scheduler_context)
    taxonomy = load_expert_taxonomy()
    weight_spec = _get_routing_weight_spec(runtime_cfg)
    score_weights = weight_spec["family"]
    score_scale = float(weight_spec["score_scale"]["family"])
    by_family: dict[str, list[dict[str, Any]]] = {}
    for profile in profiles:
        family_id = _normalize_tag(profile.get("expert_family"))
        if not family_id:
            continue
        by_family.setdefault(family_id, []).append(profile)

    family_profiles: list[dict[str, Any]] = []
    for family in taxonomy.get("expert_families") or []:
        if not isinstance(family, dict):
            continue
        family_id = _normalize_tag(family.get("family_id"))
        family_entries = list(by_family.get(family_id) or [])
        if not family_entries:
            continue

        rules = family.get("selection_rules") or {}
        family_score = float(max(float(item.get("routing_priority") or 0) for item in family_entries))
        reason_parts: list[str] = []
        score_breakdown: dict[str, float] = {"routing_priority": family_score}

        matched_failures_ratio, matched_failures = _match_ratio(_expand_tags(rules.get("failure_categories")), set(routing_context.get("failure_categories") or []))
        if matched_failures:
            component = score_scale * score_weights["failure_categories"] * matched_failures_ratio
            family_score += component
            score_breakdown["failure_categories"] = component
            reason_parts.append(f"家族匹配失败类别: {', '.join(matched_failures)}")

        matched_errors_ratio, matched_errors = _match_ratio(_expand_tags(rules.get("error_patterns")), set(routing_context.get("target_error_patterns") or []))
        if matched_errors:
            component = score_scale * score_weights["error_patterns"] * matched_errors_ratio
            family_score += component
            score_breakdown["error_patterns"] = component
            reason_parts.append(f"家族匹配误差模式: {', '.join(matched_errors)}")

        matched_terrain_ratio, matched_terrain = _match_ratio(_expand_tags(rules.get("terrain_tags")), set(routing_context.get("terrain_tags") or []))
        if matched_terrain:
            component = score_scale * score_weights["terrain_tags"] * matched_terrain_ratio
            family_score += component
            score_breakdown["terrain_tags"] = component
            reason_parts.append(f"家族匹配地形标签: {', '.join(matched_terrain)}")

        matched_forest_ratio, matched_forest = _match_ratio(_expand_tags(rules.get("forest_types")), set(routing_context.get("forest_tags") or []))
        if matched_forest:
            component = score_scale * score_weights["forest_tags"] * matched_forest_ratio
            family_score += component
            score_breakdown["forest_tags"] = component
            reason_parts.append(f"家族匹配林分类型: {', '.join(matched_forest)}")

        matched_scene_ratio, matched_scene = _match_ratio(_expand_tags(rules.get("scene_tags")), set(routing_context.get("scene_tags") or []))
        if matched_scene:
            component = score_scale * score_weights["scene_tags"] * matched_scene_ratio
            family_score += component
            score_breakdown["scene_tags"] = component
            reason_parts.append(f"家族匹配场景标签: {', '.join(matched_scene)}")

        matched_roi_signal_ratio, matched_roi_signal = _match_ratio(
            _expand_tags(rules.get("scene_tags")) | _expand_tags(rules.get("failure_categories")),
            set(routing_context.get("roi_signal_tags") or []),
        )
        if matched_roi_signal:
            component = score_scale * score_weights["roi_signal_tags"] * matched_roi_signal_ratio
            family_score += component
            score_breakdown["roi_signal_tags"] = component
            reason_parts.append(f"家族匹配 ROI 信号: {', '.join(matched_roi_signal)}")

        if not reason_parts:
            reason_parts.append("未命中家族显式规则，回退为已部署专家家族默认候选。")

        ranked_family_entries = sorted(
            family_entries,
            key=lambda item: (
                float(item.get("score") or 0.0) + _algorithms_priority_bonus(item.get("algorithm"), family.get("algorithms_priority") or []),
                float(item.get("score") or 0.0),
                int(item.get("routing_priority") or 0),
            ),
            reverse=True,
        )
        family_profiles.append(
            {
                "family_id": family_id,
                "display_name": str(family.get("display_name") or family_id),
                "description": str(family.get("description") or ""),
                "algorithms_priority": [str(item) for item in (family.get("algorithms_priority") or [])],
                "selection_reason": "；".join(reason_parts),
                "score": family_score,
                "candidate_models": [str(item.get("name")) for item in ranked_family_entries if item.get("name")],
                "candidate_profiles": ranked_family_entries,
                "score_breakdown": score_breakdown,
                "matched_failures": matched_failures,
                "matched_errors": matched_errors,
                "matched_terrain": matched_terrain,
                "matched_forest": matched_forest,
                "matched_scene": matched_scene,
            }
        )

    family_profiles.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return family_profiles


def _accept_preferred_expert_family(
    preferred_family: str | None,
    family_profiles: list[dict[str, Any]],
    *,
    source: str,
    top_k: int = 2,
    max_score_gap: float = 28.0,
) -> tuple[str | None, str | None]:
    normalized = _normalize_tag(preferred_family)
    if not normalized or not family_profiles:
        return None, None

    target_idx = None
    target_profile = None
    for idx, profile in enumerate(family_profiles):
        if _normalize_tag(profile.get("family_id")) == normalized:
            target_idx = idx
            target_profile = profile
            break
    if target_profile is None or target_idx is None:
        return None, None

    if target_idx == 0:
        return normalized, f"{source} 指定专家家族: {normalized}"

    top_score = float(family_profiles[0].get("score") or 0.0)
    target_score = float(target_profile.get("score") or 0.0)
    if target_idx < min(top_k, len(family_profiles)) and (top_score - target_score) <= max_score_gap:
        return normalized, (
            f"{source} 指定专家家族: {normalized}；"
            f"家族排名第 {target_idx + 1}，且与最优家族分差 {top_score - target_score:.1f}，允许保留。"
        )
    return None, (
        f"{source} 指定专家家族 {normalized} 被规则路由拒绝；"
        f"其家族排名第 {target_idx + 1}，较最优家族低 {top_score - target_score:.1f} 分。"
    )


def _accept_preferred_expert_model(
    preferred_name: str | None,
    profiles: list[dict[str, Any]],
    *,
    source: str,
    top_k: int = 2,
    max_score_gap: float = 24.0,
) -> tuple[str | None, str | None]:
    normalized = str(preferred_name or "").strip()
    if not normalized or not profiles:
        return None, None

    target_idx = None
    target_profile = None
    for idx, profile in enumerate(profiles):
        if str(profile.get("name") or "").strip() == normalized:
            target_idx = idx
            target_profile = profile
            break
    if target_profile is None or target_idx is None:
        return None, None

    if target_idx == 0:
        return normalized, f"{source} 指定子模型模板: {normalized}"

    top_score = float(profiles[0].get("score") or 0.0)
    target_score = float(target_profile.get("score") or 0.0)
    if target_idx < min(top_k, len(profiles)) and (top_score - target_score) <= max_score_gap:
        return normalized, (
            f"{source} 指定子模型模板: {normalized}；"
            f"排名第 {target_idx + 1}，且与最优模板分差 {top_score - target_score:.1f}，允许保留。"
        )
    return None, (
        f"{source} 指定模板 {normalized} 被规则路由拒绝；"
        f"其排名第 {target_idx + 1}，较最优模板低 {top_score - target_score:.1f} 分。"
    )


def _resolve_preferred_expert_route(
    *,
    runtime_cfg: dict[str, Any],
    scheduler_context: dict[str, Any],
    llm_result: dict[str, Any] | None,
) -> dict[str, Any]:
    profiles = _rank_child_model_profiles(runtime_cfg, scheduler_context)
    family_profiles = _build_expert_family_profiles(runtime_cfg, scheduler_context, profiles)
    by_name = {_normalize_tag(item.get("name")): item for item in profiles if item.get("name")}

    expert_plan = (llm_result or {}).get("expert_model_call_plan") or (llm_result or {}).get("child_model_call_plan") or {}
    roi_plan = (llm_result or {}).get("roi_refine_plan") or {}
    roi_decision = ((scheduler_context.get("roi_assessment") or {}).get("decision") or {})

    llm_preferred_expert_model = str(
        expert_plan.get("preferred_expert_model")
        or expert_plan.get("preferred_child_model")
        or ""
    ).strip()
    roi_preferred_expert_model = str(
        roi_decision.get("preferred_expert_model")
        or roi_decision.get("preferred_child_model")
        or ""
    ).strip()
    llm_preferred_family = str(
        expert_plan.get("preferred_expert_family")
        or roi_plan.get("preferred_expert_family")
        or ((by_name.get(_normalize_tag(llm_preferred_expert_model)) or {}).get("expert_family"))
        or ""
    ).strip()
    roi_preferred_family = str(
        roi_decision.get("preferred_expert_family")
        or ((by_name.get(_normalize_tag(roi_preferred_expert_model)) or {}).get("expert_family"))
        or ""
    ).strip()

    selected_family = None
    family_reason = None
    accepted_family, family_reason = _accept_preferred_expert_family(llm_preferred_family, family_profiles, source="LLM")
    if accepted_family:
        selected_family = accepted_family
    else:
        accepted_family, roi_family_reason = _accept_preferred_expert_family(roi_preferred_family, family_profiles, source="ROI 决策")
        if accepted_family:
            selected_family = accepted_family
            family_reason = roi_family_reason
        elif family_profiles:
            selected_family = str(family_profiles[0].get("family_id") or "")
            rejected_reasons = [reason for reason in [family_reason, roi_family_reason if 'roi_family_reason' in locals() else None] if reason]
            family_reason = "；".join(rejected_reasons + [f"回退到规则最优专家家族: {selected_family}"]).strip("；")

    selected_family_profile = next(
        (item for item in family_profiles if _normalize_tag(item.get("family_id")) == _normalize_tag(selected_family)),
        family_profiles[0] if family_profiles else None,
    )
    family_candidates = list((selected_family_profile or {}).get("candidate_profiles") or [])

    child_reason = None
    preferred_expert_model = None
    accepted_name, child_reason = _accept_preferred_expert_model(llm_preferred_expert_model, family_candidates, source="LLM")
    if accepted_name:
        preferred_expert_model = accepted_name
    else:
        accepted_name, roi_child_reason = _accept_preferred_expert_model(roi_preferred_expert_model, family_candidates, source="ROI 决策")
        if accepted_name:
            preferred_expert_model = accepted_name
            child_reason = roi_child_reason
        elif family_candidates:
            preferred_expert_model = str(family_candidates[0].get("name") or "")
            rejected_reasons = [reason for reason in [child_reason, roi_child_reason if 'roi_child_reason' in locals() else None] if reason]
            child_reason = "；".join(rejected_reasons + [f"回退到家族内最优模板: {preferred_expert_model}"]).strip("；")

    if not preferred_expert_model and profiles:
        preferred_expert_model = str(profiles[0].get("name") or "")
    if not selected_family:
        selected_family = str(((by_name.get(_normalize_tag(preferred_expert_model)) or {}).get("expert_family")) or "")

    preferred_profile = by_name.get(_normalize_tag(preferred_expert_model)) or {}
    selection_reason_parts = [
        reason
        for reason in [
            family_reason,
            str((selected_family_profile or {}).get("selection_reason") or ""),
            child_reason,
            str(preferred_profile.get("selection_reason") or ""),
        ]
        if reason
    ]
    selection_reason = "；".join(dict.fromkeys(selection_reason_parts))

    candidate_models = [str(item.get("name")) for item in family_candidates if item.get("name")]
    if not candidate_models:
        candidate_models = [str(item.get("name")) for item in profiles if item.get("name")]

    return {
        "preferred_expert_family": selected_family or None,
        "preferred_expert_model": preferred_expert_model or None,
        "family_profiles": family_profiles,
        "candidate_expert_families": [str(item.get("family_id")) for item in family_profiles if item.get("family_id")],
        "candidate_expert_models": candidate_models,
        "candidate_profiles": family_candidates or profiles,
        "global_candidate_profiles": profiles,
        "preferred_profile": preferred_profile,
        "selection_reason": selection_reason or "未配置独立子模型模板，回退为当前分割引擎的 ROI 局部重跑。",
    }


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
    route = _resolve_preferred_expert_route(
        runtime_cfg=runtime_cfg,
        scheduler_context=scheduler_context,
        llm_result=llm_result,
    )
    preferred_expert_family = route.get("preferred_expert_family")
    preferred_expert_model = route.get("preferred_expert_model")
    ranked_profiles = route.get("candidate_profiles") or []
    selection_reason = str(route.get("selection_reason") or "")
    candidates = list(route.get("candidate_expert_models") or []) or _get_candidate_expert_models(runtime_cfg, scheduler_context)
    candidate_families = list(route.get("candidate_expert_families") or [])
    selection_rules = _as_str_list(llm_plan.get("selection_rules")) or [
        "优先选择综合质量分数最低且多轮未收敛的 ROI。",
        "优先细化纹理复杂、DOM 地形起伏大、实例重叠冲突明显的 ROI。",
        "先按专家家族聚合失败类别、地形标签与误差模式进行一级路由，再在家族内部选择最优专家模板。",
    ]
    stop_rules = _as_str_list(llm_plan.get("stop_rules")) or [
        "达到 ROI 质量阈值后停止。",
        "连续两轮提升不足时停止。",
        "超过最大 ROI 轮次后停止。",
    ]
    return ROIRefinePlan(
        enabled=_normalize_bool(roi_cfg.get("enabled", planning_stage == EXPERT_MODEL_ROLE)),
        use_llm=_normalize_bool(roi_cfg.get("use_llm", True)),
        max_rounds=effective_max_rounds,
        top_k=int(llm_plan.get("top_k", roi_cfg.get("top_k", 3))),
        buffer_m=float(llm_plan.get("buffer_m", roi_cfg.get("buffer_m", 5.0))),
        strategy_mode=str(llm_plan.get("strategy_mode", roi_cfg.get("strategy_mode", "auto"))),
        preferred_expert_family=str(preferred_expert_family) if preferred_expert_family else None,
        preferred_expert_model=str(preferred_expert_model) if preferred_expert_model else None,
        candidate_expert_families=candidate_families,
        candidate_expert_models=candidates,
        selection_rules=selection_rules + ([f"默认模板选择依据: {selection_reason}"] if selection_reason else []),
        stop_rules=stop_rules,
    ).to_dict()


def _build_expert_model_call_plan(
    *,
    runtime_cfg: dict[str, Any],
    scheduler_context: dict[str, Any],
    llm_result: dict[str, Any] | None,
    planning_stage: str,
) -> dict[str, Any]:
    llm_plan = (llm_result or {}).get("expert_model_call_plan") or (llm_result or {}).get("child_model_call_plan") or {}
    routing_context = _build_child_model_routing_context(scheduler_context)
    route = _resolve_preferred_expert_route(
        runtime_cfg=runtime_cfg,
        scheduler_context=scheduler_context,
        llm_result=llm_result,
    )
    preferred_expert_family = route.get("preferred_expert_family")
    preferred_expert_model = route.get("preferred_expert_model")
    ranked_profiles = route.get("candidate_profiles") or []
    family_profiles = route.get("family_profiles") or []
    selection_reason = str(route.get("selection_reason") or "")
    candidates = list(route.get("candidate_expert_models") or []) or _get_candidate_expert_models(runtime_cfg, scheduler_context)
    candidate_families = list(route.get("candidate_expert_families") or [])
    routing_rules = _as_str_list(llm_plan.get("routing_rules")) or [
        "先做专家家族一级路由，按失败模式、地形标签、林分标签和误差模式筛出最合适的专家家族。",
        "再在家族内部按算法优先级、模板得分和运行时约束选择唯一最终专家模型。",
        "全局 DEM 地形标签仅作为弱背景，不得替代 DOM/ROI 层地形上下文做主判断。",
        "默认不允许同一 ROI 同时试跑多个专家；仅在下一轮无提升时才允许家族内降级切换。",
    ]
    escalation_rules = _as_str_list(llm_plan.get("escalation_rules")) or [
        "首选子模型连续一轮无提升时切换到候选列表中的下一个模型。",
        "无可用子模型或 ROI 信息不足时返回主模型结果并停止细化。",
    ]
    routing_mode = str(llm_plan.get("routing_mode") or ("two_stage_family_routing" if ranked_profiles else "roi_quality_driven"))
    plan = ExpertModelCallPlan(
        enabled=planning_stage == EXPERT_MODEL_ROLE,
        planning_stage=planning_stage,
        routing_mode=routing_mode,
        preferred_expert_family=str(preferred_expert_family) if preferred_expert_family else None,
        preferred_expert_model=str(preferred_expert_model) if preferred_expert_model else None,
        candidate_expert_families=candidate_families,
        candidate_models=_as_str_list(llm_plan.get("candidate_models")) or candidates,
        routing_rules=routing_rules,
        escalation_rules=escalation_rules,
    ).to_dict()
    plan["family_profiles"] = family_profiles
    plan["candidate_profiles"] = ranked_profiles
    plan["global_candidate_profiles"] = route.get("global_candidate_profiles") or []
    plan["selection_reason"] = selection_reason
    plan["routing_context"] = routing_context
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
    target_model_role: str,
    target_expert_family: str | None,
    segmentation_algorithm: str | None,
    expert_training_strategy: dict[str, Any],
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
        "target_model_role": target_model_role,
        "target_expert_family": target_expert_family,
        "segmentation_algorithm": segmentation_algorithm,
        "expert_training_strategy": expert_training_strategy,
        "knowledge_injection_strategy": {
            "mode": "expert_guided",
            "target_expert_family": target_expert_family,
            "prior_axes": list(expert_training_strategy.get("prior_axes") or []),
            "replay_ratio": expert_training_strategy.get("replay_ratio"),
            "hard_case_ratio": expert_training_strategy.get("hard_case_ratio"),
            "curriculum_mode": expert_training_strategy.get("curriculum_mode"),
        },
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
    route = _resolve_preferred_expert_route(
        runtime_cfg=runtime_cfg,
        scheduler_context=scheduler_context,
        llm_result=llm_result,
    )
    preferred_profile = dict(route.get("preferred_profile") or {})
    target_model_role = str(
        llm_plan.get("target_model_role")
        or recommendation.get("target_model_role")
        or (EXPERT_MODEL_ROLE if preferred_profile else MAIN_MODEL_ROLE)
    )
    target_model_role = normalize_model_role(target_model_role, default=MAIN_MODEL_ROLE)
    target_expert_family = str(
        llm_plan.get("target_expert_family")
        or recommendation.get("target_expert_family")
        or route.get("preferred_expert_family")
        or preferred_profile.get("expert_family")
        or "cross_domain_generalist"
    )
    segmentation_algorithm = str(
        llm_plan.get("segmentation_algorithm")
        or recommendation.get("segmentation_algorithm")
        or preferred_profile.get("algorithm")
        or runtime_cfg.get("segmentation_algorithm")
        or ""
    )
    expert_defaults = build_expert_training_defaults(target_expert_family, segmentation_algorithm)
    template_path = (
        llm_plan.get("template_config_path")
        or recommendation.get("template_config_path")
        or expert_defaults.get("template_config_path")
        or resolve_expert_template_path(target_expert_family, segmentation_algorithm)
        or pipeline_cfg.get("finetune_config")
    )
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
    train_mode = str(llm_plan.get("train_mode") or expert_defaults.get("train_mode") or ("head_only" if target_module == "segmentation_model" else "full"))
    freeze_backbone = _normalize_bool(llm_plan.get("freeze_backbone", train_mode == "head_only"))
    epochs = int(llm_plan.get("epochs", expert_defaults.get("epochs", 8 if max_error >= 0.25 else 4)))
    batch_size = int(llm_plan.get("batch_size", expert_defaults.get("batch_size", 1)))
    num_workers = int(llm_plan.get("num_workers", expert_defaults.get("num_workers", 4)))
    lr = float(llm_plan.get("lr", expert_defaults.get("lr", 1.0e-4 if freeze_backbone else 5.0e-5)))
    weight_decay = float(llm_plan.get("weight_decay", expert_defaults.get("weight_decay", 1.0e-4)))
    generated_output_dir = str(
        Path(runtime_cfg.get("output_dir") or ".").resolve().parent
        / "finetune"
        / str(runtime_cfg.get("run_name") or "itd_agent")
        / str(target_expert_family or "cross_domain_generalist")
    )
    expert_training_strategy = {
        key: value
        for key, value in expert_defaults.items()
        if key
        in {
            "dataset_wrapper",
            "curriculum_mode",
            "replay_ratio",
            "hard_case_ratio",
            "prior_axes",
            "supervision_mode",
            "target_expert_family",
            "segmentation_algorithm",
        }
    }
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
        target_model_role=target_model_role,
        target_expert_family=target_expert_family,
        segmentation_algorithm=segmentation_algorithm,
        expert_training_strategy=expert_training_strategy,
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
        target_model_role=target_model_role,
        target_expert_family=target_expert_family,
        segmentation_algorithm=segmentation_algorithm,
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
        supervision_mode=str(llm_plan.get("supervision_mode") or expert_defaults.get("supervision_mode") or "hybrid"),
        dataset_bundle_path=None,
        dataset_selection_summary={},
        expert_training_strategy=expert_training_strategy,
        config_overrides=config_overrides,
        reason=str(
            llm_plan.get("reason")
            or recommendation.get("reason")
            or f"按专家家族 {target_expert_family} 与算法 {segmentation_algorithm} 生成差异化微调计划。"
        ),
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
        expert_model_call_plan = _build_expert_model_call_plan(
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
            expert_model_call_plan=expert_model_call_plan,
            knowledge_embedding_plan=knowledge_embedding_plan,
            pilot_search_result={},
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
    expert_model_call_plan = _build_expert_model_call_plan(
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
        expert_model_call_plan=expert_model_call_plan,
        knowledge_embedding_plan=knowledge_embedding_plan,
        pilot_search_result=planning_result.get("pilot_search_result") or {},
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


def build_expert_model_planning_runtime_cfg(
    *,
    cfg: dict[str, Any],
    input_assessment: dict[str, Any],
    input_manifest: dict[str, Any],
    data_processing_summary: dict[str, Any],
    roi_assessment: dict[str, Any],
    previous_round_summary: dict[str, Any],
) -> dict[str, Any]:
    plan_cfg = dict(cfg)
    plan_cfg["_planning_stage"] = EXPERT_MODEL_ROLE
    plan_cfg["_input_assessment"] = input_assessment
    plan_cfg["_input_manifest"] = input_manifest
    plan_cfg["_data_processing_summary"] = data_processing_summary
    plan_cfg["_roi_assessment"] = roi_assessment
    plan_cfg["_previous_round_summary"] = previous_round_summary
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
    return build_expert_model_planning_runtime_cfg(
        cfg=cfg,
        input_assessment=input_assessment,
        input_manifest=input_manifest,
        data_processing_summary=data_processing_summary,
        roi_assessment=roi_assessment,
        previous_round_summary=previous_round_summary,
    )
