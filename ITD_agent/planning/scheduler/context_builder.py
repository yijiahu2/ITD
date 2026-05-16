from __future__ import annotations

from pathlib import Path
from typing import Any

from input_layer.mainline_profiles import get_mainline_capabilities, resolve_mainline_profile

from ITD_agent.common.values import safe_float as _safe_float
from ITD_agent.evaluation_analysis.detail_ranker import summarize_details_csv
from ITD_agent.finetune_pool.query import load_finetune_pool_snapshot, load_recent_failed_cases
from ITD_agent.memory_store.query import (
    infer_scene_profile_from_runtime,
    load_recent_execution_traces,
    load_recent_failure_patterns,
    load_recent_success_strategies,
    load_scene_similar_memories,
)
from ITD_agent.skill_store.matcher import match_skill_context
from ITD_agent.skill_store.query import load_skill_records


def _load_json(path: str | Path) -> dict[str, Any]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _input_manifest_summary(data_processing_summary: dict[str, Any]) -> dict[str, Any]:
    metadata = data_processing_summary.get("metadata") or {}
    summary = metadata.get("input_manifest_summary") or {}
    return summary if isinstance(summary, dict) else {}


def _round_to_step(value: float, *, step: int, min_value: int, max_value: int) -> int:
    rounded = int(round(value / float(step)) * step)
    return max(min_value, min(max_value, rounded))


def _normalize_diam_triplet(values: list[int]) -> str:
    normalized: list[int] = []
    for value in values:
        value = int(value)
        if value not in normalized:
            normalized.append(value)
    return ",".join(str(item) for item in normalized)


def _build_legacy_cellpose_sam_parameter_recommendation(runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    input_assessment = runtime_cfg.get("_input_assessment") or {}
    scene_analysis = input_assessment.get("scene_analysis") or {}
    stand_condition = scene_analysis.get("stand_condition") or {}
    inventory_stats = scene_analysis.get("inventory_scene_stats") or {}
    texture = (scene_analysis.get("image_texture_analysis") or {})
    quality = (scene_analysis.get("image_quality_analysis") or {})
    terrain_analysis = (scene_analysis.get("terrain_analysis") or {})
    global_terrain = terrain_analysis.get("global_background") or {}
    dom_terrain = terrain_analysis.get("dom_context") or {}
    terrain_labels = set(terrain_analysis.get("labels") or [])
    texture_levels = texture.get("levels") or {}
    texture_labels = set(texture.get("labels") or [])
    quality_levels = quality.get("levels") or {}
    quality_labels = set(quality.get("labels") or [])
    data_processing_summary = runtime_cfg.get("_data_processing_summary") or {}
    image_profiles = data_processing_summary.get("image_profiles") or []
    image_profile = image_profiles[0] if image_profiles else {}

    crown_width_m = _safe_float(inventory_stats.get("crown_width_mean"))
    density_mean = _safe_float(inventory_stats.get("density_mean"))
    closure_mean = _safe_float(inventory_stats.get("closure_mean"))
    resolution_m = _safe_float(image_profile.get("resolution_x_m")) or _safe_float(image_profile.get("resolution_y_m"))
    image_width = int(image_profile.get("width") or 0)
    image_height = int(image_profile.get("height") or 0)
    max_dim = max(image_width, image_height, 0)

    scene_tags = set()
    for value in [
        runtime_cfg.get("forest_type"),
        scene_analysis.get("forest_type"),
        *(stand_condition.get("labels") or []),
        *texture_labels,
        *quality_labels,
        *terrain_labels,
    ]:
        if value:
            scene_tags.add(str(value))

    high_complexity = texture_levels.get("complexity") == "high" or "texture_complex" in texture_labels
    strong_edge = texture_levels.get("edge_strength") == "strong" or "strong_edge" in texture_labels
    closed_canopy = stand_condition.get("closure_level") == "high" or "closed_canopy" in scene_tags
    high_density = stand_condition.get("density_level") == "high" or "high_density" in scene_tags
    blur_high = quality_levels.get("blur") in {"high", "severe"} or "blur_high" in quality_labels
    shadow_level = str(quality_levels.get("shadow") or "").lower()
    shadow_present = shadow_level in {"moderate", "heavy"} or any(label in quality_labels for label in {"shadow_heavy", "shadow_moderate"})
    heavy_shadow = shadow_level == "heavy" or "shadow_heavy" in quality_labels
    stripe_noise = quality_levels.get("stripe_noise") in {"medium", "high"} or "stripe_noise" in quality_labels
    exposure_risk = quality_levels.get("exposure") in {"high", "severe"} or bool({"overexposed", "underexposed"} & quality_labels)
    color_cast = quality_levels.get("color_cast") in {"medium", "high"} or "color_cast" in quality_labels
    dom_steep = "dom_steep_surface" in terrain_labels or (_safe_float(dom_terrain.get("slope_mean_deg")) or 0.0) >= 25.0
    dom_shadow_aspect = "dom_shadow_aspect" in terrain_labels
    dom_transition = "dom_ridge_valley_transition" in terrain_labels
    dom_upper_slope = "dom_upper_slope" in terrain_labels
    global_shadow_background = "global_shadow_aspect_background" in terrain_labels

    crown_width_px = None
    if crown_width_m is not None and resolution_m and resolution_m > 0:
        crown_width_px = crown_width_m / resolution_m

    reasons: list[str] = []
    parameter_updates: dict[str, Any] = {}

    high_res_closed_complex_scene = bool(
        max_dim >= 3500
        and crown_width_px is not None
        and crown_width_px >= 180
        and closed_canopy
        and (high_complexity or strong_edge)
    )

    if crown_width_px is not None:
        d_small = _round_to_step(crown_width_px * 0.75, step=32, min_value=96, max_value=224)
        d_mid = _round_to_step(crown_width_px * 1.25, step=32, min_value=160, max_value=320)
        d_large = _round_to_step(crown_width_px * 1.85, step=32, min_value=224, max_value=384)
        diameters = []
        for value in [d_small, d_mid, d_large]:
            if value not in diameters:
                diameters.append(value)
        if len(diameters) >= 2:
            parameter_updates["diam_list"] = _normalize_diam_triplet(diameters)
            reasons.append(
                f"根据平均冠幅 {crown_width_m:.2f} m 和分辨率 {resolution_m:.4f} m/px，估算平均树冠尺度约 {crown_width_px:.0f} px，"
                f"建议采用多尺度直径 {parameter_updates['diam_list']}。"
            )

    if max_dim >= 3500:
        parameter_updates["tile"] = 2048
        parameter_updates["overlap"] = 128
        reasons.append(f"影像尺寸约为 {image_width}x{image_height}，建议整图滑窗使用 tile=2048、overlap=128 以控制跨窗重复干扰。")
    elif max_dim >= 2200:
        parameter_updates["tile"] = 1792
        parameter_updates["overlap"] = 128

    if high_complexity or strong_edge or closed_canopy:
        parameter_updates["tile_overlap"] = 0.35
        parameter_updates["augment"] = True
        reasons.append("闭冠/强边缘/高复杂纹理场景下，Cellpose 内部推理宜提高 tile_overlap 并开启 augment。")

    if closed_canopy and (high_complexity or strong_edge):
        parameter_updates["iou_merge_thr"] = 0.50
        reasons.append("密闭冠层且纹理复杂时，提高多尺度/跨窗实例合并阈值可减少误合并。")
    elif high_density or high_complexity:
        parameter_updates["iou_merge_thr"] = 0.35

    if high_res_closed_complex_scene:
        parameter_updates["diam_list"] = "160,256,384"
        parameter_updates["tile"] = 2048
        parameter_updates["overlap"] = 128
        parameter_updates["tile_overlap"] = max(float(parameter_updates.get("tile_overlap", 0.35)), 0.35)
        parameter_updates["augment"] = True
        parameter_updates["iou_merge_thr"] = 0.50
        reasons.append("高分辨率闭冠复杂场景优先采用实测更稳的大窗多尺度配置，避免过度保守导致欠分割。")

    if blur_high:
        parameter_updates["tile_overlap"] = max(float(parameter_updates.get("tile_overlap", 0.35)), 0.40)
        parameter_updates["augment"] = True
        parameter_updates["overlap"] = max(int(parameter_updates.get("overlap", 128)), 192)
        reasons.append("影像模糊较重，建议提高滑窗重叠并开启 augment 以降低边界不稳定影响。")
    if shadow_present:
        parameter_updates["tile_overlap"] = max(float(parameter_updates.get("tile_overlap", 0.35)), 0.35)
        parameter_updates["augment"] = True
        if heavy_shadow:
            parameter_updates["tile_overlap"] = max(float(parameter_updates.get("tile_overlap", 0.35)), 0.40)
            parameter_updates["overlap"] = max(int(parameter_updates.get("overlap", 128)), 192)
        reasons.append("阴影存在时优先保持大窗和多尺度配置，仅温和提高重叠与增强来补偿阴影影响。")
    if stripe_noise:
        parameter_updates["augment"] = True
        parameter_updates["tile_overlap"] = max(float(parameter_updates.get("tile_overlap", 0.35)), 0.35)
        if not high_res_closed_complex_scene and (blur_high or heavy_shadow):
            parameter_updates["overlap"] = max(int(parameter_updates.get("overlap", 128)), 192)
        reasons.append("条带噪声优先通过增强和适度上下文补偿处理，避免直接缩小 tile 导致整体欠分割。")
    if exposure_risk or color_cast:
        parameter_updates["augment"] = True
        reasons.append("曝光异常或色偏会削弱颜色稳定性，建议开启 augment 提高鲁棒性。")
    if dom_steep or dom_transition:
        parameter_updates["tile_overlap"] = max(float(parameter_updates.get("tile_overlap", 0.35)), 0.40)
        parameter_updates["overlap"] = max(int(parameter_updates.get("overlap", 128)), 192)
        reasons.append("DOM 范围处于陡坡或脊谷过渡带时，优先提高滑窗上下文重叠，降低地形驱动的边界不稳定。")
    if dom_shadow_aspect:
        parameter_updates["augment"] = True
        parameter_updates["tile_overlap"] = max(float(parameter_updates.get("tile_overlap", 0.35)), 0.40)
        reasons.append("DOM 范围以阴坡为主，作为主决策依据提高阴影鲁棒性。")
    if dom_upper_slope and high_density:
        parameter_updates["iou_merge_thr"] = max(float(parameter_updates.get("iou_merge_thr", 0.35)), 0.40)
        reasons.append("DOM 范围位于上坡位且林分密度较高，应适当提高实例合并阈值，减少破碎冠幅。")
    if global_shadow_background and not dom_shadow_aspect:
        reasons.append("全局 DEM 背景存在阴坡特征，但当前仅作为弱约束参与专家模型排序，不直接主导主模型参数。")

    if density_mean is not None and density_mean >= 450:
        reasons.append(f"小班先验平均密度约 {density_mean:.1f} 株/公顷，属于中高密度林分。")
    if closure_mean is not None and closure_mean >= 0.70:
        reasons.append(f"小班先验平均郁闭度约 {closure_mean:.2f}，属于闭冠场景。")

    confidence = 0.0
    populated = sum(1 for value in [crown_width_px, density_mean, closure_mean, resolution_m] if value is not None)
    if populated >= 3:
        confidence = 0.9
    elif populated == 2:
        confidence = 0.75
    elif populated == 1:
        confidence = 0.55

    return {
        "model_family": "legacy_cellpose_sam",
        "parameter_updates": parameter_updates,
        "evidence": {
            "crown_width_m": crown_width_m,
            "crown_width_px": crown_width_px,
            "density_mean": density_mean,
            "closure_mean": closure_mean,
            "resolution_m": resolution_m,
            "image_width": image_width,
            "image_height": image_height,
            "scene_tags": sorted(scene_tags),
            "texture_levels": texture_levels,
            "quality_levels": quality_levels,
            "global_terrain_background": global_terrain,
            "dom_terrain_context": dom_terrain,
            "terrain_labels": sorted(terrain_labels),
        },
        "confidence": confidence,
        "reasons": reasons,
    }


def build_scheduler_context(
    *,
    runtime_cfg: dict[str, Any],
    metrics_json: str | None = None,
    details_csv: str | None = None,
    summary_json: str | None = None,
    recent_success_limit: int = 10,
    recent_failure_limit: int = 10,
) -> dict[str, Any]:
    mainline_profile = resolve_mainline_profile(runtime_cfg)
    capabilities = runtime_cfg.get("_mainline_capabilities") or get_mainline_capabilities(mainline_profile)
    allow_dem = bool(capabilities.get("allow_dem"))
    allow_external_knowledge = bool(capabilities.get("allow_external_knowledge"))
    allow_public_datasets = bool(capabilities.get("allow_public_datasets"))
    allow_memory_context = bool(capabilities.get("allow_memory_context"))
    allow_finetune_pool_context = bool(capabilities.get("allow_finetune_pool_context"))
    metrics: dict[str, Any] = {}
    if metrics_json and Path(metrics_json).exists():
        metrics = _load_json(metrics_json)

    summary: dict[str, Any] = {}
    if summary_json and Path(summary_json).exists():
        summary = _load_json(summary_json)

    details_summary = {}
    if details_csv and Path(details_csv).exists():
        details_summary = summarize_details_csv(details_csv, top_k=10)

    scene_profile = infer_scene_profile_from_runtime(runtime_cfg)
    terrain_analysis = ((runtime_cfg.get("_input_assessment") or {}).get("scene_analysis") or {}).get("terrain_analysis") or {}
    data_processing_summary = runtime_cfg.get("_data_processing_summary") or {}
    failure_pattern_context = load_recent_failure_patterns(limit=recent_failure_limit) if allow_memory_context else []
    skill_records = load_skill_records(
        db_path=runtime_cfg.get("state_db_path"),
        review_output_dir=runtime_cfg.get("review_output_dir"),
        statuses=["draft", "shadow", "active"],
        limit=50,
    ) if allow_memory_context else []
    skill_context = match_skill_context(
        skills=skill_records,
        scene_profile=scene_profile,
        evaluation_metrics=metrics,
        roi_assessment=runtime_cfg.get("_roi_assessment") or {},
        failure_pattern_context=failure_pattern_context,
    ) if allow_memory_context else {"matched_skill_count": 0, "matched_skills": [], "application_mode": "context_only_readonly_suggestion"}

    return {
        "mainline_profile": mainline_profile,
        "mainline_capabilities": capabilities,
        "scene_profile": scene_profile,
        "run_name": runtime_cfg.get("run_name"),
        "planning_stage": runtime_cfg.get("_planning_stage"),
        "pipeline": runtime_cfg.get("pipeline") or {},
        "template_runtime": {
            "input_image": runtime_cfg.get("input_image"),
            "reference_vector_path": runtime_cfg.get("reference_vector_path") or runtime_cfg.get("inventory_vector_path") or runtime_cfg.get("xiaoban_shp"),
            "dem_tif": runtime_cfg.get("dem_tif"),
        },
        "current_parameters": {
            key: runtime_cfg.get(key)
            for key in ["diam_list", "tile", "overlap", "tile_overlap", "augment", "iou_merge_thr", "segmentation_algorithm"]
            if key in runtime_cfg
        },
        "evaluation_metrics": metrics,
        "details_summary": details_summary,
        "summary_snapshot": {
            "mode": summary.get("mode"),
            "group_count": summary.get("group_count"),
            "cleanup": summary.get("cleanup"),
        },
        "input_assessment": runtime_cfg.get("_input_assessment") or {},
        "image_texture_analysis": ((runtime_cfg.get("_input_assessment") or {}).get("scene_analysis") or {}).get("image_texture_analysis") or {},
        "image_quality_analysis": ((runtime_cfg.get("_input_assessment") or {}).get("scene_analysis") or {}).get("image_quality_analysis") or {},
        "terrain_analysis": terrain_analysis if allow_dem else {},
        "global_terrain_background": (terrain_analysis.get("global_background") or {}) if allow_dem else {},
        "dom_terrain_context": (terrain_analysis.get("dom_context") or {}) if allow_dem else {},
        "data_processing_summary": data_processing_summary,
        "roi_assessment": runtime_cfg.get("_roi_assessment") or {},
        "previous_round_summary": runtime_cfg.get("_previous_round_summary") or {},
        "input_manifest": runtime_cfg.get("_input_manifest") or {},
        "segmentation_models": ((runtime_cfg.get("ITD_agent") or {}).get("segmentation_models") or {}),
        "segmentation_parameter_recommendation": _build_legacy_cellpose_sam_parameter_recommendation(runtime_cfg),
        "knowledge_profiles": (_input_manifest_summary(data_processing_summary).get("domain_knowledge_items") or []) if allow_external_knowledge else [],
        "public_dataset_profiles": (_input_manifest_summary(data_processing_summary).get("public_datasets") or []) if allow_public_datasets else [],
        "memory_store_context": load_recent_success_strategies(limit=recent_success_limit) if allow_memory_context else [],
        "failure_pattern_context": failure_pattern_context,
        "execution_trace_context": load_recent_execution_traces(limit=recent_success_limit) if allow_memory_context else [],
        "scene_similar_memory_context": load_scene_similar_memories(
            scene_profile=scene_profile,
            limit=min(recent_success_limit, 5),
        ) if allow_memory_context else [],
        "finetune_pool_context": load_finetune_pool_snapshot() if allow_finetune_pool_context else [],
        "finetune_pool_recent_cases": load_recent_failed_cases(limit=recent_failure_limit) if allow_finetune_pool_context else [],
        "skill_context": skill_context,
    }


def build_evolve_infer_plan_context(
    *,
    cfg: dict[str, Any],
    input_manifest: dict[str, Any],
    data_processing_context: dict[str, Any],
    experience_context: dict[str, Any],
) -> dict[str, Any]:
    mainline_profile = resolve_mainline_profile(cfg)
    capabilities = get_mainline_capabilities(mainline_profile)
    main_model_cfg = dict(cfg.get("main_model") or {})
    expert_models_cfg = dict(cfg.get("expert_models") or {})
    model_configs = dict(cfg.get("model_configs") or {})
    route_policy = dict(cfg.get("expert_routing_policy") or {})
    public_dataset_summary = dict(data_processing_context.get("public_dataset_summary") or {})

    model_profiles = []
    for model_id, model_cfg in model_configs.items():
        model_profiles.append(
            {
                "model_id": str(model_id),
                "model_role": "expert_model" if str(model_id) != str(main_model_cfg.get("model_id")) else "main_model",
                "segmentation_algorithm": model_cfg.get("segmentation_algorithm"),
                "execution_mode": expert_models_cfg.get("execution_mode") if str(model_id) != str(main_model_cfg.get("model_id")) else main_model_cfg.get("execution_mode"),
                "capability_source": "config.model_configs",
            }
        )
    if main_model_cfg.get("model_id") and not any(item["model_id"] == str(main_model_cfg["model_id"]) for item in model_profiles):
        model_profiles.append(
            {
                "model_id": str(main_model_cfg["model_id"]),
                "model_role": "main_model",
                "execution_mode": main_model_cfg.get("execution_mode", "prediction_json"),
                "capability_source": "config.main_model",
            }
        )

    memory_context = experience_context.get("memory_context") or {}
    return {
        "mainline_profile": mainline_profile,
        "mainline_capabilities": capabilities,
        "gt_leakage_guard": {
            "policy": "COCO GT annotation content is only consumed by evaluation_analysis.",
            "main_model_plan_uses_gt": False,
            "expert_routing_uses_gt": False,
        },
        "input_context": {
            "manifest_status": ((input_manifest.get("validation") or {}).get("status") or "unknown"),
            "input_modalities": (input_manifest.get("metadata") or {}).get("input_modalities") or {},
            "public_dataset_count": len(input_manifest.get("public_datasets") or []),
            "public_dataset_summary": {
                "dataset_format": public_dataset_summary.get("dataset_format"),
                "image_count": public_dataset_summary.get("image_count"),
                "selected_image_count": public_dataset_summary.get("selected_image_count"),
                "category_count": public_dataset_summary.get("category_count"),
            },
        },
        "experience_injection": {
            "enabled": bool(experience_context.get("enabled", True)),
            "recent_success_count": len(memory_context.get("recent_success") or []),
            "recent_failure_count": len(memory_context.get("recent_failure") or []),
            "recent_execution_count": len(memory_context.get("recent_execution") or []),
            "skill_record_count": int((experience_context.get("skill_context") or {}).get("record_count") or 0),
            "application_mode": "readonly_context_injection",
        },
        "main_model_plan": {
            "model_id": main_model_cfg.get("model_id", "legacy_cellpose_sam"),
            "execution_mode": main_model_cfg.get("execution_mode", "prediction_json"),
            "planning_policy": "dom_image_or_coco_dataset_real_inference",
            "input_policy": "DOM image and configured prediction/runtime inputs only; no GT instances in prompt or model input.",
            "repair_interface": {
                "supported": False,
                "reserved_decision": "retry_main_plan",
                "reason": "Plan repair hook is reserved; current minimal path uses objective ROI escalation.",
            },
        },
        "roi_strategy": {
            "policy": dict(cfg.get("roi_policy") or {}),
            "source": "evaluation_analysis_error_decomposition_and_geometry_review",
            "statuses": ["record_only", "monitor", "actionable"],
        },
        "expert_routing_context": {
            "policy_version": route_policy.get("version", "v1_rule_based"),
            "route_map": route_policy.get("route_map") or route_policy.get("expert_map") or {},
            "model_profiles": model_profiles,
            "routing_history": experience_context.get("expert_routing_history") or [],
            "llm_decision_policy": "LLM may explain routing only; objective rules select accept/reject/fusion.",
        },
    }
