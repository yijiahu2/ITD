from __future__ import annotations

import json
from typing import Any

from .retrospective_input import build_run_retrospective_input


def _short_text(value: Any, limit: int = 160) -> str | None:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _take_list(value: Any, limit: int = 5) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[: max(int(limit), 0)]


def _compact_child_model_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": entry.get("name") or entry.get("algorithm"),
        "algorithm": entry.get("algorithm"),
        "description": _short_text(entry.get("description"), 120),
        "scene_tags": _take_list(entry.get("scene_tags") or entry.get("scene_labels"), 6),
        "terrain_tags": _take_list(entry.get("terrain_tags"), 6),
        "failure_categories": _take_list(entry.get("failure_categories"), 6),
        "target_error_patterns": _take_list(entry.get("target_error_patterns"), 6),
        "routing_priority": entry.get("routing_priority"),
        "runtime_overrides": entry.get("runtime_overrides") or {},
    }


def _compact_template_cfg(template_cfg: dict[str, Any]) -> dict[str, Any]:
    itd_cfg = template_cfg.get("ITD_agent") or {}
    planning_cfg = itd_cfg.get("planning") or {}
    seg_cfg = itd_cfg.get("segmentation_models") or {}
    return {
        "runtime": {
            "run_name": (template_cfg.get("runtime") or {}).get("run_name"),
            "conda_env": (template_cfg.get("runtime") or {}).get("conda_env"),
        },
        "planning": {
            "adaptive_generation": planning_cfg.get("adaptive_generation") or {},
            "roi_extraction": planning_cfg.get("roi_extraction") or planning_cfg.get("roi_refine") or {},
            "grouped_inference": planning_cfg.get("grouped_inference") or {},
        },
        "segmentation_models": {
            "main_model": (seg_cfg.get("main_model") or {}),
            "child_models": [
                _compact_child_model_entry(item)
                for item in _take_list(seg_cfg.get("child_models"), 6)
                if isinstance(item, dict)
            ],
        },
        "default_segmentation_params": {
            key: template_cfg.get(key)
            for key in ["diam_list", "tile", "overlap", "tile_overlap", "augment", "iou_merge_thr", "bsize"]
            if key in template_cfg
        },
    }


def _compact_top_problem_cases(details_summary: dict[str, Any]) -> list[dict[str, Any]]:
    cases = details_summary.get("top_k_xiaoban") or []
    compacted: list[dict[str, Any]] = []
    for case in cases[:5]:
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


def _compact_memory_rows(rows: Any, *, limit: int = 3) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return compacted
    for item in rows[:limit]:
        if not isinstance(item, dict):
            continue
        compacted.append(
            {
                "memory_type": item.get("memory_type"),
                "run_name": item.get("run_name"),
                "reason": _short_text(item.get("reason"), 120),
                "strategy_summary": _short_text(item.get("strategy_summary"), 120),
                "failure_modes": _take_list(item.get("failure_modes"), 3),
                "tags": _take_list(item.get("tags"), 6),
            }
        )
    return compacted


def _compact_recent_failed_cases(rows: Any, *, limit: int = 3) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return compacted
    for item in rows[:limit]:
        if not isinstance(item, dict):
            continue
        compacted.append(
            {
                "sample_id": item.get("sample_id"),
                "failure_category": item.get("failure_category"),
                "target_model_role": item.get("target_model_role"),
                "xiaoban_id": ((item.get("metadata") or {}).get("xiaoban_id")),
            }
        )
    return compacted


def _compact_scheduler_context(scheduler_context: dict[str, Any]) -> dict[str, Any]:
    scene_profile = scheduler_context.get("scene_profile") or {}
    segmentation_reco = scheduler_context.get("segmentation_parameter_recommendation") or {}
    roi_assessment = scheduler_context.get("roi_assessment") or {}
    return {
        "run_name": scheduler_context.get("run_name"),
        "planning_stage": scheduler_context.get("planning_stage"),
        "template_runtime": scheduler_context.get("template_runtime") or {},
        "current_parameters": scheduler_context.get("current_parameters") or {},
        "evaluation_metrics": scheduler_context.get("evaluation_metrics") or {},
        "scene_profile": {
            "forest_type": scene_profile.get("forest_type"),
            "terrain_type": scene_profile.get("terrain_type"),
            "knowledge_profile_types": _take_list(scene_profile.get("knowledge_profile_types"), 6),
            "public_dataset_roles": _take_list(scene_profile.get("public_dataset_roles"), 6),
            "stand_condition_labels": _take_list(scene_profile.get("stand_condition_labels"), 6),
            "texture_labels": _take_list(scene_profile.get("texture_labels"), 6),
            "quality_labels": _take_list(scene_profile.get("quality_labels"), 6),
            "terrain_labels": _take_list(scene_profile.get("terrain_labels"), 6),
        },
        "image_texture_analysis": {
            "labels": _take_list((scheduler_context.get("image_texture_analysis") or {}).get("labels"), 6),
            "levels": (scheduler_context.get("image_texture_analysis") or {}).get("levels") or {},
        },
        "image_quality_analysis": {
            "labels": _take_list((scheduler_context.get("image_quality_analysis") or {}).get("labels"), 6),
            "levels": (scheduler_context.get("image_quality_analysis") or {}).get("levels") or {},
        },
        "terrain_analysis": {
            "labels": _take_list((scheduler_context.get("terrain_analysis") or {}).get("labels"), 8),
            "global_background": scheduler_context.get("global_terrain_background") or {},
            "dom_context": scheduler_context.get("dom_terrain_context") or {},
        },
        "details_summary": {
            "num_units": (scheduler_context.get("details_summary") or {}).get("num_units"),
            "top_problem_cases": _compact_top_problem_cases(scheduler_context.get("details_summary") or {}),
        },
        "roi_assessment": {
            "quality_label": roi_assessment.get("quality_label"),
            "current_score": roi_assessment.get("current_score"),
            "improvement": roi_assessment.get("improvement"),
            "trigger_metrics": _take_list(roi_assessment.get("trigger_metrics"), 5),
            "candidate_roi_ids": [
                item.get("candidate_id") or item.get("xiaoban_id")
                for item in _take_list(roi_assessment.get("candidate_rois"), 5)
                if isinstance(item, dict)
            ],
        },
        "segmentation_parameter_recommendation": {
            "model_family": segmentation_reco.get("model_family"),
            "parameter_updates": segmentation_reco.get("parameter_updates") or {},
            "confidence": segmentation_reco.get("confidence"),
            "reasons": _take_list(segmentation_reco.get("reasons"), 4),
            "evidence": {
                key: value
                for key, value in (segmentation_reco.get("evidence") or {}).items()
                if key
                in {
                    "crown_width_m",
                    "crown_width_px",
                    "density_mean",
                    "closure_mean",
                    "resolution_m",
                    "image_width",
                    "image_height",
                    "scene_tags",
                    "texture_levels",
                    "quality_levels",
                    "global_terrain_background",
                    "dom_terrain_context",
                    "terrain_labels",
                }
            },
        },
        "knowledge_profiles": [
            {
                "source_id": item.get("source_id"),
                "normalized_type": item.get("normalized_type"),
                "role": item.get("role"),
                "tags": _take_list(item.get("tags"), 6),
            }
            for item in _take_list(scheduler_context.get("knowledge_profiles"), 5)
            if isinstance(item, dict)
        ],
        "public_dataset_profiles": [
            {
                "dataset_id": item.get("dataset_id"),
                "role": item.get("role"),
                "tags": _take_list(item.get("tags"), 6),
            }
            for item in _take_list(scheduler_context.get("public_dataset_profiles"), 5)
            if isinstance(item, dict)
        ],
        "memory_store_context": _compact_memory_rows(scheduler_context.get("memory_store_context"), limit=3),
        "scene_similar_memory_context": _compact_memory_rows(scheduler_context.get("scene_similar_memory_context"), limit=3),
        "failure_pattern_context": _compact_memory_rows(scheduler_context.get("failure_pattern_context"), limit=3),
        "finetune_pool_recent_cases": _compact_recent_failed_cases(scheduler_context.get("finetune_pool_recent_cases"), limit=3),
    }


def _compact_roi_assessment(roi_assessment: dict[str, Any]) -> dict[str, Any]:
    return {
        "assessment_phase": roi_assessment.get("assessment_phase"),
        "round_idx": roi_assessment.get("round_idx"),
        "quality_label": roi_assessment.get("quality_label"),
        "current_score": roi_assessment.get("current_score"),
        "previous_score": roi_assessment.get("previous_score"),
        "improvement": roi_assessment.get("improvement"),
        "trigger_metrics": _take_list(roi_assessment.get("trigger_metrics"), 5),
        "candidate_rois": [
            {
                "candidate_id": item.get("candidate_id") or item.get("xiaoban_id"),
                "score": item.get("score"),
                "priority_score": item.get("priority_score"),
                "terrain_score_mean": item.get("terrain_score_mean"),
                "boundary_score_mean": item.get("boundary_score_mean"),
                "prior_overlap_ratio": item.get("prior_overlap_ratio"),
            }
            for item in _take_list(roi_assessment.get("candidate_rois"), 5)
            if isinstance(item, dict)
        ],
        "heuristic_continue": roi_assessment.get("heuristic_continue"),
        "candidate_source": roi_assessment.get("candidate_source"),
        "signal_roi_summary": {
            "selected_candidate_count": len((roi_assessment.get("signal_roi_summary") or {}).get("selected_candidates") or []),
            "pruned_candidate_count": (roi_assessment.get("signal_roi_summary") or {}).get("pruned_candidate_count"),
            "pruned_candidate_ids": _take_list((roi_assessment.get("signal_roi_summary") or {}).get("pruned_candidate_ids"), 5),
        },
    }


def _compact_candidate_rois(candidate_rois: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in candidate_rois[:6]:
        compacted.append(
            {
                "candidate_id": item.get("candidate_id"),
                "xiaoban_id": item.get("xiaoban_id"),
                "score": item.get("score"),
                "priority_score": item.get("priority_score"),
                "problem_type": item.get("problem_type"),
                "prior_overlap_ratio": item.get("prior_overlap_ratio"),
                "terrain_score_mean": item.get("terrain_score_mean"),
                "boundary_score_mean": item.get("boundary_score_mean"),
                "texture_score_mean": item.get("texture_score_mean"),
                "aspect_class": item.get("aspect_class"),
                "landform_type": item.get("landform_type"),
            }
        )
    return compacted


def _build_planning_prompt(
    *,
    planning_stage: str,
    template_cfg: dict[str, Any],
    scheduler_context: dict[str, Any],
) -> str:
    stage_label = "主模型" if planning_stage == "main_model" else "子模型"
    compact_template_cfg = _compact_template_cfg(template_cfg)
    compact_scheduler_context = _compact_scheduler_context(scheduler_context)
    return f"""
你是 ITD_agent 的 LLM网关，当前负责 {stage_label} 配置决策。
你的任务不是重写整份配置，而是基于配置模板、评估分析结果、记忆上下文和处理中上下文，产出结构化参数更新建议。

配置模板：
{json.dumps(compact_template_cfg, ensure_ascii=False, indent=2)}

调度上下文：
{json.dumps(compact_scheduler_context, ensure_ascii=False, indent=2)}

要求：
1. 只输出需要更新的参数。
2. 重点考虑主/子模型运行配置、ROI 规则、子模型调用规则、微调训练配置、知识嵌入规则和中间数据处理规则。
3. 若上下文中存在影像纹理指标或纹理标签，例如 contrast、entropy、asm、energy、correlation、homogeneity、texture_complex、texture_smooth，应将其纳入场景判断与参数决策。
4. 若上下文中存在影像质量指标或质量标签，例如 blur、overexposed_ratio、underexposed_ratio、shadow_ratio_estimate、stripe_noise_score、color_cast_score、blur_high、shadow_heavy、stripe_noise、color_cast，也必须纳入参数决策。
5. 若当前主模型是 legacy_cellpose_sam / Cellpose-SAM，必须优先判断并输出这些核心参数是否需要更新：diam_list、tile、overlap、tile_overlap、augment、iou_merge_thr、bsize。
6. 对 Cellpose-SAM 的参数判断必须结合影像分辨率、先验平均冠幅、郁闭度、密度、纹理复杂度、边缘强度、模糊、阴影、曝光异常、条带噪声以及色偏风险，不能只做泛化描述。
7. 若上下文中同时存在 global_terrain_background 和 dom_terrain_context，必须区分两者角色：
   - global_terrain_background 只作为整体场景背景和弱约束
   - dom_terrain_context 才是主模型参数与子模型排序的主要地形依据
8. ROI / 子模型相关决策时，禁止用全局 DEM 地形标签替代 DOM/ROI 层地形上下文。
9. 若近期成功策略可复用，优先复用。
10. 若近期失败模式反复出现，给出进入微调池与记忆库的建议。
11. 只输出 JSON。

输出格式：
{{
  "parameter_updates": {{}},
  "roi_refine_plan": {{
    "max_rounds": 2,
    "top_k": 3,
    "buffer_m": 5.0,
    "strategy_mode": "auto",
    "selection_rules": ["..."],
    "stop_rules": ["..."]
  }},
  "child_model_call_plan": {{
    "preferred_child_model": "可选",
    "candidate_models": ["..."],
    "routing_rules": ["..."],
    "escalation_rules": ["..."]
  }},
  "finetune_training_plan": {{
    "should_prepare": false,
    "target_module": "segmentation_model",
    "trigger_mode": "defer_until_pool_threshold",
    "train_mode": "head_only",
    "freeze_backbone": false,
    "epochs": 4,
    "batch_size": 1,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "data_selection_rule": "..."
  }},
  "knowledge_embedding_plan": {{
    "backbone_rules": [{{"rule": "...", "condition": "...", "parameter_hint": "..."}}],
    "neck_rules": [{{"rule": "...", "condition": "...", "parameter_hint": "..."}}],
    "initial_prediction_rules": [{{"rule": "...", "condition": "...", "parameter_hint": "..."}}],
    "head_rules": [{{"rule": "...", "condition": "...", "parameter_hint": "..."}}],
    "config_hints": {{}}
  }},
  "reason": "简短说明",
  "memory_rule": "成功策略写入记忆库规则",
  "finetune_rule": "失败样本写入微调池规则"
}}
"""


def _build_roi_decision_prompt(
    *,
    roi_assessment: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    compact_roi_assessment = _compact_roi_assessment(roi_assessment)
    return f"""
你是 ITD_agent 的 LLM网关，当前负责 ROI 细化决策。
请基于 ROI 评估和当前指标，判断是否继续调用子模型细化。

ROI评估：
{json.dumps(compact_roi_assessment, ensure_ascii=False, indent=2)}

当前指标：
{json.dumps(metrics, ensure_ascii=False, indent=2)}

只输出 JSON：
{{
  "continue_refinement": true,
  "reason": "简短说明",
  "preferred_child_model": "可选，若不需要可省略"
}}
    """


def _build_roi_candidate_selection_prompt(
    *,
    candidate_rois: list[dict[str, Any]],
    metrics: dict[str, Any],
    scene_analysis: dict[str, Any] | None = None,
) -> str:
    compact_scene_analysis = {
        "forest_type": (scene_analysis or {}).get("forest_type"),
        "stand_condition": ((scene_analysis or {}).get("stand_condition") or {}).get("labels") or [],
        "texture_labels": ((scene_analysis or {}).get("image_texture_analysis") or {}).get("labels") or [],
        "quality_labels": ((scene_analysis or {}).get("image_quality_analysis") or {}).get("labels") or [],
        "terrain_labels": ((scene_analysis or {}).get("terrain_analysis") or {}).get("labels") or [],
        "dom_terrain_context": ((scene_analysis or {}).get("terrain_analysis") or {}).get("dom_context") or {},
    }
    compact_candidates = _compact_candidate_rois(candidate_rois)
    return f"""
你是 ITD_agent 的 LLM网关，当前负责 ROI 候选区域排序。
请基于候选区域的纹理、阴影、地形复杂度、实例边界异常、语义先验覆盖情况，以及当前场景分析和全局指标，
从这些“已生成的候选区域”中选出最值得优先细化的 ROI。你不能虚构新的几何区域，只能在候选列表中排序和筛选。
其中：
- 全局地形标签只可作为背景说明
- DOM/ROI 层地形复杂度、坡向、坡位才是 ROI 优先级判断的主依据之一

场景分析：
{json.dumps(compact_scene_analysis, ensure_ascii=False, indent=2)}

当前全局指标：
{json.dumps(metrics, ensure_ascii=False, indent=2)}

候选 ROI：
{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}

只输出 JSON：
{{
  "selected_candidate_ids": ["signal_roi_00_01", "signal_roi_00_02"],
  "reason": "简短说明",
  "discarded_candidate_ids": ["signal_roi_00_03"]
}}
"""


def _build_retrospective_prompt(
    *,
    run_summary: dict[str, Any],
    memory_context: list[dict[str, Any]] | None,
    finetune_context: list[dict[str, Any]] | None,
) -> str:
    retrospective_input = build_run_retrospective_input(
        run_summary=run_summary,
        memory_context=memory_context,
        finetune_context=finetune_context,
    )
    return f"""
你是 ITD_agent 的 LLM网关，当前负责本轮运行复盘与知识更新建议。

请严格基于下面这个“模板化复盘输入”进行总结，不要自行假设不存在的上下文。

模板化复盘输入：
{json.dumps(retrospective_input, ensure_ascii=False, indent=2)}

请输出 JSON，总结：
1. 本轮成功策略
2. 本轮主要失败模式
3. 是否建议写入记忆库
4. 是否建议进入微调池
5. 是否建议触发微调训练

输出格式：
{{
  "success_strategies": ["..."],
  "failure_modes": ["..."],
  "memory_update": {{
    "should_record": true,
    "reason": "..."
  }},
  "finetune_pool_update": {{
    "should_enqueue": true,
    "reason": "..."
  }},
  "training_recommendation": {{
    "should_trigger": false,
    "target_module": "segmentation_model",
    "reason": "..."
  }}
}}
"""
