from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from ITD_agent.config_adapter import save_runtime_config
from ITD_agent.llm_gateway import request_planning_decision
from ITD_agent.planning.scheduler.context_builder import build_scheduler_context
from ITD_agent.planning.scheduler.parameter_search import run_main_model_parameter_search
from ITD_agent.planning.scheduler.template_manager import apply_parameter_updates, load_config_template


def _deep_merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _preserve_runtime_context(generated_cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Preserve orchestration blocks from the live runtime config so generated planning
    configs keep agent metadata such as llm/planning/model templates. Avoid copying
    the high-level input/output schema blocks because those would trigger a second
    normalization pass and rewrite the flat runtime fields.
    """
    merged = deepcopy(generated_cfg)
    for key in ["pipeline", "ITD_agent"]:
        runtime_block = runtime_cfg.get(key)
        generated_block = merged.get(key)
        if isinstance(runtime_block, dict) and isinstance(generated_block, dict):
            merged[key] = _deep_merge_dict(generated_block, runtime_block)
        elif key not in merged and runtime_block is not None:
            merged[key] = deepcopy(runtime_block)

    for key, value in runtime_cfg.items():
        if not str(key).startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(value, merged[key])
        elif key not in merged:
            merged[key] = deepcopy(value)

    for key in ["grouped_inference_enabled", "grouped_inference_use_llm", "grouped_inference_buffer_m"]:
        if key in runtime_cfg:
            merged[key] = deepcopy(runtime_cfg[key])
    # Preserve the live flattened runtime fields so planning templates do not
    # overwrite per-run paths or normalized inputs.
    for key in [
        "experiment_name",
        "run_name",
        "input_image",
        "dem_tif",
        "slope_tif",
        "aspect_tif",
        "landform_tif",
        "slope_position_tif",
        "xiaoban_shp",
        "xiaoban_id_field",
        "tree_count_field",
        "crown_field",
        "closure_field",
        "density_field",
        "area_ha_field",
        "output_dir",
        "persistent_output_dir",
        "metrics_json",
        "details_csv",
        "cleanup_policy",
        "cleanup_temp_runtime",
        "use_temp_runtime",
        "conda_sh",
        "conda_env",
        "work_dir",
        "semantic_prior_script",
        "semantic_prior_ckpt",
        "semantic_prior_extra_args",
        "segmentation_script",
        "segmentation_algorithm",
        "segmentation_algorithm_module",
        "segmentation_algorithm_cfg",
        "diam_list",
        "tile",
        "overlap",
        "tile_overlap",
        "bsize",
        "augment",
        "iou_merge_thr",
        "flat_slope_threshold_deg",
        "plain_relief_threshold_m",
        "terrain_landform_field",
        "terrain_slope_class_field",
        "terrain_aspect_class_field",
        "terrain_slope_position_field",
        "disable_mlflow",
    ]:
        if key in runtime_cfg:
            merged[key] = deepcopy(runtime_cfg[key])
    return merged


def _enforce_runtime_caps(generated_cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    capped = deepcopy(generated_cfg)
    runtime_planning = ((runtime_cfg.get("ITD_agent") or {}).get("planning") or {})
    runtime_roi = runtime_planning.get("roi_extraction") or runtime_planning.get("roi_refine") or {}
    if isinstance(runtime_roi, dict) and "max_rounds" in runtime_roi:
        capped.setdefault("ITD_agent", {}).setdefault("planning", {}).setdefault("roi_extraction", {})
        capped["ITD_agent"]["planning"]["roi_extraction"]["max_rounds"] = int(runtime_roi["max_rounds"])
    capped["bsize"] = 256
    return capped


def build_scheduler_prompt(
    *,
    template_cfg: dict[str, Any],
    scheduler_context: dict[str, Any],
) -> str:
    return f"""
你是 ITD_agent 的规划调度模块。
你的任务不是重新设计整份配置，而是基于已有配置模板和评估分析结果，推理出需要更新的参数。

配置模板：
{json.dumps(template_cfg, ensure_ascii=False, indent=2)}

调度上下文：
{json.dumps(scheduler_context, ensure_ascii=False, indent=2)}

要求：
1. 只输出需要更新的参数，不要重复整份模板。
2. 优先考虑：
   - 主模型/子模型运行配置表
   - 主模型/子模型微调训练配置
   - ROI 区域提取参数
   - 先验知识嵌入规则
   - 中间数据处理规则
3. 若近期成功策略可复用，优先复用并解释。
4. 若近期失败案例反复出现，给出进入微调集的触发规则。
5. 只输出 JSON。

输出格式：
{{
  "parameter_updates": {{
    "key": "value"
  }},
  "reason": "简短说明",
  "memory_rule": "成功策略写入记忆库规则",
  "finetune_rule": "失败样本写入微调集规则"
}}
"""


def _fallback_updates(template_cfg: dict[str, Any], scheduler_context: dict[str, Any]) -> dict[str, Any]:
    metrics = scheduler_context.get("evaluation_metrics") or {}
    texture_analysis = scheduler_context.get("image_texture_analysis") or {}
    texture_levels = texture_analysis.get("levels") or {}
    texture_labels = set(texture_analysis.get("labels") or [])
    updates: dict[str, Any] = {}
    tree_err = metrics.get("tree_count_error_ratio")
    crown_err = metrics.get("mean_crown_width_error_ratio")
    if tree_err is not None and float(tree_err) > 0.25:
        updates["tile_overlap"] = 0.35
        updates["overlap"] = 512
    if crown_err is not None and float(crown_err) > 0.25:
        updates["diam_list"] = "128,192,320"
    if not metrics:
        if texture_levels.get("complexity") == "high" or "texture_complex" in texture_labels:
            updates.setdefault("tile_overlap", 0.3)
            updates.setdefault("overlap", 384)
        if texture_levels.get("continuity") == "low" or "texture_discontinuous" in texture_labels:
            updates.setdefault("iou_merge_thr", 0.22)
        if texture_levels.get("uniformity") == "high" and texture_levels.get("smoothness") == "high":
            updates.setdefault("diam_list", "160,224,320")
        if texture_levels.get("edge_strength") == "strong":
            updates.setdefault("tile_overlap", 0.35)
    if template_cfg.get("ITD_agent"):
        updates["ITD_agent"] = {
            "planning": {
                "memory_rule": "子模型分割成功且关键误差达标时写入记忆库。",
                "finetune_rule": "同类区域多次分割失败时写入微调集。",
            }
        }
    return updates


def _merge_core_segmentation_updates(
    *,
    base_updates: dict[str, Any],
    scheduler_context: dict[str, Any],
    llm_result: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = deepcopy(base_updates or {})
    recommendation = scheduler_context.get("segmentation_parameter_recommendation") or {}
    recommended_updates = recommendation.get("parameter_updates") or {}
    recommendation_confidence = float(recommendation.get("confidence") or 0.0)
    if not isinstance(recommended_updates, dict) or not recommended_updates:
        return merged

    core_keys = {"diam_list", "tile", "overlap", "tile_overlap", "augment", "iou_merge_thr", "bsize"}
    model_family = str(recommendation.get("model_family") or "").strip().lower()
    has_llm = bool(llm_result)
    if has_llm and model_family == "legacy_cellpose_sam" and recommendation_confidence >= 0.75:
        for key in core_keys:
            if key in recommended_updates:
                merged[key] = deepcopy(recommended_updates[key])
        return merged
    if has_llm:
        for key in core_keys:
            if key not in merged and key in recommended_updates:
                merged[key] = deepcopy(recommended_updates[key])
        return merged

    for key in core_keys:
        if key in recommended_updates:
            merged[key] = deepcopy(recommended_updates[key])
    return merged


def _is_parameter_search_enabled(runtime_cfg: dict[str, Any]) -> bool:
    planning_cfg = ((runtime_cfg.get("ITD_agent") or {}).get("planning") or {})
    adaptive_cfg = planning_cfg.get("adaptive_generation") or {}
    search_cfg = adaptive_cfg.get("parameter_search")
    if not isinstance(search_cfg, dict):
        return False
    return bool(search_cfg.get("enabled", False))


def generate_adaptive_config_from_template(
    *,
    template_path: str | Path,
    output_path: str | Path,
    runtime_cfg: dict[str, Any],
    metrics_json: str | None = None,
    details_csv: str | None = None,
    summary_json: str | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    template_cfg = load_config_template(template_path)
    scheduler_context = build_scheduler_context(
        runtime_cfg=runtime_cfg,
        metrics_json=metrics_json,
        details_csv=details_csv,
        summary_json=summary_json,
    )

    llm_result: dict[str, Any] | None = None
    llm_gateway_result: dict[str, Any] | None = None
    if use_llm:
        planning_stage = str(runtime_cfg.get("_planning_stage") or "main_model")
        llm_gateway_result = request_planning_decision(
            planning_stage=planning_stage,
            template_cfg=template_cfg,
            scheduler_context=scheduler_context,
            runtime_cfg=runtime_cfg,
            use_llm=use_llm,
        )
        llm_result = llm_gateway_result.get("parsed_result") if isinstance(llm_gateway_result, dict) else None

    parameter_updates = (llm_result or {}).get("parameter_updates") or _fallback_updates(template_cfg, scheduler_context)
    parameter_updates = _merge_core_segmentation_updates(
        base_updates=parameter_updates,
        scheduler_context=scheduler_context,
        llm_result=llm_result,
    )
    pilot_search_result: dict[str, Any] | None = None
    if (
        str(runtime_cfg.get("_planning_stage") or "").strip().lower() == "main_model"
        and _is_parameter_search_enabled(runtime_cfg)
    ):
        pilot_search_result = run_main_model_parameter_search(
            runtime_cfg=runtime_cfg,
            scheduler_context=scheduler_context,
            preliminary_updates=parameter_updates,
            output_root=Path(output_path).resolve().parent / "pilot_parameter_search",
        )
        selected_updates = (pilot_search_result or {}).get("selected_parameter_updates") or {}
        if isinstance(selected_updates, dict) and selected_updates:
            for key in ["diam_list", "tile", "overlap", "tile_overlap", "augment", "iou_merge_thr", "bsize"]:
                if key in selected_updates:
                    parameter_updates[key] = deepcopy(selected_updates[key])
        scheduler_context["pilot_parameter_search"] = pilot_search_result or {}
    generated_cfg = apply_parameter_updates(template_cfg, parameter_updates)
    generated_cfg = _preserve_runtime_context(generated_cfg, runtime_cfg)
    # Re-apply the explicit planning updates after runtime preservation so only
    # intentional scheduler outputs override live per-run settings.
    generated_cfg = apply_parameter_updates(generated_cfg, parameter_updates)
    generated_cfg = _enforce_runtime_caps(generated_cfg, runtime_cfg)
    save_runtime_config(generated_cfg, output_path)
    return {
        "template_path": str(template_path),
        "output_path": str(output_path),
        "scheduler_context": scheduler_context,
        "parameter_updates": parameter_updates,
        "llm_result": llm_result,
        "llm_gateway_result": llm_gateway_result,
        "pilot_search_result": pilot_search_result or {},
    }
