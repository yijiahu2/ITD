from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.evaluation_analysis.detail_ranker import summarize_details_csv
from ITD_agent.finetune_pool.query import load_finetune_pool_snapshot, load_recent_failed_cases
from ITD_agent.memory_store.query import (
    infer_scene_profile_from_runtime,
    load_recent_execution_traces,
    load_recent_failure_patterns,
    load_recent_success_strategies,
    load_scene_similar_memories,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _round_to_step(value: float, *, step: int, min_value: int, max_value: int) -> int:
    rounded = int(round(value / float(step)) * step)
    return max(min_value, min(max_value, rounded))


def _build_legacy_cellpose_sam_parameter_recommendation(runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    input_assessment = runtime_cfg.get("_input_assessment") or {}
    scene_analysis = input_assessment.get("scene_analysis") or {}
    stand_condition = scene_analysis.get("stand_condition") or {}
    inventory_stats = scene_analysis.get("inventory_scene_stats") or {}
    texture = (scene_analysis.get("image_texture_analysis") or {})
    quality = (scene_analysis.get("image_quality_analysis") or {})
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
    ]:
        if value:
            scene_tags.add(str(value))

    high_complexity = texture_levels.get("complexity") == "high" or "texture_complex" in texture_labels
    strong_edge = texture_levels.get("edge_strength") == "strong" or "strong_edge" in texture_labels
    closed_canopy = stand_condition.get("closure_level") == "high" or "closed_canopy" in scene_tags
    high_density = stand_condition.get("density_level") == "high" or "high_density" in scene_tags
    blur_high = quality_levels.get("blur") in {"high", "severe"} or "blur_high" in quality_labels
    heavy_shadow = quality_levels.get("shadow") in {"moderate", "heavy"} or any(label in quality_labels for label in {"shadow_heavy", "shadow_moderate"})
    stripe_noise = quality_levels.get("stripe_noise") in {"medium", "high"} or "stripe_noise" in quality_labels
    exposure_risk = quality_levels.get("exposure") in {"high", "severe"} or bool({"overexposed", "underexposed"} & quality_labels)
    color_cast = quality_levels.get("color_cast") in {"medium", "high"} or "color_cast" in quality_labels

    crown_width_px = None
    if crown_width_m is not None and resolution_m and resolution_m > 0:
        crown_width_px = crown_width_m / resolution_m

    reasons: list[str] = []
    parameter_updates: dict[str, Any] = {}

    if crown_width_px is not None:
        d_small = _round_to_step(crown_width_px * 0.70, step=32, min_value=96, max_value=224)
        d_mid = _round_to_step(crown_width_px * 1.10, step=32, min_value=160, max_value=320)
        d_large = _round_to_step(crown_width_px * 1.65, step=32, min_value=224, max_value=384)
        diameters = []
        for value in [d_small, d_mid, d_large]:
            if value not in diameters:
                diameters.append(value)
        if len(diameters) >= 2:
            parameter_updates["diam_list"] = ",".join(str(x) for x in diameters)
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

    if blur_high:
        parameter_updates["tile_overlap"] = max(float(parameter_updates.get("tile_overlap", 0.35)), 0.40)
        parameter_updates["augment"] = True
        parameter_updates["overlap"] = max(int(parameter_updates.get("overlap", 128)), 192)
        reasons.append("影像模糊较重，建议提高滑窗重叠并开启 augment 以降低边界不稳定影响。")
    if heavy_shadow:
        parameter_updates["tile_overlap"] = max(float(parameter_updates.get("tile_overlap", 0.35)), 0.40)
        parameter_updates["augment"] = True
        parameter_updates["iou_merge_thr"] = max(float(parameter_updates.get("iou_merge_thr", 0.35)), 0.35)
        reasons.append("阴影占比较高，建议提高上下文重叠并保持较稳健的实例合并阈值。")
    if stripe_noise:
        parameter_updates["tile"] = min(int(parameter_updates.get("tile", 2048)), 1792)
        parameter_updates["overlap"] = max(int(parameter_updates.get("overlap", 128)), 256)
        parameter_updates["tile_overlap"] = max(float(parameter_updates.get("tile_overlap", 0.35)), 0.40)
        reasons.append("存在条带噪声时，建议适当减小 tile、增大 overlap，降低方向性伪影影响。")
    if exposure_risk or color_cast:
        parameter_updates["augment"] = True
        reasons.append("曝光异常或色偏会削弱颜色稳定性，建议开启 augment 提高鲁棒性。")

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
    metrics: dict[str, Any] = {}
    if metrics_json and Path(metrics_json).exists():
        metrics = _load_json(metrics_json)

    summary: dict[str, Any] = {}
    if summary_json and Path(summary_json).exists():
        summary = _load_json(summary_json)

    details_summary = {}
    if details_csv and Path(details_csv).exists():
        details_summary = summarize_details_csv(details_csv, top_k=10)

    return {
        "scene_profile": infer_scene_profile_from_runtime(runtime_cfg),
        "run_name": runtime_cfg.get("run_name"),
        "planning_stage": runtime_cfg.get("_planning_stage"),
        "pipeline": runtime_cfg.get("pipeline") or {},
        "template_runtime": {
            "input_image": runtime_cfg.get("input_image"),
            "xiaoban_shp": runtime_cfg.get("xiaoban_shp"),
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
        "data_processing_summary": runtime_cfg.get("_data_processing_summary") or {},
        "roi_assessment": runtime_cfg.get("_roi_assessment") or {},
        "previous_round_summary": runtime_cfg.get("_previous_round_summary") or {},
        "input_manifest": runtime_cfg.get("_input_manifest") or {},
        "segmentation_models": ((runtime_cfg.get("ITD_agent") or {}).get("segmentation_models") or {}),
        "segmentation_parameter_recommendation": _build_legacy_cellpose_sam_parameter_recommendation(runtime_cfg),
        "knowledge_profiles": (runtime_cfg.get("_data_processing_summary") or {}).get("knowledge_profiles") or [],
        "public_dataset_profiles": (runtime_cfg.get("_data_processing_summary") or {}).get("public_dataset_profiles") or [],
        "memory_store_context": load_recent_success_strategies(limit=recent_success_limit),
        "failure_pattern_context": load_recent_failure_patterns(limit=recent_failure_limit),
        "execution_trace_context": load_recent_execution_traces(limit=recent_success_limit),
        "scene_similar_memory_context": load_scene_similar_memories(
            scene_profile=infer_scene_profile_from_runtime(runtime_cfg),
            limit=min(recent_success_limit, 5),
        ),
        "finetune_pool_context": load_finetune_pool_snapshot(),
        "finetune_pool_recent_cases": load_recent_failed_cases(limit=recent_failure_limit),
    }
