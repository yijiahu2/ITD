from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from .contracts import InputAssessment


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _normalize_closure(value: Any) -> float | None:
    closure = _safe_float(value)
    if closure is None:
        return None
    if closure > 1.5:
        closure = closure / 100.0
    if closure < 0:
        return None
    return min(closure, 1.0)


def _bucketize(value: float | None, *, high: float, medium: float, high_label: str, medium_label: str, low_label: str) -> str | None:
    if value is None:
        return None
    if value >= high:
        return high_label
    if value >= medium:
        return medium_label
    return low_label


def _mean_numeric(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return None
    return float(values.mean())


def _load_inventory_scene_stats(cfg: dict[str, Any]) -> dict[str, Any]:
    xiaoban_path = cfg.get("xiaoban_shp")
    if not xiaoban_path or not Path(str(xiaoban_path)).exists():
        return {}

    try:
        gdf = gpd.read_file(xiaoban_path)
    except Exception as exc:
        return {"status": "failed", "reason": f"读取小班矢量失败: {exc}"}

    if gdf.empty:
        return {"status": "empty", "reason": "小班矢量为空。"}

    closure_field = cfg.get("closure_field")
    density_field = cfg.get("density_field")
    crown_field = cfg.get("crown_field")
    tree_count_field = cfg.get("tree_count_field")
    area_ha_field = cfg.get("area_ha_field")

    closure_mean = _mean_numeric(gdf[closure_field].map(_normalize_closure) if closure_field in gdf.columns else None)
    density_mean = _mean_numeric(gdf[density_field] if density_field in gdf.columns else None)
    crown_mean = _mean_numeric(gdf[crown_field] if crown_field in gdf.columns else None)
    tree_count_mean = _mean_numeric(gdf[tree_count_field] if tree_count_field in gdf.columns else None)
    area_ha_mean = _mean_numeric(gdf[area_ha_field] if area_ha_field in gdf.columns else None)

    if density_mean is None and tree_count_field in gdf.columns and area_ha_field in gdf.columns:
        derived_density = pd.to_numeric(gdf[tree_count_field], errors="coerce") / pd.to_numeric(gdf[area_ha_field], errors="coerce").replace(0, pd.NA)
        density_mean = _mean_numeric(derived_density)

    return {
        "status": "ok",
        "feature_count": int(len(gdf)),
        "closure_mean": closure_mean,
        "density_mean": density_mean,
        "crown_width_mean": crown_mean,
        "tree_count_mean": tree_count_mean,
        "area_ha_mean": area_ha_mean,
        "field_sources": {
            "closure": closure_field if closure_field in gdf.columns else None,
            "density": density_field if density_field in gdf.columns else None,
            "crown_width": crown_field if crown_field in gdf.columns else None,
            "tree_count": tree_count_field if tree_count_field in gdf.columns else None,
            "area_ha": area_ha_field if area_ha_field in gdf.columns else None,
        },
    }


def _infer_forest_type(cfg: dict[str, Any], stats: dict[str, Any]) -> tuple[str | None, str, list[str]]:
    configured = cfg.get("forest_type")
    if configured:
        return str(configured), "config", ["配置中已显式指定 forest_type。"]

    closure_mean = _safe_float(stats.get("closure_mean"))
    density_mean = _safe_float(stats.get("density_mean"))
    crown_mean = _safe_float(stats.get("crown_width_mean"))
    reasons: list[str] = []

    if closure_mean is not None:
        reasons.append(f"平均郁闭度约为 {closure_mean:.2f}。")
    if density_mean is not None:
        reasons.append(f"平均密度约为 {density_mean:.1f} 株/公顷。")
    if crown_mean is not None:
        reasons.append(f"平均冠幅约为 {crown_mean:.2f} m。")

    if closure_mean is None and density_mean is None and crown_mean is None:
        return None, "unknown", ["缺少可用于推断森林类型的小班统计字段。"]

    if closure_mean is not None and closure_mean >= 0.70:
        if density_mean is not None and density_mean >= 450:
            return "dense_mixed", "heuristic", reasons + ["高郁闭度且密度偏高，按稠密混交林处理。"]
        return "dense_forest", "heuristic", reasons + ["郁闭度较高，按密闭林分处理。"]
    if closure_mean is not None and closure_mean <= 0.40:
        if crown_mean is not None and crown_mean >= 5.5:
            return "open_large_crown", "heuristic", reasons + ["低郁闭度且冠幅偏大，按稀疏大冠幅林分处理。"]
        return "sparse_open_forest", "heuristic", reasons + ["郁闭度较低，按稀疏开阔林分处理。"]
    if density_mean is not None and density_mean >= 700 and (crown_mean is None or crown_mean < 4.5):
        return "young_dense_stand", "heuristic", reasons + ["密度较高且冠幅偏小，按幼龄稠密林分处理。"]
    return "mixed_forest", "heuristic", reasons + ["整体指标处于中间区间，按一般混交林场景处理。"]


def _build_scene_analysis(cfg: dict[str, Any]) -> dict[str, Any]:
    stats = _load_inventory_scene_stats(cfg)
    forest_type, source, reasons = _infer_forest_type(cfg, stats)

    closure_level = _bucketize(
        _safe_float(stats.get("closure_mean")),
        high=0.70,
        medium=0.40,
        high_label="high",
        medium_label="medium",
        low_label="low",
    )
    density_level = _bucketize(
        _safe_float(stats.get("density_mean")),
        high=900.0,
        medium=450.0,
        high_label="high",
        medium_label="medium",
        low_label="low",
    )
    crown_level = _bucketize(
        _safe_float(stats.get("crown_width_mean")),
        high=6.0,
        medium=3.5,
        high_label="large",
        medium_label="medium",
        low_label="small",
    )

    stand_condition_labels: list[str] = []
    if closure_level == "high":
        stand_condition_labels.append("closed_canopy")
    elif closure_level == "low":
        stand_condition_labels.append("open_canopy")
    if density_level == "high":
        stand_condition_labels.append("high_density")
    elif density_level == "low":
        stand_condition_labels.append("low_density")
    if crown_level == "large":
        stand_condition_labels.append("large_crown")
    elif crown_level == "small":
        stand_condition_labels.append("small_crown")
    if closure_level == "high" and density_level == "high":
        stand_condition_labels.append("dense_compact_stand")
    elif closure_level == "low" and crown_level == "large":
        stand_condition_labels.append("sparse_large_crown_stand")

    scene_tags = []
    for value in [forest_type, *stand_condition_labels]:
        if value and value not in scene_tags:
            scene_tags.append(value)

    return {
        "forest_type": forest_type,
        "forest_type_source": source,
        "forest_type_reasons": reasons,
        "inventory_scene_stats": stats,
        "stand_condition": {
            "closure_level": closure_level,
            "density_level": density_level,
            "crown_width_level": crown_level,
            "labels": stand_condition_labels,
        },
        "scene_tags": scene_tags,
    }


def _categorize_texture(value: float | None, *, high: float, medium: float, high_label: str, medium_label: str, low_label: str) -> str | None:
    return _bucketize(
        value,
        high=high,
        medium=medium,
        high_label=high_label,
        medium_label=medium_label,
        low_label=low_label,
    )


def _build_image_texture_analysis(data_processing_summary: dict[str, Any] | None) -> dict[str, Any]:
    summary = data_processing_summary or {}
    image_profiles = summary.get("image_profiles") or []
    if not image_profiles:
        return {}

    texture = (image_profiles[0] or {}).get("texture_summary") or {}
    if not texture:
        return {}

    contrast = _safe_float(texture.get("contrast"))
    entropy = _safe_float(texture.get("entropy"))
    asm = _safe_float(texture.get("asm"))
    energy = _safe_float(texture.get("energy"))
    correlation = _safe_float(texture.get("correlation"))
    homogeneity = _safe_float(texture.get("homogeneity"))
    gradient_mean = _safe_float(texture.get("gradient_mean"))

    edge_strength = _categorize_texture(
        contrast if contrast is not None else gradient_mean,
        high=4.0,
        medium=2.0,
        high_label="strong",
        medium_label="medium",
        low_label="weak",
    )
    complexity = _categorize_texture(
        entropy,
        high=5.5,
        medium=4.5,
        high_label="high",
        medium_label="medium",
        low_label="low",
    )
    uniformity = _categorize_texture(
        energy if energy is not None else asm,
        high=0.22,
        medium=0.12,
        high_label="high",
        medium_label="medium",
        low_label="low",
    )
    continuity = _categorize_texture(
        correlation,
        high=0.75,
        medium=0.45,
        high_label="high",
        medium_label="medium",
        low_label="low",
    )
    smoothness = _categorize_texture(
        homogeneity,
        high=0.55,
        medium=0.35,
        high_label="high",
        medium_label="medium",
        low_label="low",
    )

    labels: list[str] = []
    reasons: list[str] = []
    if edge_strength == "strong":
        labels.append("strong_edge")
        reasons.append("Contrast 偏高，边界和局部起伏较强。")
    if complexity == "high":
        labels.append("texture_complex")
        reasons.append("Entropy 偏高，纹理复杂度较高。")
    if uniformity == "high":
        labels.append("texture_uniform")
        reasons.append("Energy/ASM 偏高，局部更规整均匀。")
    elif uniformity == "low":
        labels.append("texture_nonuniform")
        reasons.append("Energy/ASM 偏低，局部均匀性较弱。")
    if continuity == "low":
        labels.append("texture_discontinuous")
        reasons.append("Correlation 偏低，纹理连续性较弱。")
    elif continuity == "high":
        labels.append("texture_continuous")
        reasons.append("Correlation 偏高，纹理连续性较好。")
    if smoothness == "high":
        labels.append("texture_smooth")
        reasons.append("Homogeneity/IDM 偏高，区域较平滑。")
    elif smoothness == "low":
        labels.append("texture_rough")
        reasons.append("Homogeneity/IDM 偏低，邻域差异较明显。")

    return {
        "metrics": {
            "contrast": contrast,
            "entropy": entropy,
            "asm": asm,
            "energy": energy,
            "correlation": correlation,
            "homogeneity": homogeneity,
            "idm": _safe_float(texture.get("idm")),
            "gradient_mean": gradient_mean,
            "gradient_std": _safe_float(texture.get("gradient_std")),
        },
        "levels": {
            "edge_strength": edge_strength,
            "complexity": complexity,
            "uniformity": uniformity,
            "continuity": continuity,
            "smoothness": smoothness,
        },
        "labels": labels,
        "reasons": reasons,
    }


def _build_image_quality_analysis(data_processing_summary: dict[str, Any] | None) -> dict[str, Any]:
    summary = data_processing_summary or {}
    image_profiles = summary.get("image_profiles") or []
    if not image_profiles:
        return {}

    quality_summary = (image_profiles[0] or {}).get("quality_summary") or {}
    metrics = quality_summary.get("quality_metrics") or {}
    if not metrics:
        return {}

    lap_var = _safe_float(metrics.get("laplacian_variance"))
    tenengrad = _safe_float(metrics.get("tenengrad"))
    over_ratio = _safe_float(metrics.get("overexposed_ratio")) or 0.0
    under_ratio = _safe_float(metrics.get("underexposed_ratio")) or 0.0
    shadow_ratio = _safe_float(metrics.get("shadow_ratio_estimate")) or 0.0
    stripe_score = _safe_float(metrics.get("stripe_noise_score")) or 0.0
    color_cast_score = _safe_float(metrics.get("color_cast_score")) or 0.0

    if (lap_var is not None and lap_var < 40.0) or (tenengrad is not None and tenengrad < 300.0):
        blur_level = "severe"
    elif (lap_var is not None and lap_var < 100.0) or (tenengrad is not None and tenengrad < 700.0):
        blur_level = "high"
    elif (lap_var is not None and lap_var < 220.0) or (tenengrad is not None and tenengrad < 1600.0):
        blur_level = "medium"
    else:
        blur_level = "low"

    exposure_peak = max(over_ratio, under_ratio)
    if exposure_peak >= 0.20:
        exposure_level = "severe"
    elif exposure_peak >= 0.08:
        exposure_level = "high"
    elif exposure_peak >= 0.03:
        exposure_level = "medium"
    else:
        exposure_level = "low"

    if shadow_ratio >= 0.35:
        shadow_level = "heavy"
    elif shadow_ratio >= 0.18:
        shadow_level = "moderate"
    elif shadow_ratio >= 0.08:
        shadow_level = "light"
    else:
        shadow_level = "low"

    if stripe_score >= 0.35:
        stripe_level = "high"
    elif stripe_score >= 0.22:
        stripe_level = "medium"
    elif stripe_score >= 0.12:
        stripe_level = "low"
    else:
        stripe_level = "none"

    if color_cast_score >= 0.18:
        color_cast_level = "high"
    elif color_cast_score >= 0.10:
        color_cast_level = "medium"
    elif color_cast_score >= 0.05:
        color_cast_level = "low"
    else:
        color_cast_level = "none"

    labels: list[str] = []
    reasons: list[str] = []
    if blur_level in {"high", "severe"}:
        labels.append("blur_high")
        reasons.append("Laplacian/梯度清晰度偏低，影像存在明显模糊。")
    if over_ratio >= 0.08:
        labels.append("overexposed")
        reasons.append("高亮饱和像元占比较高，存在过曝风险。")
    if under_ratio >= 0.08:
        labels.append("underexposed")
        reasons.append("极暗像元占比较高，存在欠曝风险。")
    if shadow_level in {"moderate", "heavy"}:
        labels.append("shadow_heavy" if shadow_level == "heavy" else "shadow_moderate")
        reasons.append("低亮度阴影区域占比较高，可能干扰树冠边界。")
    if stripe_level in {"medium", "high"}:
        labels.append("stripe_noise")
        reasons.append("行列方向存在异常周期波动，疑似条带噪声。")
    if color_cast_level in {"medium", "high"}:
        labels.append("color_cast")
        reasons.append("RGB 通道均值失衡明显，存在色偏。")

    return {
        "metrics": {
            "laplacian_variance": lap_var,
            "tenengrad": tenengrad,
            "overexposed_ratio": over_ratio,
            "underexposed_ratio": under_ratio,
            "shadow_ratio_estimate": shadow_ratio,
            "stripe_noise_score": stripe_score,
            "stripe_noise_row_score": _safe_float(metrics.get("stripe_noise_row_score")),
            "stripe_noise_col_score": _safe_float(metrics.get("stripe_noise_col_score")),
            "stripe_noise_direction": metrics.get("stripe_noise_direction"),
            "color_cast_score": color_cast_score,
            "channel_means": metrics.get("channel_means") or {},
            "dominant_channel": metrics.get("dominant_channel"),
            "brightness_mean": _safe_float(metrics.get("brightness_mean")),
            "brightness_std": _safe_float(metrics.get("brightness_std")),
        },
        "levels": {
            "blur": blur_level,
            "exposure": exposure_level,
            "shadow": shadow_level,
            "stripe_noise": stripe_level,
            "color_cast": color_cast_level,
        },
        "labels": labels,
        "reasons": reasons,
    }


def _normalize_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _build_terrain_analysis(terrain_info: dict[str, Any] | None) -> dict[str, Any]:
    info = terrain_info or {}
    global_background = info.get("global_terrain_background") or {}
    dom_context = info.get("dom_terrain_context") or {}
    xiaoban_summary = info.get("xiaoban_terrain_class_summary") or info.get("terrain_class_summary") or {}

    labels: list[str] = []
    reasons: list[str] = []

    global_landform = _normalize_text(global_background.get("landform_type"))
    global_slope = _normalize_text(global_background.get("slope_class"))
    global_aspect = _normalize_text(global_background.get("aspect_class"))
    global_position = _normalize_text(global_background.get("slope_position_class"))

    dom_landform = _normalize_text(dom_context.get("landform_type"))
    dom_slope = _normalize_text(dom_context.get("slope_class"))
    dom_aspect = _normalize_text(dom_context.get("aspect_class"))
    dom_position = _normalize_text(dom_context.get("slope_position_class"))
    dom_mean_slope = _safe_float(dom_context.get("slope_mean_deg"))

    if global_landform:
        labels.append(f"global_landform_{global_landform}")
        reasons.append(f"全局 DEM 背景地貌为 {global_landform}。")
    if global_position:
        labels.append(f"global_slope_position_{global_position}")
    if global_aspect in {"north", "northeast", "northwest"}:
        labels.append("global_shadow_aspect_background")
        reasons.append("全局背景存在偏阴坡特征，仅作为弱约束参与子模型排序。")
    if global_landform in {"ridge_transition", "valley_transition"}:
        labels.append("global_transition_background")
        reasons.append("全局背景存在脊谷过渡带特征，仅作为弱约束参与调度。")

    if dom_landform:
        labels.append(f"dom_landform_{dom_landform}")
        reasons.append(f"DOM 范围主地貌为 {dom_landform}。")
    if dom_slope:
        labels.append(f"dom_slope_{dom_slope}")
    if dom_aspect:
        labels.append(f"dom_aspect_{dom_aspect}")
    if dom_position:
        labels.append(f"dom_slope_position_{dom_position}")
    if dom_mean_slope is not None and dom_mean_slope >= 25.0:
        labels.append("dom_steep_surface")
        reasons.append(f"DOM 范围平均坡度约 {dom_mean_slope:.1f}°，属于较陡坡面。")
    elif dom_mean_slope is not None and dom_mean_slope >= 12.0:
        labels.append("dom_moderate_slope_surface")
    if dom_aspect in {"north", "northeast", "northwest"}:
        labels.append("dom_shadow_aspect")
        reasons.append("DOM 范围以阴坡方向为主，应更关注阴影干扰。")
    if dom_landform in {"ridge_transition", "valley_transition"}:
        labels.append("dom_ridge_valley_transition")
        reasons.append("DOM 范围位于脊谷过渡带，局部形态变化可能影响分割稳定性。")
    if dom_position in {"upper", "shoulder"}:
        labels.append("dom_upper_slope")
    elif dom_position in {"lower", "foot"}:
        labels.append("dom_lower_slope")

    return {
        "global_background": global_background,
        "dom_context": dom_context,
        "xiaoban_context": xiaoban_summary,
        "labels": list(dict.fromkeys(labels)),
        "reasons": reasons,
        "policy": {
            "global_role": "weak_background_constraint",
            "dom_role": "primary_context_for_scheduler_and_child_model_routing",
            "roi_role": "inherit_dom_context_products",
        },
    }


def assess_input_bundle(
    cfg: dict[str, Any],
    input_manifest: dict[str, Any],
    terrain_info: dict[str, Any],
    data_processing_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    modalities = ((input_manifest.get("metadata") or {}).get("input_modalities") or {}).copy()
    issues: list[str] = []
    strengths: list[str] = []

    if modalities.get("image"):
        strengths.append("已接入高分辨率遥感影像。")
    else:
        issues.append("缺少遥感影像输入。")

    if modalities.get("dem"):
        strengths.append("已接入 DEM，可提取坡度、坡向、坡位和地貌先验。")
    else:
        issues.append("缺少 DEM，地形先验不足。")

    if modalities.get("inventory"):
        strengths.append("已接入样地调查表或行业约束数据，可用于结果质量评估。")
    else:
        issues.append("缺少调查或行业约束数据，结果评估约束不足。")

    if modalities.get("knowledge"):
        strengths.append("已接入领域知识数据，可支持先验嵌入和策略推理。")
    else:
        issues.append("缺少领域知识数据，策略推理上下文不足。")

    if modalities.get("public_datasets"):
        strengths.append("已接入公开数据集索引，可支持后续微调样本积累。")

    if terrain_info.get("terrain_generated"):
        strengths.append("DEM 派生地形产品将在本次运行中自动生成。")

    if not cfg.get("xiaoban_shp"):
        issues.append("缺少行业矢量边界数据，无法稳定执行 ROI 细化。")

    recommended_actions: list[str] = []
    if terrain_info.get("dem_tif") and not terrain_info.get("landform_tif"):
        recommended_actions.append("优先生成坡度、坡向、坡位和地貌栅格，增强 ROI 判定依据。")
    if not modalities.get("knowledge"):
        recommended_actions.append("建议补充领域知识或历史成功策略。")
    if not modalities.get("public_datasets"):
        recommended_actions.append("建议补充公开数据集来源，为后续微调池积累样本。")

    scene_analysis = _build_scene_analysis(cfg)
    image_texture_analysis = _build_image_texture_analysis(data_processing_summary)
    image_quality_analysis = _build_image_quality_analysis(data_processing_summary)
    terrain_analysis = _build_terrain_analysis(terrain_info)
    forest_type = scene_analysis.get("forest_type")
    stand_labels = ((scene_analysis.get("stand_condition") or {}).get("labels") or [])
    texture_labels = image_texture_analysis.get("labels") or []
    quality_labels = image_quality_analysis.get("labels") or []
    terrain_labels = terrain_analysis.get("labels") or []
    if forest_type:
        strengths.append(f"已识别当前森林类型倾向为 {forest_type}。")
    else:
        recommended_actions.append("建议补充 forest_type 或完善小班统计字段，以增强场景识别。")
    if stand_labels:
        strengths.append(f"已提取林分条件标签: {', '.join(str(label) for label in stand_labels)}。")
    if texture_labels:
        strengths.append(f"已提取影像纹理标签: {', '.join(str(label) for label in texture_labels)}。")
    if quality_labels:
        issues.append(f"已识别影像质量风险标签: {', '.join(str(label) for label in quality_labels)}。")
    if terrain_labels:
        strengths.append(f"已提取地形上下文标签: {', '.join(str(label) for label in terrain_labels[:6])}。")

    quality_levels = image_quality_analysis.get("levels") or {}
    if quality_levels.get("blur") in {"high", "severe"}:
        recommended_actions.append("建议在分割前增加锐度评估或去模糊/增强预处理，并提高滑窗重叠。")
    if quality_levels.get("exposure") in {"high", "severe"}:
        recommended_actions.append("建议增加亮度归一化，并针对过曝/欠曝区域进行曝光校正。")
    if quality_levels.get("shadow") in {"moderate", "heavy"}:
        recommended_actions.append("建议增加阴影抑制或地形阴影先验，避免阴影与树冠混淆。")
    if quality_levels.get("stripe_noise") in {"medium", "high"}:
        recommended_actions.append("建议增加条带去噪或方向性滤波，并降低单窗决策敏感性。")
    if quality_levels.get("color_cast") in {"medium", "high"}:
        recommended_actions.append("建议增加颜色归一化或白平衡校正，降低色偏影响。")

    scene_tags = list(scene_analysis.get("scene_tags") or [])
    for label in texture_labels:
        if label not in scene_tags:
            scene_tags.append(str(label))
    for label in quality_labels:
        if label not in scene_tags:
            scene_tags.append(str(label))
    for label in terrain_labels:
        if label not in scene_tags:
            scene_tags.append(str(label))
    scene_analysis["scene_tags"] = scene_tags
    scene_analysis["image_texture_analysis"] = image_texture_analysis
    scene_analysis["image_quality_analysis"] = image_quality_analysis
    scene_analysis["terrain_analysis"] = terrain_analysis
    readiness_score = max(0.2, 1.0 - 0.10 * len(issues)) if issues else 1.0

    payload = InputAssessment(
        readiness_score=readiness_score,
        modality_status=modalities,
        strengths=strengths,
        issues=issues,
        recommended_actions=recommended_actions,
        scene_analysis=scene_analysis,
        terrain_summary={
            "dem_tif": terrain_info.get("dem_tif"),
            "slope_tif": terrain_info.get("slope_tif"),
            "aspect_tif": terrain_info.get("aspect_tif"),
            "landform_tif": terrain_info.get("landform_tif"),
            "slope_position_tif": terrain_info.get("slope_position_tif"),
            "global_dem_tif": terrain_info.get("global_dem_tif"),
            "global_slope_tif": terrain_info.get("global_slope_tif"),
            "global_aspect_tif": terrain_info.get("global_aspect_tif"),
            "global_landform_tif": terrain_info.get("global_landform_tif"),
            "global_slope_position_tif": terrain_info.get("global_slope_position_tif"),
            "global_terrain_background": terrain_info.get("global_terrain_background") or {},
            "dom_terrain_context": terrain_info.get("dom_terrain_context") or {},
            "terrain_layer_policy": terrain_info.get("terrain_layer_policy") or {},
        },
        data_processing_summary=data_processing_summary or {},
    )
    return payload.to_dict()
