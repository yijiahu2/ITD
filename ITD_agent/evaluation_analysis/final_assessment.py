from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .benchmark_engine import evaluate_benchmark_vector_result
from .contracts import FinalAssessmentResult
from .online_quality_engine import evaluate_online_quality


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_report_cfg(runtime_cfg: dict[str, Any] | None) -> dict[str, Any]:
    evaluation_cfg = (runtime_cfg or {}).get("evaluation") or {}
    return evaluation_cfg.get("final_report") or {}


def _resolve_inst_shp(summary: dict[str, Any]) -> str | None:
    return (
        summary.get("tree_crowns_shp")
        or summary.get("merged_inst_shp")
        or (summary.get("segmentation_model") or {}).get("tree_crowns_shp")
        or (summary.get("segmentation_model") or {}).get("y_inst_shp")
    )


def _resolve_patch_raster(summary: dict[str, Any], runtime_cfg: dict[str, Any] | None = None) -> str | None:
    return (
        (runtime_cfg or {}).get("input_image")
        or (summary.get("run_meta") or {}).get("input_image")
        or (summary.get("input_layer") or {}).get("input_image")
    )


def _resolve_semantic_prior_tif(summary: dict[str, Any]) -> str | None:
    data_processing = summary.get("data_processing") or {}
    if data_processing.get("m_sem_tif"):
        return data_processing.get("m_sem_tif")
    semantic_prior = data_processing.get("semantic_prior") or {}
    if semantic_prior.get("m_sem_tif"):
        return semantic_prior.get("m_sem_tif")
    return (summary.get("output_aliases") or {}).get("m_sem_tif")


def _resolve_chm_tif(summary: dict[str, Any], runtime_cfg: dict[str, Any] | None = None) -> str | None:
    if (runtime_cfg or {}).get("chm_tif"):
        return (runtime_cfg or {}).get("chm_tif")
    input_layer = summary.get("input_layer") or {}
    for item in input_layer.get("canopy_height") or []:
        if isinstance(item, dict) and item.get("path"):
            return str(item.get("path"))
    return None


def _build_online_quality_result(
    summary: dict[str, Any],
    *,
    runtime_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inst_shp = _resolve_inst_shp(summary)
    if not inst_shp or not Path(inst_shp).exists():
        return {}
    return evaluate_online_quality(
        inst_shp=str(inst_shp),
        m_sem_tif=_resolve_semantic_prior_tif(summary),
        chm_tif=_resolve_chm_tif(summary, runtime_cfg=runtime_cfg),
        patch_raster=_resolve_patch_raster(summary, runtime_cfg=runtime_cfg),
    )


def _resolve_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and metrics:
        return metrics
    metrics_json = summary.get("metrics_json") or (summary.get("evaluation") or {}).get("metrics_json")
    if metrics_json and Path(metrics_json).exists():
        payload = _load_json(metrics_json)
        if isinstance(payload, dict):
            return payload
    return {}


def evaluate_reference_quality_result(
    summary: dict[str, Any],
    *,
    runtime_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = _resolve_metrics(summary)
    keys = [
        "pred_tree_count",
        "expected_tree_count",
        "tree_count_error_ratio",
        "tree_count_error_abs",
        "pred_mean_crown_width",
        "expected_mean_crown_width",
        "mean_crown_width_error_ratio",
        "mean_crown_width_error_abs",
        "pred_cover_ratio",
        "expected_closure",
        "closure_error_abs",
        "pred_density_trees_per_ha",
        "expected_density",
        "density_error_abs",
    ]
    selected = {key: metrics.get(key) for key in keys if key in metrics}
    result = {
        "evaluation_mode": "reference_quality",
        "metrics_source": summary.get("metrics_json") or (summary.get("evaluation") or {}).get("metrics_json"),
        "selected_metrics": selected,
    }
    online_quality = _build_online_quality_result(summary, runtime_cfg=runtime_cfg)
    if isinstance(online_quality, dict) and online_quality:
        result["online_quality_score"] = online_quality.get("quality_score")
        online_metrics = online_quality.get("metrics") or {}
        for source_key, target_key in [
            ("patch_context", "patch_context"),
            ("semantic_instance_consistency", "area_consistency"),
            ("geometry_plausibility", "geometry_diagnostics"),
            ("height_consistency", "height_diagnostics"),
        ]:
            payload = online_metrics.get(source_key)
            if isinstance(payload, dict) and payload:
                result[target_key] = payload
    return result


def evaluate_final_phase(
    summary: dict[str, Any],
    runtime_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report_cfg = _normalize_report_cfg(runtime_cfg)
    preferred_mode = str(report_cfg.get("preferred_mode", "auto")).strip().lower() or "auto"
    benchmark_cfg = report_cfg.get("benchmark") or {}
    gt_shp = (
        benchmark_cfg.get("gt_tree_crowns_shp")
        or benchmark_cfg.get("ground_truth_shp")
        or benchmark_cfg.get("gt_shp")
    )
    pred_shp = _resolve_inst_shp(summary)
    if gt_shp and pred_shp and preferred_mode in {"auto", "benchmark"} and Path(gt_shp).exists():
        payload = evaluate_benchmark_vector_result(
            pred_shp=str(pred_shp),
            gt_shp=str(gt_shp),
            score_field=benchmark_cfg.get("score_field"),
        )
        return FinalAssessmentResult(evaluation_mode="benchmark", payload=payload).to_dict()
    if preferred_mode == "benchmark":
        return FinalAssessmentResult(
            evaluation_mode="benchmark_unavailable",
            payload={
                "message": "已请求 benchmark 评估，但缺少可用的 ground-truth 树冠矢量数据，已无法计算 AP50/AP75/R2。"
            },
        ).to_dict()
    payload = evaluate_reference_quality_result(summary, runtime_cfg=runtime_cfg)
    return FinalAssessmentResult(evaluation_mode="reference_quality", payload=payload).to_dict()
