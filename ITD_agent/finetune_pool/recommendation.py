from __future__ import annotations

from typing import Any

from ITD_agent.evaluation_analysis.detail_ranker import summarize_details_csv


def build_finetune_recommendation(
    cfg: dict[str, Any],
    *,
    metrics: dict[str, Any],
    details_csv: str | None,
    roi_round_count: int,
) -> dict[str, Any]:
    details_summary = summarize_details_csv(details_csv, top_k=5, cfg=cfg) if details_csv else {"top_k_reference_units": []}
    tree_ratio = float(metrics.get("tree_count_error_ratio") or 0.0)
    crown_ratio = float(metrics.get("mean_crown_width_error_ratio") or 0.0)
    closure_abs = float(metrics.get("closure_error_abs") or 0.0)
    density_abs = float(metrics.get("density_error_abs") or 0.0)

    target_module = "segmentation_model"
    if closure_abs >= 0.16 and tree_ratio < 0.18:
        target_module = "data_processing"

    planning_cfg = ((cfg.get("ITD_agent") or {}).get("planning") or {})
    roi_cfg = planning_cfg.get("roi_extraction") or planning_cfg.get("roi_refine") or {}
    should_recommend = (
        tree_ratio >= 0.22
        or crown_ratio >= 0.25
        or closure_abs >= 0.12
        or roi_round_count >= int(roi_cfg.get("max_rounds", 2))
    )
    trigger_mode = "defer_until_pool_threshold"
    if len(details_summary.get("top_k_reference_units") or []) >= 5 and should_recommend:
        trigger_mode = "ready_for_pool_accumulation"

    return {
        "should_recommend": should_recommend,
        "target_module": target_module,
        "trigger_mode": trigger_mode,
        "reason": "建议累计同类失败样本后再触发微调训练。" if should_recommend else "当前任务未达到微调建议阈值。",
        "failure_summary": {
            "tree_count_error_ratio": tree_ratio,
            "mean_crown_width_error_ratio": crown_ratio,
            "closure_error_abs": closure_abs,
            "density_error_abs": density_abs,
            "top_problem_cases": details_summary.get("top_k_reference_units") or [],
        },
    }
