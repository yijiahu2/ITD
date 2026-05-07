from __future__ import annotations

from typing import Any


def _clamp01(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def _get_flag_cfg(runtime_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    evaluation_cfg = (runtime_cfg or {}).get("evaluation") or {}
    return evaluation_cfg.get("decision_flags") or {}


def _normalize_quality_score(result: dict[str, Any]) -> float | None:
    if result.get("evaluation_mode") == "benchmark":
        ap50 = result.get("ap50")
        ap75 = result.get("ap75")
        f1_score50 = result.get("f1_score50")
        if ap50 is None or ap75 is None or f1_score50 is None:
            return None
        return _clamp01((float(ap50) * 0.40) + (float(ap75) * 0.35) + (float(f1_score50) * 0.25))

    selected = result.get("selected_metrics") or {}
    online_quality = result.get("online_quality") or {}
    error_terms = [
        float(selected.get("tree_count_error_ratio") or 0.0),
        float(selected.get("mean_crown_width_error_ratio") or 0.0),
        float(selected.get("closure_error_abs") or 0.0),
    ]
    if "density_error_abs" in selected:
        expected_density = float(selected.get("expected_density") or 0.0)
        density_abs = float(selected.get("density_error_abs") or 0.0)
        density_ratio = density_abs / max(expected_density, 1000.0)
        error_terms.append(density_ratio)
    reference_score = 1.0 - _clamp01(sum(error_terms) / max(len(error_terms), 1))
    online_score = online_quality.get("quality_score")
    online_quality_score = None if online_score is None else 1.0 - _clamp01(float(online_score))
    if online_quality_score is None and not selected:
        return None
    if online_quality_score is None:
        return reference_score
    if not selected:
        return online_quality_score
    return _clamp01((reference_score * 0.65) + (online_quality_score * 0.35))


def build_decision_flags(
    result: dict[str, Any],
    *,
    runtime_cfg: dict[str, Any] | None = None,
    previous_overall_score: float | None = None,
) -> dict[str, Any]:
    cfg = _get_flag_cfg(runtime_cfg)
    pass_threshold = float(cfg.get("pass_threshold", 0.72))
    param_search_threshold = float(cfg.get("param_search_threshold", 0.55))
    finetune_threshold = float(cfg.get("finetune_threshold", 0.45))
    manual_review_confidence_threshold = float(cfg.get("manual_review_confidence_threshold", 0.35))
    accepted_gain_threshold = float(cfg.get("accepted_gain_threshold", 0.03))
    regression_threshold = float(cfg.get("regression_threshold", 0.03))

    overall_score = _normalize_quality_score(result)
    quality_pass_flag = bool(overall_score is not None and overall_score >= pass_threshold)
    error_decomposition = result.get("error_decomposition") or {}
    online_quality = result.get("online_quality") or {}
    geometry_diagnostics = (online_quality.get("metrics") or {}).get("geometry_diagnostics") or {}
    semantic_instance_conflict_flag = bool(geometry_diagnostics.get("semantic_instance_conflict_flag", False))
    failure_confidence = error_decomposition.get("failure_confidence")

    need_local_refine_flag = bool(
        result.get("continue_refinement", False)
        or len(result.get("candidate_rois") or []) > 0
        or result.get("candidate_roi_count", 0) > 0
    )
    need_param_search_flag = bool(overall_score is not None and overall_score < param_search_threshold and not quality_pass_flag)
    regression_flag = bool(
        overall_score is not None
        and previous_overall_score is not None
        and overall_score < (previous_overall_score - regression_threshold)
    )
    need_finetune_flag = bool(
        overall_score is not None
        and overall_score < finetune_threshold
        and not quality_pass_flag
        and not regression_flag
    )
    need_manual_review_flag = bool(
        semantic_instance_conflict_flag
        or (failure_confidence is not None and float(failure_confidence) < manual_review_confidence_threshold)
    )
    accepted_improvement_flag = bool(
        overall_score is not None
        and previous_overall_score is not None
        and overall_score >= (previous_overall_score + accepted_gain_threshold)
        and not regression_flag
    )

    return {
        "overall_score": overall_score,
        "quality_pass_flag": quality_pass_flag,
        "need_local_refine_flag": need_local_refine_flag,
        "need_param_search_flag": need_param_search_flag,
        "need_finetune_flag": need_finetune_flag,
        "need_manual_review_flag": need_manual_review_flag,
        "accepted_improvement_flag": accepted_improvement_flag,
        "regression_flag": regression_flag,
    }
