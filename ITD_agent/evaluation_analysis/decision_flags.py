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

    reference_quality_score = result.get("reference_quality_score")
    if reference_quality_score is not None:
        reference_score = _clamp01(float(reference_quality_score))
    else:
        reference_error_score = result.get("reference_error_score")
        if reference_error_score is not None:
            reference_score = 1.0 - _clamp01(float(reference_error_score))
        else:
            reference_score = None

    selected = result.get("selected_metrics") or {}
    online_quality = result.get("online_quality") or {}
    if reference_score is None:
        error_terms: list[float] = []
        for key in ["tree_count_error_ratio", "mean_crown_width_error_ratio", "closure_error_abs"]:
            if key in selected and selected.get(key) is not None:
                error_terms.append(_clamp01(float(selected.get(key))))
        if "density_error_abs" in selected:
            expected_density = float(selected.get("expected_density") or 0.0)
            density_abs = float(selected.get("density_error_abs") or 0.0)
            if expected_density > 0:
                density_ratio = density_abs / expected_density
                error_terms.append(_clamp01(density_ratio))
        reference_score = 1.0 - _clamp01(sum(error_terms) / max(len(error_terms), 1)) if error_terms else None

    online_quality_score = None
    if online_quality.get("online_risk_score") is not None or result.get("online_risk_score") is not None:
        online_quality_score = _clamp01(
            float(
                online_quality.get("quality_score")
                if online_quality.get("quality_score") is not None
                else result.get("quality_score")
            )
        )
    else:
        online_score = online_quality.get("quality_score")
        if online_score is not None:
            online_quality_score = 1.0 - _clamp01(float(online_score))
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
    empty_output_flag = bool(geometry_diagnostics.get("empty_output_flag", False))
    failure_severity = error_decomposition.get("failure_severity")
    failure_pattern_confidence = error_decomposition.get("failure_pattern_confidence")
    semantic_gap = geometry_diagnostics.get("semantic_coverage_gap")
    fragmentation_score = geometry_diagnostics.get("fragmentation_score")
    merge_blob_score = geometry_diagnostics.get("merge_blob_score")
    edge_artifact_score = geometry_diagnostics.get("edge_artifact_score")
    param_issue_threshold = float(cfg.get("param_issue_threshold", 0.30))
    refinement_attempted = bool(
        result.get("roi_round_count", 0) > 0
        or result.get("param_search_exhausted", False)
        or result.get("refinement_attempted", False)
        or result.get("assessment_phase") in {"expert_model", "child_model"}
        or result.get("evaluation_mode") == "benchmark"
    )
    actionable_param_issue = bool(
        result.get("evaluation_mode") == "benchmark"
        or bool(error_decomposition)
        or semantic_instance_conflict_flag
        or empty_output_flag
        or (semantic_gap is not None and float(semantic_gap) >= param_issue_threshold)
        or (fragmentation_score is not None and float(fragmentation_score) >= param_issue_threshold)
        or (merge_blob_score is not None and float(merge_blob_score) >= param_issue_threshold)
        or (edge_artifact_score is not None and float(edge_artifact_score) >= param_issue_threshold)
    )

    need_local_refine_flag = bool(
        result.get("continue_refinement", False)
        or len(result.get("candidate_rois") or []) > 0
        or result.get("candidate_roi_count", 0) > 0
    )
    need_param_search_flag = bool(
        overall_score is not None
        and overall_score < param_search_threshold
        and not quality_pass_flag
        and actionable_param_issue
    )
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
        and (refinement_attempted or not actionable_param_issue)
    )
    need_manual_review_flag = bool(
        semantic_instance_conflict_flag
        or empty_output_flag
        or (failure_severity is not None and float(failure_severity) > (1.0 - manual_review_confidence_threshold))
        or (failure_pattern_confidence is not None and float(failure_pattern_confidence) < manual_review_confidence_threshold)
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
