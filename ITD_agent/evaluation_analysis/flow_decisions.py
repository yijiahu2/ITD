from __future__ import annotations

from typing import Any


def _pick(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: source.get(key) for key in keys if key in source}


def build_flow_decision(
    *,
    decision_stage: str,
    decision_question: str,
    core_metrics: dict[str, Any],
    evidence_metrics: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "decision_stage": decision_stage,
        "decision_question": decision_question,
        "core_metrics": dict(core_metrics),
        "evidence_metrics": dict(evidence_metrics or {}),
        "decision": dict(decision or {}),
    }


def build_main_model_flow_decision(assessment: dict[str, Any]) -> dict[str, Any]:
    metrics = assessment.get("metrics") or {}
    online_quality = assessment.get("online_quality") or {}
    decision_flags = assessment.get("decision_flags") or {}
    return build_flow_decision(
        decision_stage="main_model_assessment",
        decision_question="主模型第一版结果是否可接受，是否需要进入 ROI？",
        core_metrics={
            "quality_score": assessment.get("quality_score"),
            "reference_error_score": assessment.get("quality_score"),
            "online_quality_score": online_quality.get("quality_score"),
            **_pick(
                decision_flags,
                [
                    "overall_score",
                    "quality_pass_flag",
                    "need_local_refine_flag",
                    "need_param_search_flag",
                    "need_finetune_flag",
                    "need_manual_review_flag",
                ],
            ),
            **_pick(
                metrics,
                [
                    "tree_count_error_ratio",
                    "mean_crown_width_error_ratio",
                    "closure_error_abs",
                    "density_error_abs",
                ],
            ),
        },
        evidence_metrics={
            "selected_metrics": _pick(
                metrics,
                [
                    "pred_tree_count",
                    "expected_tree_count",
                    "pred_mean_crown_width",
                    "expected_mean_crown_width",
                    "pred_cover_ratio",
                    "expected_closure",
                    "pred_density_trees_per_ha",
                    "expected_density",
                ],
            ),
            "online_quality": online_quality,
            "decision_flags": decision_flags,
            "detail_summary": assessment.get("detail_summary") or {},
            "score_breakdown": assessment.get("score_breakdown") or {},
        },
    )


def build_roi_flow_decision(assessment: dict[str, Any]) -> dict[str, Any]:
    return build_flow_decision(
        decision_stage="roi_refinement_decision",
        decision_question="是否进入或继续 ROI 局部细化？",
        core_metrics={
            "continue_refinement": bool(assessment.get("continue_refinement", False)),
            "heuristic_continue": bool(assessment.get("heuristic_continue", False)),
            "trigger_metrics": list(assessment.get("trigger_metrics") or []),
            "candidate_roi_count": len(assessment.get("candidate_rois") or []),
            "current_score": assessment.get("current_score"),
            "previous_score": assessment.get("previous_score"),
            "improvement": assessment.get("improvement"),
        },
        evidence_metrics={
            "metric_thresholds": assessment.get("metric_thresholds") or {},
            "trigger_details": assessment.get("trigger_details") or {},
            "candidate_rois": assessment.get("candidate_rois") or [],
            "details_summary": assessment.get("details_summary") or {},
            "signal_roi_summary": assessment.get("signal_roi_summary") or {},
            "score_breakdown": assessment.get("score_breakdown") or {},
        },
        decision=assessment.get("decision") or {
            "decision_source": assessment.get("decision_source"),
            "reason": assessment.get("decision_reason"),
        },
    )


def build_expert_acceptance_flow_decision(assessment: dict[str, Any]) -> dict[str, Any]:
    metrics = assessment.get("metrics") or {}
    decision_flags = assessment.get("decision_flags") or {}
    return build_flow_decision(
        decision_stage="expert_model_acceptance",
        decision_question="专家模型或局部细化结果是否优于旧结果？",
        core_metrics={
            "current_score": assessment.get("current_score"),
            "previous_score": assessment.get("previous_score"),
            "improvement": assessment.get("improvement"),
            **_pick(decision_flags, ["accepted_improvement_flag", "regression_flag", "overall_score"]),
            **_pick(
                metrics,
                [
                    "tree_count_error_ratio",
                    "mean_crown_width_error_ratio",
                    "closure_error_abs",
                    "density_error_abs",
                ],
            ),
        },
        evidence_metrics={
            "metrics": metrics,
            "decision_flags": decision_flags,
            "details_summary": assessment.get("detail_summary") or {},
            "roi_assessment": assessment.get("roi_assessment") or {},
        },
    )


def build_final_reference_flow_decision(result: dict[str, Any]) -> dict[str, Any]:
    selected = result.get("selected_metrics") or {}
    decision_flags = result.get("decision_flags") or {}
    return build_flow_decision(
        decision_stage="final_result_assessment",
        decision_question="最终结果质量如何？",
        core_metrics={
            **_pick(
                decision_flags,
                [
                    "overall_score",
                    "quality_pass_flag",
                    "need_local_refine_flag",
                    "need_param_search_flag",
                    "need_finetune_flag",
                    "need_manual_review_flag",
                ],
            ),
            **_pick(
                selected,
                [
                    "tree_count_error_ratio",
                    "mean_crown_width_error_ratio",
                    "closure_error_abs",
                    "density_error_abs",
                ],
            ),
            "online_quality_score": (result.get("online_quality") or {}).get("quality_score"),
        },
        evidence_metrics={
            "selected_metrics": selected,
            "online_quality": result.get("online_quality") or {},
            "decision_flags": decision_flags,
            "metrics_source": result.get("metrics_source"),
        },
    )


def build_final_benchmark_flow_decision(result: dict[str, Any]) -> dict[str, Any]:
    decision_flags = result.get("decision_flags") or {}
    return build_flow_decision(
        decision_stage="final_benchmark_assessment",
        decision_question="有 GT/COCO 时标准分割质量如何？",
        core_metrics={
            **_pick(
                decision_flags,
                [
                    "overall_score",
                    "quality_pass_flag",
                    "need_param_search_flag",
                    "need_finetune_flag",
                    "need_manual_review_flag",
                ],
            ),
            **_pick(
                result,
                [
                    "precision",
                    "recall",
                    "ap50",
                    "ap75",
                    "f1_score50",
                    "mean_iou_matched",
                    "mae",
                    "rmse",
                    "rmse_percent",
                    "r2",
                ],
            ),
        },
        evidence_metrics=_pick(
            result,
            [
                "tp50",
                "fp50",
                "fn50",
                "tp75",
                "fp75",
                "fn75",
                "iou_0_75",
                "crown_area_iou_0_50",
                "crown_area_iou_0_75",
                "prediction_file",
                "ground_truth_file",
                "score_field",
                "score_source",
                "error_decomposition",
            ],
        ),
    )


def build_finetune_effect_flow_decision(summary: dict[str, Any]) -> dict[str, Any]:
    return build_flow_decision(
        decision_stage="finetune_effect_assessment",
        decision_question="微调后是否真的提升？",
        core_metrics=_pick(
            summary,
            [
                "mean_gain_tree_count",
                "mean_gain_crown",
                "mean_gain_closure",
                "mean_gain_density",
                "num_tree_improved",
                "num_crown_improved",
                "num_closure_improved",
                "num_density_improved",
                "accepted_improvement_flag",
                "regression_flag",
            ],
        ),
        evidence_metrics={
            "stratified_gain": summary.get("stratified_gain") or [],
            "benchmark_gain": summary.get("benchmark_gain") or {},
        },
    )
