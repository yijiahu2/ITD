from __future__ import annotations

from typing import Any

from .contracts import ExpertModelAssessment
from .decision_flags import build_decision_flags
from .detail_ranker import summarize_details_csv
from .flow_decisions import build_expert_acceptance_flow_decision
from .reference_quality_engine import score_reference_metrics
from .roi_assessment import build_roi_assessment


def evaluate_expert_model_assessment(
    cfg: dict[str, Any],
    *,
    metrics: dict[str, Any],
    metrics_json: str,
    details_csv: str,
    round_idx: int,
    previous_score: float | None = None,
    y_inst_tif: str | None = None,
    m_sem_tif: str | None = None,
    terrain_info: dict[str, Any] | None = None,
    candidate_rois: list[dict[str, Any]] | None = None,
    signal_roi_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_score = score_reference_metrics(metrics, cfg=cfg)
    roi_assessment = build_roi_assessment(
        cfg,
        metrics,
        details_csv,
        round_idx=round_idx,
        previous_score=previous_score,
        y_inst_tif=y_inst_tif,
        m_sem_tif=m_sem_tif,
        terrain_info=terrain_info,
        candidate_rois=candidate_rois,
        signal_roi_summary=signal_roi_summary,
    )
    payload = ExpertModelAssessment(
        assessment_phase="expert_model",
        round_idx=round_idx,
        metrics_json=metrics_json,
        details_csv=details_csv,
        metrics=metrics,
        detail_summary=summarize_details_csv(details_csv, top_k=5, cfg=cfg),
        current_score=current_score,
        previous_score=previous_score,
        improvement=roi_assessment.get("improvement"),
        roi_assessment=roi_assessment,
    )
    result = payload.to_dict()
    previous_overall_score = None if previous_score is None else max(0.0, min(1.0, 1.0 - float(previous_score)))
    current_result = {"selected_metrics": metrics}
    result["decision_flags"] = build_decision_flags(current_result, runtime_cfg=cfg, previous_overall_score=previous_overall_score)
    result["flow_decision"] = build_expert_acceptance_flow_decision(result)
    return result


def evaluate_child_model_assessment(
    cfg: dict[str, Any],
    *,
    metrics: dict[str, Any],
    metrics_json: str,
    details_csv: str,
    round_idx: int,
    previous_score: float | None = None,
    y_inst_tif: str | None = None,
    m_sem_tif: str | None = None,
    terrain_info: dict[str, Any] | None = None,
    candidate_rois: list[dict[str, Any]] | None = None,
    signal_roi_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return evaluate_expert_model_assessment(
        cfg,
        metrics=metrics,
        metrics_json=metrics_json,
        details_csv=details_csv,
        round_idx=round_idx,
        previous_score=previous_score,
        y_inst_tif=y_inst_tif,
        m_sem_tif=m_sem_tif,
        terrain_info=terrain_info,
        candidate_rois=candidate_rois,
        signal_roi_summary=signal_roi_summary,
    )
