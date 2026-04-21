from __future__ import annotations

from typing import Any

from .contracts import ExpertModelAssessment
from .detail_ranker import summarize_details_csv
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
    return payload.to_dict()


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
    )
