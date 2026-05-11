from __future__ import annotations

from typing import Any

from .expert_model_assessment import evaluate_expert_model_assessment
from .final_assessment import evaluate_final_phase
from .finetune_effect_assessment import compare_finetune_effect
from .flow_decisions import build_roi_flow_decision
from .main_model_assessment import evaluate_main_model_assessment
from .roi_assessment import build_roi_assessment, decide_roi_continuation


def evaluate_main_model_phase(
    cfg: dict[str, Any],
    *,
    inst_shp: str,
    terrain_info: dict[str, Any],
    metrics_json: str | None = None,
    details_csv: str | None = None,
    command_runner=None,
) -> dict[str, Any]:
    return evaluate_main_model_assessment(
        cfg,
        inst_shp=inst_shp,
        terrain_info=terrain_info,
        metrics_json=metrics_json,
        details_csv=details_csv,
        command_runner=command_runner,
    )


def evaluate_roi_phase(
    cfg: dict[str, Any],
    *,
    metrics: dict[str, Any],
    details_csv: str,
    round_idx: int,
    previous_score: float | None = None,
    y_inst_tif: str | None = None,
    m_sem_tif: str | None = None,
    terrain_info: dict[str, Any] | None = None,
    candidate_rois: list[dict[str, Any]] | None = None,
    signal_roi_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assessment = build_roi_assessment(
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
    decision = decide_roi_continuation(
        cfg,
        roi_assessment=assessment,
        metrics=metrics,
    )
    assessment["decision"] = decision
    assessment["continue_refinement"] = bool(decision.get("continue_refinement", False))
    assessment["decision_source"] = str(decision.get("decision_source") or assessment.get("decision_source") or "heuristic")
    assessment["decision_reason"] = str(decision.get("reason") or assessment.get("decision_reason") or "")
    assessment["flow_decision"] = build_roi_flow_decision(assessment)
    return assessment


def evaluate_expert_model_phase(
    cfg: dict[str, Any],
    *,
    metrics: dict[str, Any],
    metrics_json: str,
    details_csv: str,
    round_idx: int,
    previous_score: float | None = None,
    terrain_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return evaluate_expert_model_assessment(
        cfg,
        metrics=metrics,
        metrics_json=metrics_json,
        details_csv=details_csv,
        round_idx=round_idx,
        previous_score=previous_score,
        terrain_info=terrain_info,
    )


def evaluate_finetune_effect_phase(
    *,
    before_csv: str,
    after_csv: str,
    out_dir: str,
    join_col: str = "reference_unit_id",
) -> dict[str, Any]:
    return compare_finetune_effect(
        before_csv=before_csv,
        after_csv=after_csv,
        out_dir=out_dir,
        join_col=join_col,
    )


__all__ = [
    "evaluate_main_model_phase",
    "evaluate_roi_phase",
    "evaluate_expert_model_phase",
    "evaluate_final_phase",
    "evaluate_finetune_effect_phase",
]
