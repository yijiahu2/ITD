from __future__ import annotations

from typing import Any

from .child_model_assessment import evaluate_child_model_assessment
from .final_assessment import evaluate_final_phase
from .finetune_effect_assessment import compare_finetune_effect
from .input_assessment import assess_input_bundle
from .main_model_assessment import evaluate_main_model_assessment
from .roi_assessment import build_roi_assessment, decide_roi_continuation


def evaluate_input_phase(
    cfg: dict[str, Any],
    *,
    input_manifest: dict[str, Any],
    terrain_info: dict[str, Any],
    data_processing_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return assess_input_bundle(
        cfg,
        input_manifest=input_manifest,
        terrain_info=terrain_info,
        data_processing_summary=data_processing_summary,
    )


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
    return assessment


def evaluate_child_model_phase(
    cfg: dict[str, Any],
    *,
    metrics: dict[str, Any],
    metrics_json: str,
    details_csv: str,
    round_idx: int,
    previous_score: float | None = None,
    terrain_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return evaluate_child_model_assessment(
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
    join_col: str = "xiaoban_id",
) -> dict[str, Any]:
    return compare_finetune_effect(
        before_csv=before_csv,
        after_csv=after_csv,
        out_dir=out_dir,
        join_col=join_col,
    )


__all__ = [
    "evaluate_input_phase",
    "evaluate_main_model_phase",
    "evaluate_roi_phase",
    "evaluate_child_model_phase",
    "evaluate_final_phase",
    "evaluate_finetune_effect_phase",
]
