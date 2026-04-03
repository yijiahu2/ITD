from __future__ import annotations

from typing import Any

from ITD_agent.data_processing.roi.extractor import extract_signal_driven_roi_candidates
from ITD_agent.llm_gateway import request_roi_decision
from ITD_agent.llm_gateway import request_roi_candidate_selection

from .contracts import ROIAssessment
from .detail_ranker import summarize_details_csv
from .reference_quality_engine import score_reference_metrics


def _normalize_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def _get_planning_block(cfg: dict[str, Any]) -> dict[str, Any]:
    return ((cfg.get("ITD_agent") or {}).get("planning") or {})


def get_roi_refine_block(cfg: dict[str, Any]) -> dict[str, Any]:
    planning_cfg = _get_planning_block(cfg)
    roi_cfg = planning_cfg.get("roi_extraction")
    if isinstance(roi_cfg, dict):
        return roi_cfg
    roi_cfg = planning_cfg.get("roi_refine")
    if isinstance(roi_cfg, dict):
        return roi_cfg
    if planning_cfg:
        return {"enabled": True}
    return {"enabled": False}


def build_roi_assessment(
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    details_csv: str,
    *,
    round_idx: int,
    previous_score: float | None = None,
    y_inst_tif: str | None = None,
    m_sem_tif: str | None = None,
    terrain_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    roi_cfg = get_roi_refine_block(cfg)
    enabled = _normalize_bool(roi_cfg.get("enabled", True))
    max_rounds = int(roi_cfg.get("max_rounds", 2))
    top_k = int(roi_cfg.get("top_k", 3))
    min_problem_cases = int(roi_cfg.get("min_problem_cases", 1))
    improvement_epsilon = float(roi_cfg.get("improvement_epsilon", 0.01))
    use_llm = _normalize_bool(roi_cfg.get("use_llm", True))

    tree_ratio = float(metrics.get("tree_count_error_ratio") or 0.0)
    crown_ratio = float(metrics.get("mean_crown_width_error_ratio") or 0.0)
    closure_abs = float(metrics.get("closure_error_abs") or 0.0)
    current_score = score_reference_metrics(metrics)
    details_summary = summarize_details_csv(details_csv, top_k=top_k, cfg=cfg)
    top_cases = details_summary.get("top_k_xiaoban") or []

    signal_roi_summary: dict[str, Any] = {}
    candidate_rois = []
    if y_inst_tif:
        signal_roi_summary = extract_signal_driven_roi_candidates(
            base_cfg=cfg,
            y_inst_tif=y_inst_tif,
            m_sem_tif=m_sem_tif,
            terrain_info=terrain_info or {},
            top_k=top_k * 2,
            round_idx=round_idx,
        )
        candidate_rois = list(signal_roi_summary.get("selected_candidates") or [])
        if candidate_rois and use_llm:
            scene_analysis = ((cfg.get("_input_assessment") or {}).get("scene_analysis") or {})
            llm_selection = request_roi_candidate_selection(
                candidate_rois=candidate_rois[: max(top_k * 2, top_k)],
                metrics=metrics,
                scene_analysis=scene_analysis,
                runtime_cfg=cfg,
                use_llm=use_llm,
            )
            llm_output = llm_selection.get("parsed_result") if isinstance(llm_selection, dict) else None
            if isinstance(llm_output, dict):
                selected_ids = [str(item) for item in (llm_output.get("selected_candidate_ids") or []) if str(item).strip()]
                if selected_ids:
                    selected_lookup = {item.get("candidate_id"): item for item in candidate_rois}
                    reordered = [selected_lookup[item_id] for item_id in selected_ids if item_id in selected_lookup]
                    remaining = [item for item in candidate_rois if item.get("candidate_id") not in set(selected_ids)]
                    candidate_rois = (reordered + remaining)[:top_k]
                signal_roi_summary["llm_selection"] = llm_selection
        candidate_rois = candidate_rois[:top_k]
    if not candidate_rois:
        candidate_rois = top_cases

    triggers: list[str] = []
    if tree_ratio >= float(roi_cfg.get("tree_count_error_ratio_thr", 0.18)):
        triggers.append("tree_count_error_ratio")
    if crown_ratio >= float(roi_cfg.get("mean_crown_width_error_ratio_thr", 0.22)):
        triggers.append("mean_crown_width_error_ratio")
    if closure_abs >= float(roi_cfg.get("closure_error_abs_thr", 0.10)):
        triggers.append("closure_error_abs")
    if len(candidate_rois) >= min_problem_cases:
        triggers.append("problem_roi_cases")

    improvement = None
    if current_score is not None and previous_score is not None:
        improvement = previous_score - current_score

    heuristic_continue = (
        enabled
        and round_idx < max_rounds
        and bool(triggers)
        and len(candidate_rois) >= min_problem_cases
        and (improvement is None or improvement >= -improvement_epsilon)
    )
    quality_label = "acceptable" if not heuristic_continue else "needs_roi_refinement"
    payload = ROIAssessment(
        assessment_phase="roi_assessment",
        round_idx=round_idx,
        quality_label=quality_label,
        current_score=current_score,
        previous_score=previous_score,
        improvement=improvement,
        trigger_metrics=triggers,
        details_summary=details_summary,
        candidate_rois=candidate_rois,
        continue_refinement=heuristic_continue,
        decision_source="heuristic",
        decision_reason="基于 ROI 质量阈值和问题区域数量进行判定。",
    )
    result = payload.to_dict()
    result.update(
        {
            "enabled": enabled,
            "max_rounds": max_rounds,
            "top_k": top_k,
            "min_problem_cases": min_problem_cases,
            "improvement_epsilon": improvement_epsilon,
            "heuristic_continue": heuristic_continue,
            "candidate_source": "signal_driven" if signal_roi_summary.get("selected_candidates") else "inventory_detail_fallback",
            "signal_roi_summary": signal_roi_summary,
        }
    )
    return result


def decide_roi_continuation(
    cfg: dict[str, Any],
    *,
    roi_assessment: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    roi_cfg = get_roi_refine_block(cfg)
    use_llm = _normalize_bool(roi_cfg.get("use_llm", True))
    heuristic_continue = bool(roi_assessment.get("heuristic_continue", False))
    decision = {
        "continue_refinement": heuristic_continue,
        "decision_source": "heuristic",
        "reason": "基于 ROI 质量阈值和问题区域数量进行判定。",
    }

    if not use_llm:
        return decision
    llm_response = request_roi_decision(
        roi_assessment=roi_assessment,
        metrics=metrics,
        runtime_cfg=cfg,
        use_llm=use_llm,
    )
    llm_output = llm_response.get("parsed_result") if isinstance(llm_response, dict) else None
    if isinstance(llm_output, dict) and "continue_refinement" in llm_output:
        decision = {
            "continue_refinement": bool(llm_output.get("continue_refinement")),
            "decision_source": "llm",
            "reason": str(llm_output.get("reason") or ""),
            "preferred_child_model": llm_output.get("preferred_child_model"),
            "llm_output": llm_output,
            "llm_gateway_result": llm_response,
        }
    return decision
