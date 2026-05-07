from __future__ import annotations

from typing import Any

from .contracts import ROIAssessment
from .detail_ranker import summarize_details_csv
from .flow_decisions import build_roi_flow_decision
from .reference_quality_engine import METRIC_CATEGORIES, build_reference_score_breakdown


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


def _candidate_priority_score(candidate: dict[str, Any]) -> float:
    score = float(candidate.get("score") or 0.0)
    prior_overlap = float(candidate.get("prior_overlap_ratio") or 0.0)
    boundary = float(candidate.get("boundary_score_mean") or 0.0)
    terrain = float(candidate.get("terrain_score_mean") or 0.0)
    return score + 0.08 * prior_overlap + 0.05 * boundary + 0.03 * terrain


def _resolve_metric_thresholds(
    cfg: dict[str, Any],
    *,
    metrics: dict[str, Any],
    round_idx: int,
) -> dict[str, float]:
    roi_cfg = get_roi_refine_block(cfg)
    tree_error = float(metrics.get("tree_count_error_ratio") or 0.0)
    defaults = {
        "tree_count_error_ratio": 0.18 if tree_error <= 0.12 else 0.10,
        "mean_crown_width_error_ratio": 0.12 if tree_error <= 0.12 else 0.18,
        "closure_error_abs": 0.06 if tree_error <= 0.12 else 0.10,
    }
    if round_idx > 0:
        defaults["mean_crown_width_error_ratio"] = max(defaults["mean_crown_width_error_ratio"] - 0.01, 0.08)
        defaults["closure_error_abs"] = max(defaults["closure_error_abs"] - 0.01, 0.04)

    resolved: dict[str, float] = {}
    for key, default_value in defaults.items():
        raw = roi_cfg.get(key)
        value = float(raw) if raw not in (None, "") else None
        resolved[key] = default_value if value is None or value <= 0 else value
    return resolved


def _prune_candidate_rois(candidate_rois: list[dict[str, Any]], roi_cfg: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    if len(candidate_rois) <= 1:
        return candidate_rois[: max(int(roi_cfg.get("signal_candidate_max_keep", max(top_k * 4, 8))), 0)]

    ratio_thr = float(roi_cfg.get("signal_candidate_score_ratio_thr", 0.82))
    gap_thr = float(roi_cfg.get("signal_candidate_score_gap_thr", 0.08))
    keep_min = max(1, int(roi_cfg.get("signal_candidate_keep_min", 1)))
    keep_max = max(keep_min, int(roi_cfg.get("signal_candidate_max_keep", max(top_k * 4, 8))))

    ordered = sorted(candidate_rois, key=_candidate_priority_score, reverse=True)
    best_priority = _candidate_priority_score(ordered[0])
    absolute_floor = best_priority - gap_thr
    relative_floor = best_priority * ratio_thr

    kept: list[dict[str, Any]] = []
    for item in ordered:
        priority = _candidate_priority_score(item)
        if len(kept) < keep_min or priority >= absolute_floor or priority >= relative_floor:
            kept.append(item)
        if len(kept) >= keep_max:
            break

    return kept[:keep_max]


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
    candidate_rois: list[dict[str, Any]] | None = None,
    signal_roi_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    roi_cfg = get_roi_refine_block(cfg)
    enabled = _normalize_bool(roi_cfg.get("enabled", True))
    max_rounds = int(roi_cfg.get("max_rounds", 2))
    top_k = int(roi_cfg.get("top_k", 3))
    min_problem_cases = int(roi_cfg.get("min_problem_cases", 1))
    improvement_epsilon = float(roi_cfg.get("improvement_epsilon", 0.01))
    metric_thresholds = _resolve_metric_thresholds(cfg, metrics=metrics, round_idx=round_idx)

    tree_ratio = float(metrics.get("tree_count_error_ratio") or 0.0)
    crown_ratio = float(metrics.get("mean_crown_width_error_ratio") or 0.0)
    closure_abs = float(metrics.get("closure_error_abs") or 0.0)
    score_breakdown = build_reference_score_breakdown(metrics, cfg=cfg)
    current_score = score_breakdown.get("score")
    details_summary = summarize_details_csv(details_csv, top_k=top_k, cfg=cfg)
    top_cases = details_summary.get("top_k_reference_units") or []

    signal_summary = dict(signal_roi_summary or {})
    candidate_source = "inventory_detail_fallback"
    resolved_candidates = list(candidate_rois or [])
    if resolved_candidates:
        pruned_candidates = _prune_candidate_rois(resolved_candidates, roi_cfg, top_k)
        signal_summary["pruned_candidate_ids"] = [str(item.get("candidate_id") or "") for item in pruned_candidates]
        signal_summary["pruned_candidate_count"] = len(pruned_candidates)
        resolved_candidates = pruned_candidates
        candidate_source = "precomputed"
    else:
        resolved_candidates = top_cases

    if y_inst_tif or m_sem_tif or terrain_info:
        signal_summary["raw_input_ignored_by_evaluation"] = True
        signal_summary["boundary_note"] = "ROI extraction belongs to data_processing; evaluation_analysis only scores provided facts."

    triggers: list[str] = []
    trigger_details: dict[str, Any] = {}
    if tree_ratio >= metric_thresholds["tree_count_error_ratio"]:
        triggers.append("tree_count_error_ratio")
        trigger_details["tree_count_error_ratio"] = {
            **METRIC_CATEGORIES["tree_count_error_ratio"],
            "value": tree_ratio,
            "threshold": metric_thresholds["tree_count_error_ratio"],
        }
    if crown_ratio >= metric_thresholds["mean_crown_width_error_ratio"]:
        triggers.append("mean_crown_width_error_ratio")
        trigger_details["mean_crown_width_error_ratio"] = {
            **METRIC_CATEGORIES["mean_crown_width_error_ratio"],
            "value": crown_ratio,
            "threshold": metric_thresholds["mean_crown_width_error_ratio"],
        }
    if closure_abs >= metric_thresholds["closure_error_abs"]:
        triggers.append("closure_error_abs")
        trigger_details["closure_error_abs"] = {
            **METRIC_CATEGORIES["closure_error_abs"],
            "value": closure_abs,
            "threshold": metric_thresholds["closure_error_abs"],
        }
    if len(resolved_candidates) >= 1:
        triggers.append("problem_roi_cases")
        trigger_details["problem_roi_cases"] = {
            "category": "roi_candidate_availability",
            "label": "Problem ROI cases",
            "direction": "more_cases_require_refinement",
            "value": len(resolved_candidates),
            "threshold": min_problem_cases,
        }

    improvement = None
    if current_score is not None and previous_score is not None:
        improvement = previous_score - current_score

    heuristic_continue = (
        enabled
        and round_idx < max_rounds
        and bool(triggers)
        and len(resolved_candidates) >= 1
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
        candidate_rois=resolved_candidates,
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
            "metric_thresholds": metric_thresholds,
            "trigger_details": trigger_details,
            "score_breakdown": score_breakdown,
            "candidate_source": candidate_source,
            "signal_roi_summary": signal_summary,
        }
    )
    result["flow_decision"] = build_roi_flow_decision(result)
    return result


def decide_roi_continuation(
    cfg: dict[str, Any],
    *,
    roi_assessment: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    heuristic_continue = bool(roi_assessment.get("heuristic_continue", False))
    return {
        "continue_refinement": heuristic_continue,
        "decision_source": "heuristic",
        "reason": "基于 ROI 质量阈值和问题区域数量进行判定。",
        "decision_guard_reason": "evaluation_analysis 不发起 LLM 决策；如需策略调度，由 planning/scheduler 或 orchestration 负责。",
    }
