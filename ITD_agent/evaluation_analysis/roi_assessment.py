from __future__ import annotations

from typing import Any

from ITD_agent.data_processing.roi.extractor import extract_signal_driven_roi_candidates
from ITD_agent.llm_gateway import request_roi_decision
from ITD_agent.llm_gateway import request_roi_candidate_selection

from .contracts import ROIAssessment
from .detail_ranker import summarize_details_csv
from .reference_quality_engine import build_reference_score_breakdown, score_reference_metrics


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


def _should_request_llm_roi_selection(
    candidate_rois: list[dict[str, Any]],
    roi_cfg: dict[str, Any],
    top_k: int,
) -> tuple[bool, str]:
    if len(candidate_rois) <= 1:
        return False, "候选 ROI 不超过 1 个，跳过 LLM 排序。"

    ordered = sorted(candidate_rois, key=_candidate_priority_score, reverse=True)
    best_priority = _candidate_priority_score(ordered[0])
    second_priority = _candidate_priority_score(ordered[1]) if len(ordered) > 1 else None
    if second_priority is None:
        return False, "仅存在单个有效候选 ROI，跳过 LLM 排序。"

    gap = best_priority - second_priority
    ratio = best_priority / max(second_priority, 1.0e-6)
    clear_gap_thr = float(roi_cfg.get("llm_selection_clear_gap_thr", 0.10))
    clear_ratio_thr = float(roi_cfg.get("llm_selection_clear_ratio_thr", 1.12))
    if gap >= clear_gap_thr and ratio >= clear_ratio_thr:
        return False, (
            f"候选 ROI 优先级已明显拉开，top1 较 top2 高 {gap:.3f}，"
            f"倍率 {ratio:.2f}，直接采用规则排序。"
        )
    if len(ordered) <= max(top_k, 2):
        return True, "候选 ROI 数量较少但排序接近，允许 LLM 做细粒度排序。"
    return True, "候选 ROI 排序接近，允许 LLM 在已生成候选内重排。"


def _should_request_llm_roi_decision(
    roi_assessment: dict[str, Any],
    roi_cfg: dict[str, Any],
) -> tuple[bool, str]:
    heuristic_continue = bool(roi_assessment.get("heuristic_continue", False))
    if heuristic_continue:
        return True, "启发式判定仍需继续细化，允许 LLM 做补充判断。"

    round_idx = int(roi_assessment.get("round_idx") or 0)
    max_rounds = int(roi_assessment.get("max_rounds") or roi_cfg.get("max_rounds") or 0)
    candidate_count = len(roi_assessment.get("candidate_rois") or [])
    min_problem_cases = int(roi_assessment.get("min_problem_cases") or roi_cfg.get("min_problem_cases") or 1)
    triggers = roi_assessment.get("trigger_metrics") or []
    improvement = roi_assessment.get("improvement")
    improvement_epsilon = float(roi_assessment.get("improvement_epsilon") or roi_cfg.get("improvement_epsilon") or 0.01)

    if round_idx >= max_rounds:
        return False, f"已达到最大 ROI 轮次 {max_rounds}，直接停止细化。"
    if not triggers:
        return False, "当前没有命中任何 ROI 触发指标，直接停止细化。"
    if candidate_count < 1:
        return False, "当前未检测到有效问题 ROI，直接停止细化。"
    if improvement is not None and float(improvement) < -improvement_epsilon:
        return False, f"最近一轮质量下降 {float(improvement):.4f}，超过允许回撤，直接停止细化。"
    return True, "虽然启发式未建议继续，但仍存在边界不确定性，允许 LLM 复核。"


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
    metric_thresholds = _resolve_metric_thresholds(cfg, metrics=metrics, round_idx=round_idx)

    tree_ratio = float(metrics.get("tree_count_error_ratio") or 0.0)
    crown_ratio = float(metrics.get("mean_crown_width_error_ratio") or 0.0)
    closure_abs = float(metrics.get("closure_error_abs") or 0.0)
    score_breakdown = build_reference_score_breakdown(metrics, cfg=cfg)
    current_score = score_breakdown.get("score")
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
        llm_selection_allowed, llm_selection_reason = _should_request_llm_roi_selection(candidate_rois, roi_cfg, top_k)
        signal_roi_summary["llm_selection_guard_reason"] = llm_selection_reason
        if candidate_rois and use_llm and llm_selection_allowed:
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
                    candidate_rois = reordered + remaining
                signal_roi_summary["llm_selection"] = llm_selection
        elif candidate_rois and use_llm:
            signal_roi_summary["llm_selection_skipped"] = True
        pruned_candidates = _prune_candidate_rois(candidate_rois, roi_cfg, top_k)
        signal_roi_summary["pruned_candidate_ids"] = [str(item.get("candidate_id") or "") for item in pruned_candidates]
        signal_roi_summary["pruned_candidate_count"] = len(pruned_candidates)
        candidate_rois = pruned_candidates
    if not candidate_rois:
        candidate_rois = top_cases

    triggers: list[str] = []
    if tree_ratio >= metric_thresholds["tree_count_error_ratio"]:
        triggers.append("tree_count_error_ratio")
    if crown_ratio >= metric_thresholds["mean_crown_width_error_ratio"]:
        triggers.append("mean_crown_width_error_ratio")
    if closure_abs >= metric_thresholds["closure_error_abs"]:
        triggers.append("closure_error_abs")
    if len(candidate_rois) >= 1:
        triggers.append("problem_roi_cases")

    improvement = None
    if current_score is not None and previous_score is not None:
        improvement = previous_score - current_score

    heuristic_continue = (
        enabled
        and round_idx < max_rounds
        and bool(triggers)
        and len(candidate_rois) >= 1
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
            "metric_thresholds": metric_thresholds,
            "score_breakdown": score_breakdown,
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
    llm_allowed, llm_guard_reason = _should_request_llm_roi_decision(roi_assessment, roi_cfg)
    if not llm_allowed:
        decision["decision_guard_reason"] = llm_guard_reason
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
            "preferred_expert_model": llm_output.get("preferred_expert_model") or llm_output.get("preferred_child_model"),
            "llm_output": llm_output,
            "llm_gateway_result": llm_response,
            "decision_guard_reason": llm_guard_reason,
        }
    else:
        decision["decision_guard_reason"] = llm_guard_reason
    return decision
