from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from ITD_agent.common.config_refs import reference_vector_path
from ITD_agent.common.values import safe_float as _safe_float
from ITD_agent.config_adapter import load_raw_yaml, load_runtime_config, save_runtime_config
from ITD_agent.data_processing.fusion import rasterize_instances_to_label_raster
from ITD_agent.data_processing.fusion.postprocess import fuse_instance_layers
from ITD_agent.data_processing.processor import summarize_data_processing_stage
from ITD_agent.data_processing.roi import extract_signal_driven_roi_candidates
from ITD_agent.evaluation_analysis.evaluator import (
    evaluate_expert_model_phase,
    evaluate_main_model_phase,
    evaluate_roi_phase,
)
from ITD_agent.finetune_pool.recommendation import build_finetune_recommendation as build_finetune_recommendation_impl
from ITD_agent.evaluation_analysis.reference_quality_engine import score_reference_metrics
from ITD_agent.orchestration.expert_model_loop import build_expert_model_loop_trace, save_expert_model_loop_trace
from ITD_agent.orchestration.grouped_inference import run_grouped_experiment
from ITD_agent.orchestration.main_model_loop import build_main_model_loop_trace, save_main_model_loop_trace
from ITD_agent.orchestration.runtime_paths import (
    collect_run_metadata,
    get_eval_output_paths,
    prepare_terrain_inputs_from_cfg,
    validate_runtime_cfg,
)
from ITD_agent.orchestration.runtime_steps import log_to_mlflow, run_semantic_prior_task
from ITD_agent.orchestration.runtime_support import ensure_dir, load_json, run_cmd
from ITD_agent.orchestration.summary_builder import build_run_summary, finalize_run_summary
from ITD_agent.planning.agent.local_refine import run_local_refinement
from ITD_agent.planning.scheduler import (
    extract_plan_summary,
    extract_segmentation_params as extract_segmentation_params_impl,
    generate_expert_model_plan,
    generate_finetune_plan,
    generate_main_model_plan,
)
from ITD_agent.segmentation.executor import execute_segmentation_model
from input_layer.adapters import normalize_agent_runtime_config


def _get_itd_agent_block(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("ITD_agent") or {}


def _get_planning_block(cfg: dict[str, Any]) -> dict[str, Any]:
    return (_get_itd_agent_block(cfg).get("planning") or {})


def _get_roi_refine_block(cfg: dict[str, Any]) -> dict[str, Any]:
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


def _build_input_assessment_compat(input_manifest: dict[str, Any]) -> dict[str, Any]:
    modalities = ((input_manifest.get("metadata") or {}).get("input_modalities") or {})
    return {
        "readiness_score": 1.0 if modalities.get("image") else 0.9,
        "modality_status": dict(modalities),
        "strengths": [],
        "issues": [] if modalities.get("image") else ["缺少遥感影像输入。"],
        "recommended_actions": [],
    }


def _get_template_path(cfg: dict[str, Any], config_path: str) -> str:
    planning_cfg = _get_planning_block(cfg)
    template_cfg = planning_cfg.get("config_templates") or {}
    runtime_template = template_cfg.get("runtime_template")
    base_config = template_cfg.get("base_config")
    return str(runtime_template or base_config or config_path)


def prepare_runtime_config(config_path: str) -> tuple[dict[str, Any], str]:
    raw_cfg = load_raw_yaml(config_path)
    runtime_cfg, _ = normalize_agent_runtime_config(raw_cfg, config_path=config_path)

    persistent_dir = Path(runtime_cfg.get("persistent_output_dir") or runtime_cfg["output_dir"]).resolve()
    persistent_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_path = save_runtime_config(runtime_cfg, persistent_dir / "runtime_execution_config.yaml")
    return runtime_cfg, runtime_config_path


def extract_segmentation_params(cfg: dict[str, Any]) -> dict[str, Any]:
    return extract_segmentation_params_impl(cfg)


def _postprocess_instance_output(cfg: dict[str, Any], inst_shp: str, phase_tag: str) -> str:
    source_path = Path(inst_shp)
    postprocessed_path = source_path.with_name(f"{source_path.stem}_{phase_tag}_postprocessed{source_path.suffix}")
    result = fuse_instance_layers(
        instance_paths=[inst_shp],
        output_path=postprocessed_path,
        boundary_vector_path=reference_vector_path(cfg),
        overlap_ratio_thr=0.5,
        boundary_band_m=1.5,
        min_area_m2=6.0,
    )
    return str(result.get("merged_instance_path") or inst_shp)


def _build_roi_candidate_context(
    *,
    cfg: dict[str, Any],
    y_inst_tif: str | None,
    m_sem_tif: str | None,
    terrain_info: dict[str, Any] | None,
    top_k: int,
    round_idx: int,
    inst_shp: str | None = None,
) -> dict[str, Any]:
    roi_cfg = _get_roi_refine_block(cfg)
    enabled = roi_cfg.get("enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.lower() in {"1", "true", "yes", "y", "on"}
    resolved_y_inst_tif = y_inst_tif
    if not resolved_y_inst_tif and inst_shp and cfg.get("input_image"):
        resolved_y_inst_tif = rasterize_instances_to_label_raster(
            inst_shp=inst_shp,
            reference_raster=str(cfg["input_image"]),
            output_tif=str(Path(cfg["output_dir"]) / "data_processing" / "roi_signal_candidates" / f"round_{int(round_idx):02d}_Y_inst_labels.tif"),
        )
    if not enabled or not resolved_y_inst_tif:
        return {"candidate_rois": None, "signal_roi_summary": {}}

    signal_roi_summary = extract_signal_driven_roi_candidates(
        base_cfg=cfg,
        y_inst_tif=resolved_y_inst_tif,
        m_sem_tif=m_sem_tif,
        terrain_info=terrain_info or {},
        top_k=max(int(top_k) * 2, int(top_k)),
        round_idx=round_idx,
    )
    return {
        "candidate_rois": list(signal_roi_summary.get("selected_candidates") or []),
        "signal_roi_summary": signal_roi_summary,
    }


def _metric_delta(previous_metrics: dict[str, Any], candidate_metrics: dict[str, Any], key: str) -> float | None:
    prev = _safe_float(previous_metrics.get(key))
    cand = _safe_float(candidate_metrics.get(key))
    if prev is None or cand is None:
        return None
    return cand - prev


def _build_refinement_failure_modes(
    *,
    previous_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    previous_score: float | None,
    candidate_score: float | None,
) -> list[str]:
    modes: list[str] = []

    if previous_score is not None and candidate_score is not None and candidate_score > previous_score:
        modes.append("综合参考质量分数下降")

    tree_delta = _metric_delta(previous_metrics, candidate_metrics, "tree_count_error_ratio")
    if tree_delta is not None and tree_delta > 0.01:
        pred_tree = _safe_float(candidate_metrics.get("pred_tree_count"))
        exp_tree = _safe_float(candidate_metrics.get("expected_tree_count"))
        if pred_tree is not None and exp_tree is not None:
            modes.append("树木计数高估加剧" if pred_tree >= exp_tree else "树木计数低估加剧")
        else:
            modes.append("树木计数误差加剧")

    crown_delta = _metric_delta(previous_metrics, candidate_metrics, "mean_crown_width_error_ratio")
    if crown_delta is not None and crown_delta > 0.01:
        pred_crown = _safe_float(candidate_metrics.get("pred_mean_crown_width"))
        exp_crown = _safe_float(candidate_metrics.get("expected_mean_crown_width"))
        if pred_crown is not None and exp_crown is not None:
            modes.append("冠幅偏小问题加剧" if pred_crown <= exp_crown else "冠幅偏大问题加剧")
        else:
            modes.append("冠幅误差加剧")

    closure_delta = _metric_delta(previous_metrics, candidate_metrics, "closure_error_abs")
    if closure_delta is not None and closure_delta > 0.02:
        pred_cover = _safe_float(candidate_metrics.get("pred_cover_ratio"))
        exp_closure = _safe_float(candidate_metrics.get("expected_closure"))
        if pred_cover is not None and exp_closure is not None:
            modes.append("郁闭度恢复不足加剧" if pred_cover <= exp_closure else "郁闭度过覆盖问题加剧")
        else:
            modes.append("郁闭度误差加剧")

    density_delta = _metric_delta(previous_metrics, candidate_metrics, "density_error_abs")
    if density_delta is not None and density_delta > 25.0:
        pred_density = _safe_float(candidate_metrics.get("pred_density_trees_per_ha"))
        exp_density = _safe_float(candidate_metrics.get("expected_density"))
        if pred_density is not None and exp_density is not None:
            modes.append("林分密度高估加剧" if pred_density >= exp_density else "林分密度低估加剧")
        else:
            modes.append("林分密度误差加剧")

    if not modes:
        modes.append("专家模型细化未带来有效提升")
    return list(dict.fromkeys(modes))


def _summarize_refinement_review(
    *,
    initial_score: float | None,
    roi_baseline_score: float | None,
    best_score: float | None,
    best_source: str,
    round_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted = [item for item in round_summaries if item.get("accepted")]
    rejected = [item for item in round_summaries if item.get("accepted") is False]
    mode_counter: Counter[str] = Counter()
    for item in rejected:
        mode_counter.update(item.get("failure_modes") or [])
    return {
        "selection_policy": "publish_best_score",
        "initial_score": initial_score,
        "roi_baseline_score": roi_baseline_score,
        "best_score": best_score,
        "best_source": best_source,
        "accepted_round_count": len(accepted),
        "rejected_round_count": len(rejected),
        "rejected_rounds": [
            {
                "round_idx": item.get("round_idx"),
                "reason": item.get("acceptance_reason"),
                "candidate_score": item.get("candidate_score"),
                "best_score_before_round": item.get("best_score_before_round"),
                "better_than_roi_baseline": item.get("better_than_roi_baseline"),
                "failure_modes": item.get("failure_modes") or [],
            }
            for item in rejected
        ],
        "failure_mode_summary": [
            {"mode": mode, "count": count}
            for mode, count in mode_counter.most_common()
        ],
    }


def run_itd_agent_runtime(config_path: str) -> dict[str, Any]:
    cfg, input_manifest = load_runtime_config(config_path)
    validate_runtime_cfg(cfg)

    if cfg.get("grouped_inference_enabled", False) and not cfg.get("_grouped_dispatch_active", False):
        return run_grouped_experiment(config_path)

    output_dir = Path(cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    planning_root = output_dir / "planning_scheduler"
    ensure_dir(planning_root)

    input_manifest_dict = input_manifest.to_dict()
    terrain_info = prepare_terrain_inputs_from_cfg(cfg)
    data_processing_summary = summarize_data_processing_stage(
        runtime_cfg=cfg,
        input_manifest=input_manifest,
        terrain_info=terrain_info,
    )
    data_processing_summary_dict = data_processing_summary.to_dict()
    input_assessment = _build_input_assessment_compat(input_manifest_dict)

    main_plan = generate_main_model_plan(
        cfg=cfg,
        template_path=_get_template_path(cfg, config_path),
        planning_root=planning_root,
        input_assessment=input_assessment,
        input_manifest=input_manifest_dict,
        data_processing_summary=data_processing_summary_dict,
    )
    effective_main_cfg = main_plan["effective_runtime_cfg"]
    validate_runtime_cfg(effective_main_cfg)
    terrain_info = prepare_terrain_inputs_from_cfg(effective_main_cfg)

    semantic_prior_info = run_semantic_prior_task(effective_main_cfg)
    main_model_info = execute_segmentation_model(
        cfg=effective_main_cfg,
        m_sem_tif=semantic_prior_info["m_sem_tif"],
        phase="segmentation_inference",
        model_role="main_model",
        plan_summary=extract_plan_summary(main_plan),
    )
    main_model_info["y_inst_shp"] = _postprocess_instance_output(
        effective_main_cfg,
        main_model_info["y_inst_shp"],
        phase_tag="main",
    )
    eval_paths = get_eval_output_paths(effective_main_cfg)
    main_eval_info = evaluate_main_model_phase(
        effective_main_cfg,
        inst_shp=main_model_info["y_inst_shp"],
        terrain_info=terrain_info,
        metrics_json=eval_paths["metrics_json"],
        details_csv=eval_paths["details_csv"],
        command_runner=run_cmd,
    )
    main_model_loop_trace = build_main_model_loop_trace(
        run_name=str(effective_main_cfg.get("run_name") or output_dir.name),
        online_scene_state=((data_processing_summary_dict.get("metadata") or {}).get("online_scene_state") or {}),
        input_assessment=input_assessment,
        main_plan=main_plan,
        semantic_prior_info=semantic_prior_info,
        main_model_info=main_model_info,
        main_eval_info=main_eval_info,
    )
    main_plan["main_model_loop_trace_json"] = save_main_model_loop_trace(
        main_model_loop_trace,
        output_dir / "planning_scheduler" / "main_model_loop_trace.json",
    )
    main_roi_candidate_context = _build_roi_candidate_context(
        cfg=effective_main_cfg,
        y_inst_tif=main_model_info.get("y_inst_tif"),
        m_sem_tif=semantic_prior_info.get("m_sem_tif"),
        terrain_info=terrain_info,
        top_k=int(_get_roi_refine_block(effective_main_cfg).get("top_k", 3)),
        round_idx=0,
        inst_shp=main_model_info.get("y_inst_shp"),
    )
    main_roi_assessment = evaluate_roi_phase(
        effective_main_cfg,
        metrics=main_eval_info["metrics"],
        details_csv=main_eval_info["details_csv"],
        round_idx=0,
        y_inst_tif=main_model_info.get("y_inst_tif"),
        m_sem_tif=semantic_prior_info.get("m_sem_tif"),
        terrain_info=terrain_info,
        candidate_rois=main_roi_candidate_context.get("candidate_rois"),
        signal_roi_summary=main_roi_candidate_context.get("signal_roi_summary"),
    )
    roi_decision = main_roi_assessment.get("decision") or {
        "continue_refinement": bool(main_roi_assessment.get("continue_refinement")),
        "decision_source": str(main_roi_assessment.get("decision_source") or "heuristic"),
        "reason": str(main_roi_assessment.get("decision_reason") or ""),
    }

    current_inst_shp = main_model_info["y_inst_shp"]
    current_metrics_json = main_eval_info["metrics_json"]
    current_details_csv = main_eval_info["details_csv"]
    current_metrics = dict(main_eval_info["metrics"])
    current_score = score_reference_metrics(current_metrics, cfg=effective_main_cfg)
    initial_score = current_score
    roi_baseline_score = current_score
    current_config_path = main_plan["generated_config_path"]
    best_source = "main_model"
    roi_round_summaries: list[dict[str, Any]] = []

    roi_cfg = main_plan.get("roi_refine_plan") or _get_roi_refine_block(effective_main_cfg)
    max_rounds = int(roi_cfg.get("max_rounds", 2))
    local_refine_root = output_dir / "roi_refinement"
    ensure_dir(local_refine_root)

    round_idx = 0
    while roi_decision.get("continue_refinement", False) and round_idx < max_rounds:
        round_idx += 1

        expert_plan = generate_expert_model_plan(
            cfg=effective_main_cfg,
            template_path=current_config_path,
            planning_root=planning_root,
            round_idx=round_idx,
            input_assessment=input_assessment,
            input_manifest=input_manifest_dict,
            data_processing_summary=data_processing_summary_dict,
            roi_assessment=main_roi_assessment if round_idx == 1 else roi_round_summaries[-1]["roi_assessment"],
            previous_round_summary={
                "details_csv": current_details_csv,
                "metrics_json": current_metrics_json,
                "metrics": current_metrics,
                "current_score": current_score,
            },
            metrics_json=current_metrics_json,
            details_csv=current_details_csv,
        )
        expert_cfg = expert_plan["effective_runtime_cfg"]
        validate_runtime_cfg(expert_cfg)

        expert_best_params = extract_segmentation_params(expert_cfg)
        expert_roi_plan = expert_plan.get("roi_refine_plan") or roi_cfg
        refine_summary = run_local_refinement(
            base_config_path=expert_plan["generated_config_path"],
            global_details_csv=current_details_csv,
            global_inst_shp=current_inst_shp,
            best_params=expert_best_params,
            xiaoban_id_field=expert_cfg["xiaoban_id_field"],
            top_k=int(expert_roi_plan.get("top_k", 3)),
            buffer_m=float(expert_roi_plan.get("buffer_m", 5.0)),
            strategy_mode=str(expert_roi_plan.get("strategy_mode", "auto")),
            dem_tif=expert_cfg.get("dem_tif"),
            slope_tif=expert_cfg.get("slope_tif"),
            aspect_tif=expert_cfg.get("aspect_tif"),
            local_refine_root=str(local_refine_root),
            preferred_expert_model=(expert_plan.get("expert_model_call_plan") or {}).get("preferred_expert_model"),
            expert_plan_summary=extract_plan_summary(expert_plan),
            roi_candidates=(main_roi_assessment if round_idx == 1 else roi_round_summaries[-1]["roi_assessment"]).get("candidate_rois"),
        )

        best_inst_shp_before_round = current_inst_shp
        best_metrics_json_before_round = current_metrics_json
        best_details_csv_before_round = current_details_csv
        best_metrics_before_round = dict(current_metrics)
        best_score_before_round = current_score
        candidate_inst_shp = refine_summary["merged_shp"]
        candidate_metrics_json = refine_summary["merged_metrics_json"]
        candidate_details_csv = refine_summary["merged_details_csv"]
        candidate_metrics = load_json(candidate_metrics_json)
        expert_eval_info = evaluate_expert_model_phase(
            expert_cfg,
            metrics=candidate_metrics,
            metrics_json=candidate_metrics_json,
            details_csv=candidate_details_csv,
            round_idx=round_idx,
            previous_score=current_score,
            terrain_info=terrain_info,
        )
        roi_assessment = expert_eval_info["roi_assessment"]
        candidate_score = expert_eval_info.get("current_score")
        accept_epsilon = float(roi_cfg.get("improvement_epsilon", 0.01))
        better_than_roi_baseline = False
        if candidate_score is not None and roi_baseline_score is not None:
            better_than_roi_baseline = candidate_score < (roi_baseline_score - accept_epsilon)
        accepted = False
        failure_modes: list[str] = []
        if candidate_score is not None and best_score_before_round is not None:
            accepted = candidate_score < (best_score_before_round - accept_epsilon)
        elif candidate_score is not None and best_score_before_round is None:
            accepted = True
        if accepted:
            acceptance_reason = "candidate_improves_best_score"
        elif not better_than_roi_baseline:
            acceptance_reason = "candidate_not_better_than_roi_baseline"
        else:
            acceptance_reason = "candidate_not_better_than_best_score"
        if accepted:
            current_inst_shp = candidate_inst_shp
            current_metrics_json = candidate_metrics_json
            current_details_csv = candidate_details_csv
            current_metrics = candidate_metrics
            current_score = candidate_score
            current_config_path = expert_plan["generated_config_path"]
            best_source = f"roi_round_{round_idx}"
        else:
            failure_modes = _build_refinement_failure_modes(
                previous_metrics=best_metrics_before_round,
                candidate_metrics=candidate_metrics,
                previous_score=best_score_before_round,
                candidate_score=candidate_score,
            )
        round_candidate_context = _build_roi_candidate_context(
            cfg=expert_cfg,
            y_inst_tif=refine_summary.get("merged_tif") or refine_summary.get("y_inst_tif"),
            m_sem_tif=semantic_prior_info.get("m_sem_tif"),
            terrain_info=terrain_info,
            top_k=int(expert_roi_plan.get("top_k", 3)),
            round_idx=round_idx,
            inst_shp=candidate_inst_shp,
        )
        roi_round_decision = evaluate_roi_phase(
            expert_cfg,
            metrics=candidate_metrics,
            details_csv=candidate_details_csv,
            round_idx=round_idx,
            previous_score=expert_eval_info.get("previous_score"),
            terrain_info=terrain_info,
            candidate_rois=round_candidate_context.get("candidate_rois"),
            signal_roi_summary=round_candidate_context.get("signal_roi_summary"),
        )
        roi_decision = roi_round_decision.get("decision") or {
            "continue_refinement": bool(roi_round_decision.get("continue_refinement")),
            "decision_source": str(roi_round_decision.get("decision_source") or "heuristic"),
            "reason": str(roi_round_decision.get("decision_reason") or ""),
        }
        expert_model_loop_trace = build_expert_model_loop_trace(
            round_idx=round_idx,
            roi_assessment=roi_round_decision,
            expert_plan=expert_plan,
            refine_summary=refine_summary,
            expert_eval_info=expert_eval_info,
            roi_decision=roi_decision,
            accepted=accepted,
            acceptance_reason=acceptance_reason,
            failure_modes=failure_modes,
        )
        expert_model_loop_trace_json = save_expert_model_loop_trace(
            expert_model_loop_trace,
            output_dir / "planning_scheduler" / f"expert_model_loop_round_{round_idx:02d}.json",
        )
        roi_round_summaries.append(
            {
                "round_idx": round_idx,
                "expert_plan": {
                    "generated_config_path": expert_plan.get("generated_config_path"),
                    "parameter_updates": expert_plan.get("parameter_updates"),
                    "llm_result": expert_plan.get("llm_result"),
                    "llm_gateway_result": expert_plan.get("llm_gateway_result"),
                    "scheduler_context": expert_plan.get("scheduler_context"),
                    "runtime_plan": expert_plan.get("runtime_plan"),
                    "roi_refine_plan": expert_plan.get("roi_refine_plan"),
                    "expert_model_call_plan": expert_plan.get("expert_model_call_plan"),
                    "knowledge_embedding_plan": expert_plan.get("knowledge_embedding_plan"),
                },
                "best_params": expert_best_params,
                "refine_summary": refine_summary,
                "expert_model_assessment": expert_eval_info,
                "roi_assessment": roi_round_decision,
                "roi_decision": roi_decision,
                "accepted": accepted,
                "acceptance_reason": acceptance_reason,
                "candidate_inst_shp": candidate_inst_shp,
                "candidate_metrics_json": candidate_metrics_json,
                "candidate_details_csv": candidate_details_csv,
                "candidate_score": candidate_score,
                "best_inst_shp_before_round": best_inst_shp_before_round,
                "best_metrics_json_before_round": best_metrics_json_before_round,
                "best_details_csv_before_round": best_details_csv_before_round,
                "best_score_before_round": best_score_before_round,
                "roi_baseline_score": roi_baseline_score,
                "better_than_roi_baseline": better_than_roi_baseline,
                "selected_inst_shp_after_round": current_inst_shp,
                "selected_score_after_round": current_score,
                "failure_modes": failure_modes,
                "expert_model_loop_trace": expert_model_loop_trace,
                "expert_model_loop_trace_json": expert_model_loop_trace_json,
            }
        )

    final_eval_info = {
        "metrics_json": current_metrics_json,
        "details_csv": current_details_csv,
        "metrics": current_metrics,
        "terrain_info": terrain_info,
        "evaluation_metrics_json": None,
        "evaluation_details_csv": None,
    }
    final_inst_shp = current_inst_shp
    final_roi_assessment = main_roi_assessment if not roi_round_summaries else roi_round_summaries[-1]["roi_assessment"]
    final_roi_decision = roi_decision
    refinement_review = _summarize_refinement_review(
        initial_score=initial_score,
        roi_baseline_score=roi_baseline_score,
        best_score=current_score,
        best_source=best_source,
        round_summaries=roi_round_summaries,
    )
    finetune_recommendation = build_finetune_recommendation_impl(
        effective_main_cfg,
        metrics=current_metrics,
        details_csv=current_details_csv,
        roi_round_count=len(roi_round_summaries),
    )
    finetune_training_plan = generate_finetune_plan(
        runtime_cfg=effective_main_cfg,
        planning_root=planning_root,
        scheduler_context=(roi_round_summaries[-1]["expert_plan"].get("scheduler_context") if roi_round_summaries else main_plan.get("scheduler_context")) or {},
        llm_result=(roi_round_summaries[-1]["expert_plan"].get("llm_result") if roi_round_summaries else main_plan.get("llm_result")),
        finetune_recommendation=finetune_recommendation,
    )

    run_meta = collect_run_metadata(effective_main_cfg, terrain_info)
    run_meta["effective_main_config_path"] = main_plan["generated_config_path"]
    run_meta["final_effective_config_path"] = current_config_path
    run_meta["input_assessment_score"] = input_assessment.get("readiness_score")
    run_meta["best_result_source"] = best_source

    try:
        log_to_mlflow(
            effective_main_cfg,
            run_meta,
            semantic_prior_info,
            final_inst_shp,
            final_eval_info,
        )
    except Exception:
        pass

    summary = build_run_summary(
        config_path=config_path,
        run_meta=run_meta,
        input_manifest=input_manifest_dict,
        data_processing_summary=data_processing_summary_dict,
        semantic_prior_info=semantic_prior_info,
        terrain_info=terrain_info,
        input_assessment=input_assessment,
        main_eval_info=main_eval_info,
        main_plan=main_plan,
        roi_round_summaries=roi_round_summaries,
        final_eval_info=final_eval_info,
        final_inst_shp=final_inst_shp,
        final_roi_assessment=final_roi_assessment,
        final_roi_decision=final_roi_decision,
        refinement_review=refinement_review,
        finetune_recommendation=finetune_recommendation,
        finetune_training_plan=finetune_training_plan,
        main_model_info=main_model_info,
    )
    return finalize_run_summary(
        summary=summary,
        runtime_cfg=effective_main_cfg,
        input_manifest=input_manifest_dict,
        semantic_prior_info=semantic_prior_info,
        segmentation_info=main_model_info,
        final_eval_info=final_eval_info,
    )


def run_itd_agent(config_path: str) -> dict[str, Any]:
    raw_cfg = load_raw_yaml(config_path)
    prepare_only = bool((raw_cfg.get("runtime") or {}).get("config_prepare_only") or raw_cfg.get("config_prepare_only"))
    schedule_prepare_only = bool((raw_cfg.get("runtime") or {}).get("schedule_prepare_only") or raw_cfg.get("schedule_prepare_only"))
    runtime_cfg, runtime_config_path = prepare_runtime_config(config_path)
    if prepare_only or schedule_prepare_only:
        result: dict[str, Any] = {
            "status": "prepared",
            "config_path": config_path,
            "runtime_config_path": runtime_config_path,
            "output_dir": runtime_cfg["output_dir"],
            "persistent_output_dir": runtime_cfg.get("persistent_output_dir"),
            "input_type": runtime_cfg.get("input_type"),
            "input_validation": runtime_cfg.get("_input_validation"),
            "input_profile": runtime_cfg.get("_input_profile"),
        }
        if schedule_prepare_only:
            output_dir = Path(runtime_cfg["output_dir"]).resolve()
            planning_root = output_dir / "planning_scheduler"
            ensure_dir(planning_root)
            input_manifest_dict = runtime_cfg.get("_input_manifest") or {}
            plan = generate_main_model_plan(
                cfg=runtime_cfg,
                template_path=_get_template_path(runtime_cfg, runtime_config_path),
                planning_root=planning_root,
                input_assessment=_build_input_assessment_compat(input_manifest_dict),
                input_manifest=input_manifest_dict,
                data_processing_summary={"status": "schedule_prepare_only"},
            )
            result["status"] = "scheduled"
            result["main_model_plan"] = {
                "generated_config_path": plan.get("generated_config_path"),
                "runtime_plan": plan.get("runtime_plan"),
                "roi_refine_plan": plan.get("roi_refine_plan"),
                "expert_model_call_plan": plan.get("expert_model_call_plan"),
            }
        return result
    return run_itd_agent_runtime(runtime_config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ITD_agent orchestration controller.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_itd_agent(args.config)


if __name__ == "__main__":
    main()
