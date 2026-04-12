from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.evaluation_analysis.evaluator import evaluate_final_phase
from ITD_agent.finetune_pool.query import load_recent_failed_cases
from ITD_agent.llm_gateway import request_run_retrospective
from ITD_agent.memory_store.query import load_recent_success_strategies
from ITD_agent.orchestration.output_management import (
    apply_persistent_retention,
    build_retained_summary,
    cleanup_temp_runtime_dir,
    cleanup_unused_outputs,
    finalize_run_outputs,
    get_cleanup_roots,
    keep_legacy_output_aliases,
    materialize_public_output_aliases,
    sync_runtime_artifacts_to_persistent_root,
)
from ITD_agent.orchestration.runtime_support import copy_optional_file, remove_path, save_json
from ITD_agent.planning.scheduler import extract_plan_summary
from output_layer.reporting.experiment_report import build_experiment_report


RUN_SUMMARY_FILENAME = "ITD_agent_run_summary.json"
LEGACY_RUN_SUMMARY_FILENAME = "run_experiment_summary.json"
RUN_REPORT_FILENAME = "final_evaluation_report.md"
RUN_REPORT_JSON_FILENAME = "final_evaluation_report.json"
LEGACY_RUN_REPORT_FILENAME = "run_experiment_report.md"


def build_run_summary(
    *,
    config_path: str,
    run_meta: dict[str, Any],
    input_manifest: dict[str, Any],
    data_processing_summary: dict[str, Any],
    semantic_prior_info: dict[str, Any],
    terrain_info: dict[str, Any],
    input_assessment: dict[str, Any],
    main_eval_info: dict[str, Any],
    main_plan: dict[str, Any],
    roi_round_summaries: list[dict[str, Any]],
    final_eval_info: dict[str, Any],
    final_inst_shp: str,
    final_roi_assessment: dict[str, Any] | None,
    final_roi_decision: dict[str, Any] | None,
    refinement_review: dict[str, Any],
    finetune_recommendation: dict[str, Any],
    finetune_training_plan: dict[str, Any],
    main_model_info: dict[str, Any],
) -> dict[str, Any]:
    failure_analysis = dict(finetune_recommendation.get("failure_summary") or {})
    failure_analysis["refinement_review"] = refinement_review
    if refinement_review.get("failure_mode_summary"):
        failure_analysis["refinement_failure_modes"] = refinement_review["failure_mode_summary"]
    return {
        "mode": "ITD_agent_run",
        "config_path": config_path,
        "run_name": run_meta.get("run_name"),
        "run_meta": run_meta,
        "input_layer": input_manifest,
        "data_processing": {
            "processing_summary": data_processing_summary,
            **semantic_prior_info,
            "terrain_info": terrain_info,
            "input_assessment": input_assessment,
        },
        "evaluation_analysis": {
            "input_assessment": input_assessment,
            "main_model_assessment": main_eval_info,
            "child_model_assessments": [item.get("child_model_assessment") for item in roi_round_summaries],
            "reference_quality_final": final_eval_info,
            "roi_assessment": final_roi_assessment,
        },
        "llm_gateway": {
            "main_model_planning_used_llm": main_plan.get("llm_result") is not None,
            "main_model_planning_result": main_plan.get("llm_result"),
            "main_model_gateway_trace": main_plan.get("llm_gateway_result"),
            "main_model_pilot_parameter_search": main_plan.get("pilot_search_result"),
            "roi_decision": final_roi_decision,
            "roi_decision_gateway_trace": final_roi_decision.get("llm_gateway_result") if isinstance(final_roi_decision, dict) else None,
            "roi_round_llm_results": [
                item["child_plan"].get("llm_result") for item in roi_round_summaries if item["child_plan"].get("llm_result") is not None
            ],
            "roi_round_gateway_traces": [
                item["child_plan"].get("llm_gateway_result")
                for item in roi_round_summaries
                if item["child_plan"].get("llm_gateway_result") is not None
            ],
        },
        "planning_scheduler": {
            "main_model_plan": extract_plan_summary(main_plan),
            "roi_rounds": [
                {
                    "round_idx": item["round_idx"],
                    **extract_plan_summary(item["child_plan"]),
                    "roi_assessment": item["roi_assessment"],
                    "roi_decision": item["roi_decision"],
                    "accepted": item.get("accepted"),
                    "acceptance_reason": item.get("acceptance_reason"),
                    "candidate_score": item.get("candidate_score"),
                    "best_score_before_round": item.get("best_score_before_round"),
                    "selected_score_after_round": item.get("selected_score_after_round"),
                    "failure_modes": item.get("failure_modes") or [],
                }
                for item in roi_round_summaries
            ],
            "refinement_review": refinement_review,
            "finetune_recommendation": finetune_recommendation,
            "finetune_training_plan": finetune_training_plan,
        },
        "segmentation_model": {
            "main_model": main_model_info,
            "roi_rounds": [
                {
                    "round_idx": item["round_idx"],
                    "refine_summary": item["refine_summary"],
                    "accepted": item.get("accepted"),
                    "acceptance_reason": item.get("acceptance_reason"),
                    "candidate_inst_shp": item.get("candidate_inst_shp"),
                    "selected_inst_shp_after_round": item.get("selected_inst_shp_after_round"),
                    "failure_modes": item.get("failure_modes") or [],
                }
                for item in roi_round_summaries
            ],
            "y_inst_shp": final_inst_shp,
            "tree_crowns_shp": None,
            "tree_points_shp": None,
        },
        "merged_inst_shp": final_inst_shp if roi_round_summaries else None,
        "refinement_review": refinement_review,
        "evaluation": {
            "metrics_json": final_eval_info["metrics_json"],
            "details_csv": final_eval_info["details_csv"],
            "evaluation_metrics_json": final_eval_info.get("evaluation_metrics_json"),
            "evaluation_details_csv": final_eval_info.get("evaluation_details_csv"),
        },
        "metrics": final_eval_info["metrics"],
        "failure_analysis": failure_analysis,
    }


def finalize_run_summary(
    *,
    summary: dict[str, Any],
    runtime_cfg: dict[str, Any],
    input_manifest: dict[str, Any],
    semantic_prior_info: dict[str, Any],
    segmentation_info: dict[str, Any],
    final_eval_info: dict[str, Any],
) -> dict[str, Any]:
    cleanup_roots = get_cleanup_roots(runtime_cfg)
    summary["final_evaluation"] = evaluate_final_phase(summary, runtime_cfg=runtime_cfg)
    summary["llm_gateway"]["run_retrospective"] = request_run_retrospective(
        run_summary=summary,
        memory_context=load_recent_success_strategies(limit=10),
        finetune_context=load_recent_failed_cases(limit=10),
        runtime_cfg=runtime_cfg,
        use_llm=True,
    )

    metrics_parent = Path(runtime_cfg["metrics_json"]).resolve().parent
    summary_json = str(metrics_parent / RUN_SUMMARY_FILENAME)
    report_md = str(metrics_parent / RUN_REPORT_FILENAME)
    report_json = str(metrics_parent / RUN_REPORT_JSON_FILENAME)

    summary["summary_json"] = summary_json
    summary["metrics_json"] = final_eval_info["metrics_json"]
    summary["details_csv"] = final_eval_info["details_csv"]
    save_json(summary, summary_json)

    aliases = materialize_public_output_aliases(
        runtime_cfg,
        semantic_prior_info,
        segmentation_info["y_inst_shp"],
        final_eval_info,
        summary_json,
        report_md,
    )
    summary["output_aliases"] = aliases
    report_path = build_experiment_report(
        summary,
        report_md,
        runtime_cfg=runtime_cfg,
        report_json_path=report_json,
    )
    summary["report_md"] = report_path
    summary["report_json"] = report_json

    summary = finalize_run_outputs(
        summary=summary,
        runtime_cfg=runtime_cfg,
        input_manifest=input_manifest,
    )
    final_outputs = summary.get("final_outputs") or {}
    if final_outputs.get("tree_crowns_shp"):
        summary["segmentation_model"]["tree_crowns_shp"] = final_outputs["tree_crowns_shp"]
        summary["tree_crowns_shp"] = final_outputs["tree_crowns_shp"]
    if final_outputs.get("tree_points_shp"):
        summary["segmentation_model"]["tree_points_shp"] = final_outputs["tree_points_shp"]
        summary["tree_points_shp"] = final_outputs["tree_points_shp"]
    if final_outputs.get("segmentation_visualization_png"):
        summary["segmentation_visualization_png"] = final_outputs["segmentation_visualization_png"]
    if final_outputs.get("final_evaluation_report_md"):
        summary["report_md"] = final_outputs["final_evaluation_report_md"]
    if final_outputs.get("final_evaluation_report_json"):
        summary["report_json"] = final_outputs["final_evaluation_report_json"]

    cleanup_info = cleanup_unused_outputs(
        runtime_cfg,
        semantic_prior_info,
        segmentation_info,
        report_path,
        report_json,
    )
    summary["cleanup"] = cleanup_info
    if runtime_cfg.get("keep_debug_outputs", False):
        summary["report_md"] = report_path
        summary["report_json"] = report_json

    sync_info = sync_runtime_artifacts_to_persistent_root(summary=summary, runtime_cfg=runtime_cfg)
    summary["runtime_artifact_sync"] = sync_info
    copied = sync_info.get("copied", {})
    if copied.get("summary_json"):
        summary["summary_json"] = copied["summary_json"]
    if copied.get("metrics_json"):
        summary["metrics_json"] = copied["metrics_json"]
    if copied.get("details_csv"):
        summary["details_csv"] = copied["details_csv"]

    save_json(summary, summary["summary_json"])
    legacy_summary_path = Path(summary["summary_json"]).resolve().parent / LEGACY_RUN_SUMMARY_FILENAME
    if keep_legacy_output_aliases(runtime_cfg):
        copy_optional_file(summary["summary_json"], Path(summary["summary_json"]).resolve().parent / LEGACY_RUN_SUMMARY_FILENAME)
    else:
        remove_path(legacy_summary_path, allowed_roots=cleanup_roots)
    if keep_legacy_output_aliases(runtime_cfg) and runtime_cfg.get("keep_debug_outputs", False) and report_path:
        copy_optional_file(report_path, Path(report_path).resolve().parent / LEGACY_RUN_REPORT_FILENAME)

    summary["runtime_cleanup"] = cleanup_temp_runtime_dir(runtime_cfg)
    summary["retention"] = apply_persistent_retention(summary=summary, runtime_cfg=runtime_cfg)
    summary = build_retained_summary(summary=summary, runtime_cfg=runtime_cfg)
    save_json(summary, summary["summary_json"])
    return summary
