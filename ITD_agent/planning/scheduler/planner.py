from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.planning.scheduler.runtime_scheduler import (
    build_child_model_planning_runtime_cfg,
    build_expert_model_planning_runtime_cfg,
    build_finetune_training_plan,
    build_main_model_planning_runtime_cfg,
    plan_runtime_config,
)
from ITD_agent.planning.scheduler.template_manager import resolve_template_metadata


def _get_template_output_root(planning_root: str | Path, template_path: str, plan_scope: str) -> Path:
    planning_root = Path(planning_root)
    meta = resolve_template_metadata(template_path)
    return planning_root / meta["template_group"] / meta["template_category"] / meta["template_name"] / plan_scope


def generate_main_model_plan(
    *,
    cfg: dict[str, Any],
    template_path: str,
    planning_root: str | Path,
    input_assessment: dict[str, Any],
    input_manifest: dict[str, Any],
    data_processing_summary: dict[str, Any],
) -> dict[str, Any]:
    plan_dir = _get_template_output_root(planning_root, template_path, "runtime")
    runtime_cfg = build_main_model_planning_runtime_cfg(
        cfg=cfg,
        input_assessment=input_assessment,
        input_manifest=input_manifest,
        data_processing_summary=data_processing_summary,
    )
    return plan_runtime_config(
        template_path=template_path,
        output_path=plan_dir / "main_model_runtime_config.yaml",
        runtime_cfg=runtime_cfg,
    )


def generate_expert_model_plan(
    *,
    cfg: dict[str, Any],
    template_path: str,
    planning_root: str | Path,
    round_idx: int,
    input_assessment: dict[str, Any],
    input_manifest: dict[str, Any],
    data_processing_summary: dict[str, Any],
    roi_assessment: dict[str, Any],
    previous_round_summary: dict[str, Any],
    metrics_json: str | None = None,
    details_csv: str | None = None,
    summary_json: str | None = None,
) -> dict[str, Any]:
    plan_dir = _get_template_output_root(planning_root, template_path, "runtime")
    runtime_cfg = build_expert_model_planning_runtime_cfg(
        cfg=cfg,
        input_assessment=input_assessment,
        input_manifest=input_manifest,
        data_processing_summary=data_processing_summary,
        roi_assessment=roi_assessment,
        previous_round_summary=previous_round_summary,
    )
    return plan_runtime_config(
        template_path=template_path,
        output_path=plan_dir / f"expert_model_round_{round_idx:02d}.yaml",
        runtime_cfg=runtime_cfg,
        metrics_json=metrics_json,
        details_csv=details_csv,
        summary_json=summary_json,
    )


def generate_child_model_plan(
    *,
    cfg: dict[str, Any],
    template_path: str,
    planning_root: str | Path,
    round_idx: int,
    input_assessment: dict[str, Any],
    input_manifest: dict[str, Any],
    data_processing_summary: dict[str, Any],
    roi_assessment: dict[str, Any],
    previous_round_summary: dict[str, Any],
    metrics_json: str | None = None,
    details_csv: str | None = None,
    summary_json: str | None = None,
) -> dict[str, Any]:
    return generate_expert_model_plan(
        cfg=cfg,
        template_path=template_path,
        planning_root=planning_root,
        round_idx=round_idx,
        input_assessment=input_assessment,
        input_manifest=input_manifest,
        data_processing_summary=data_processing_summary,
        roi_assessment=roi_assessment,
        previous_round_summary=previous_round_summary,
        metrics_json=metrics_json,
        details_csv=details_csv,
        summary_json=summary_json,
    )


def generate_finetune_plan(
    *,
    runtime_cfg: dict[str, Any],
    planning_root: str | Path,
    scheduler_context: dict[str, Any],
    llm_result: dict[str, Any] | None,
    finetune_recommendation: dict[str, Any],
) -> dict[str, Any]:
    template_path = str(((runtime_cfg.get("pipeline") or {}).get("finetune_config") or ""))
    plan_dir = _get_template_output_root(planning_root, template_path or "adhoc_finetune", "finetune")
    return build_finetune_training_plan(
        runtime_cfg=runtime_cfg,
        scheduler_context=scheduler_context,
        llm_result=llm_result,
        finetune_recommendation=finetune_recommendation,
        output_path=plan_dir / "finetune_training_plan.yaml",
    )


def extract_plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    roi_plan = plan.get("roi_refine_plan")
    return {
        "generated_config_path": plan.get("generated_config_path"),
        "parameter_updates": plan.get("parameter_updates"),
        "scheduler_context": plan.get("scheduler_context"),
        "llm_gateway_result": plan.get("llm_gateway_result"),
        "pilot_search_result": plan.get("pilot_search_result"),
        "runtime_plan": plan.get("runtime_plan"),
        "roi_refine_plan": roi_plan,
        "roi_extraction_plan": roi_plan,
        "expert_model_call_plan": plan.get("expert_model_call_plan"),
        "knowledge_embedding_plan": plan.get("knowledge_embedding_plan"),
        "finetune_training_plan": plan.get("finetune_training_plan"),
    }
