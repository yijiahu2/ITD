from __future__ import annotations

from typing import Any

from .prompts import (
    _build_planning_prompt,
    _build_retrospective_prompt,
    _build_roi_candidate_selection_prompt,
    _build_roi_decision_prompt,
)


def build_prompt(task_type: str, **kwargs: Any) -> str:
    if task_type.startswith("plan_"):
        return _build_planning_prompt(
            planning_stage=str(kwargs["planning_stage"]),
            template_cfg=dict(kwargs["template_cfg"]),
            scheduler_context=dict(kwargs["scheduler_context"]),
        )
    if task_type == "decide_roi_continuation":
        return _build_roi_decision_prompt(
            roi_assessment=dict(kwargs["roi_assessment"]),
            metrics=dict(kwargs["metrics"]),
        )
    if task_type == "select_roi_candidates":
        return _build_roi_candidate_selection_prompt(
            candidate_rois=list(kwargs["candidate_rois"]),
            metrics=dict(kwargs["metrics"]),
            scene_analysis=kwargs.get("scene_analysis"),
        )
    if task_type == "summarize_run_retrospective":
        return _build_retrospective_prompt(
            run_summary=dict(kwargs["run_summary"]),
            memory_context=kwargs.get("memory_context"),
            finetune_context=kwargs.get("finetune_context"),
        )
    raise KeyError(f"Unknown LLM prompt task type: {task_type}")
