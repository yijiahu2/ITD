"""Scheduler exports with lazy imports.

Importing a deep submodule from this package should not force-import
`context_builder` or planner runtime helpers unless they are actually needed.
This avoids circular imports during orchestrator/bootstrap code paths.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "apply_parameter_updates": "ITD_agent.planning.scheduler.template_manager",
    "build_scheduler_context": "ITD_agent.planning.scheduler.context_builder",
    "build_expert_model_planning_runtime_cfg": "ITD_agent.planning.scheduler.runtime_scheduler",
    "build_finetune_training_plan": "ITD_agent.planning.scheduler.runtime_scheduler",
    "build_main_model_planning_runtime_cfg": "ITD_agent.planning.scheduler.runtime_scheduler",
    "extract_plan_summary": "ITD_agent.planning.scheduler.planner",
    "extract_segmentation_params": "ITD_agent.planning.scheduler.runtime_scheduler",
    "generate_expert_model_plan": "ITD_agent.planning.scheduler.planner",
    "generate_adaptive_config_from_template": "ITD_agent.planning.scheduler.adaptive_config_generator",
    "generate_finetune_plan": "ITD_agent.planning.scheduler.planner",
    "generate_main_model_plan": "ITD_agent.planning.scheduler.planner",
    "load_config_template": "ITD_agent.planning.scheduler.template_manager",
    "materialize_generated_config": "ITD_agent.planning.scheduler.template_manager",
    "plan_runtime_config": "ITD_agent.planning.scheduler.runtime_scheduler",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
