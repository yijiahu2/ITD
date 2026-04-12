"""Planning and scheduling modules for ITD_agent.

Keep package imports lazy so submodules such as
`ITD_agent.planning.scheduler.expert_taxonomy` do not eagerly import the full
scheduler stack and create circular-import chains during orchestrator startup.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "build_child_model_planning_runtime_cfg": "ITD_agent.planning.scheduler",
    "build_finetune_training_plan": "ITD_agent.planning.scheduler",
    "build_main_model_planning_runtime_cfg": "ITD_agent.planning.scheduler",
    "build_scheduler_context": "ITD_agent.planning.scheduler",
    "extract_plan_summary": "ITD_agent.planning.scheduler",
    "extract_segmentation_params": "ITD_agent.planning.scheduler",
    "generate_child_model_plan": "ITD_agent.planning.scheduler",
    "generate_adaptive_config_from_template": "ITD_agent.planning.scheduler",
    "generate_finetune_plan": "ITD_agent.planning.scheduler",
    "generate_main_model_plan": "ITD_agent.planning.scheduler",
    "plan_runtime_config": "ITD_agent.planning.scheduler",
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
