from __future__ import annotations

import inspect

from ITD_agent.orchestration import workflow


def test_workflow_module_is_the_public_orchestration_surface() -> None:
    public_names = {name for name, _ in inspect.getmembers(workflow) if not name.startswith("_")}

    assert {"run_workflow", "run_stage", "WorkflowResult"}.issubset(public_names)
