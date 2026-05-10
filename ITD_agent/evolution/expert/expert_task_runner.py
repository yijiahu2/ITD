from __future__ import annotations

from typing import Any

from ITD_agent.segmentation.adapters.mock_expert import run_mock_expert_task
from ITD_agent.segmentation.adapters.replay_expert import run_replay_expert_task

from .expert_task_builder import ExpertTask


def run_expert_tasks(
    *,
    expert_tasks: list[ExpertTask],
    gt_instances: list[dict[str, Any]],
    main_instances: list[dict[str, Any]],
    execution_mode: str,
    expert_models_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = expert_models_cfg or {}
    results: list[dict[str, Any]] = []
    for task in expert_tasks:
        if execution_mode == "replay":
            results.append(run_replay_expert_task(task=task, replay_json=cfg["replay_json"]))
        else:
            results.append(
                run_mock_expert_task(
                    task=task,
                    gt_instances=gt_instances,
                    main_instances=main_instances,
                    strategy=str(cfg.get("mock_strategy") or "use_gt_or_perturbed_gt"),
                )
            )
    return results
