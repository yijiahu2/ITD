from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ITD_agent.evolution.expert.expert_task_builder import ExpertTask


def run_replay_expert_task(*, task: ExpertTask, replay_json: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(replay_json).read_text(encoding="utf-8"))
    instances_by_task = payload.get("instances_by_task") or {}
    instances = instances_by_task.get(task.expert_task_id) or payload.get("annotations") or []
    return {
        "expert_task_id": task.expert_task_id,
        "expert_model": task.expert_model,
        "execution_mode": "replay",
        "oracle_mock": False,
        "instances": list(instances),
        "status": "completed",
    }
