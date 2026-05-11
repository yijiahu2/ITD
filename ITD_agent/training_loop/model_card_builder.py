from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.evolution.review.io_utils import write_json


def build_model_card(*, record: dict[str, Any], output_dir: str | Path, evidence: dict[str, Any]) -> str:
    card = {
        **record,
        "traceability": {
            "dataset_card_path": evidence.get("dataset_card_path"),
            "training_plan_path": evidence.get("training_plan_path"),
            "generated_config_path": evidence.get("generated_config_path"),
            "replay_guard_report_path": evidence.get("replay_guard_report_path"),
        },
        "safety_constraints": {
            "active_model_replace": "forbidden_in_v3_1",
            "active_route_map_update": "forbidden_in_v3_1",
            "active_skill_policy_update": "forbidden_in_v3_1",
        },
    }
    return write_json(Path(output_dir) / "model_registry" / "model_cards" / f"{record['model_version_id']}.json", card)
