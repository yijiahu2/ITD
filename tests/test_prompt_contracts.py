from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.llm_gateway.prompt_contracts import get_prompt_contract, list_prompt_contracts


def test_prompt_contracts_define_four_canonical_prompts() -> None:
    contracts = list_prompt_contracts()

    assert set(contracts) == {"scene_profiler", "parameter_planner", "expert_router", "retrospective"}
    assert all(item["output_format"] == "json" for item in contracts.values())


def test_parameter_planner_contract_forbids_free_parameters() -> None:
    contract = get_prompt_contract("parameter_planner")

    assert "model_search_space" in contract["allowed_inputs"]
    assert "invent_parameters_outside_search_space" in contract["forbidden_behaviors"]
