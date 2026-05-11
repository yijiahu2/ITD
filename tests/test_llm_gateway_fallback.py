from __future__ import annotations

from ITD_agent.llm_gateway.fallback import fallback_structured_plan


def test_llm_gateway_fallback_returns_structured_decision() -> None:
    plan = fallback_structured_plan(task_type="plan_expert_model_config", error="offline")

    assert plan["recommended_action"] == "use_rule_guard"
    assert plan["confidence"] <= 1.0
    assert "error" in plan
