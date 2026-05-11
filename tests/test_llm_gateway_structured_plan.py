from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.llm_gateway.audit import build_audit_record, write_audit_record
from ITD_agent.llm_gateway.fallback import fallback_structured_plan
from ITD_agent.llm_gateway.response_parser import parse_structured_plan


def test_parse_structured_plan_normalizes_llm_json() -> None:
    plan = parse_structured_plan(
        """
        ```json
        {"recommended_action":"call_expert_model","preferred_expert_family":"dense_adhesion","reason":"roi errors","confidence": 1.7}
        ```
        """
    )

    assert plan["recommended_action"] == "call_expert_model"
    assert plan["preferred_expert_family"] == "dense_adhesion"
    assert plan["confidence"] == 1.0


def test_fallback_plan_and_audit_record_are_structured(tmp_path: Path) -> None:
    plan = fallback_structured_plan(task_type="plan_expert", error="offline")
    response = {"status": "fallback", "fallback_used": True, "error": "offline", "parsed_result": plan}
    record = build_audit_record(task_type="plan_expert", provider="none", model=None, prompt="{}", response=response)
    audit_path = write_audit_record(tmp_path / "audit.jsonl", record)
    rows = [json.loads(line) for line in Path(audit_path).read_text(encoding="utf-8").splitlines()]

    assert plan["recommended_action"] == "use_rule_guard"
    assert rows[0]["task_type"] == "plan_expert"
    assert rows[0]["fallback_used"] is True
    assert rows[0]["prompt_hash"]
    assert rows[0]["stage"] == "plan_expert"
    assert "input_context_path" in rows[0]
    assert "raw_response_path" in rows[0]
    assert "parsed_response_path" in rows[0]
    assert rows[0]["validation_status"] == "valid"
