from __future__ import annotations

from typing import Any


def _score_delta(event: dict[str, Any]) -> float | None:
    before = event.get("score_before")
    after = event.get("score_after")
    try:
        if before is None or after is None:
            return None
        return float(before) - float(after)
    except Exception:
        return None


def decide_learning_event(event: dict[str, Any]) -> dict[str, Any]:
    event_type = str(event.get("event_type") or "unknown")
    delta = _score_delta(event)
    repeat_count = int(event.get("repeat_count") or 1)
    residual_type = str(event.get("residual_type") or "").strip()
    has_scene_signature = bool(event.get("scene_signature"))
    has_parameter_signature = bool(event.get("parameter_signature"))
    should_write_success = bool(delta is not None and delta >= 0.03 and has_scene_signature and has_parameter_signature)
    should_write_failure = bool(repeat_count >= 2 and residual_type and has_scene_signature)
    should_write_finetune = bool(should_write_failure and event_type in {"expert_model_trial", "main_model_trial"})
    return {
        "event_type": event_type,
        "score_delta": delta,
        "should_write_success_memory": should_write_success,
        "should_write_failure_pattern": should_write_failure,
        "should_write_finetune_pool": should_write_finetune,
        "reason": (
            "significant_positive_delta"
            if should_write_success
            else "repeated_residual_failure"
            if should_write_failure
            else "insufficient_evidence"
        ),
    }
