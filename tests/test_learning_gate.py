from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.learning_gate import decide_learning_event


def test_learning_gate_writes_success_only_with_delta_and_signatures() -> None:
    decision = decide_learning_event(
        {
            "event_type": "main_model_trial",
            "score_before": 0.30,
            "score_after": 0.22,
            "scene_signature": {"domain": "subtropical"},
            "parameter_signature": {"tile": 2048},
        }
    )

    assert decision["should_write_success_memory"] is True
    assert decision["should_write_failure_pattern"] is False


def test_learning_gate_writes_failure_and_finetune_for_repeated_residual() -> None:
    decision = decide_learning_event(
        {
            "event_type": "expert_model_trial",
            "repeat_count": 3,
            "residual_type": "boundary_height_mismatch",
            "scene_signature": {"domain": "subtropical"},
        }
    )

    assert decision["should_write_failure_pattern"] is True
    assert decision["should_write_finetune_pool"] is True
