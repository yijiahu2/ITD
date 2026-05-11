from __future__ import annotations

from ITD_agent.training_loop import run_training_loop


def test_controlled_training_entrypoint_exists() -> None:
    assert callable(run_training_loop)
