from __future__ import annotations

from ITD_agent.evolution.adaptive_inference import run_adaptive_inference_stage


def test_adaptive_inference_stage_has_formal_entrypoint() -> None:
    assert callable(run_adaptive_inference_stage)
