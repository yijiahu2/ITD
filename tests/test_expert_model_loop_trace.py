from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.orchestration.expert_model_loop import build_expert_model_loop_trace


def test_expert_model_loop_trace_contains_residual_and_route() -> None:
    trace = build_expert_model_loop_trace(
        round_idx=1,
        roi_assessment={"trigger_metrics": ["mean_crown_width_error_ratio"], "candidate_rois": [{"candidate_id": "roi_1"}]},
        expert_plan={
            "expert_model_call_plan": {
                "preferred_expert_family": "boundary_calibration",
                "preferred_expert_model": "boundary_calibration_template",
                "candidate_models": ["boundary_calibration_template"],
            },
            "parameter_updates": {"score_thr": 0.2},
        },
        refine_summary={"merged_shp": "/tmp/merged.shp"},
        expert_eval_info={"assessment_phase": "expert_model", "current_score": 0.2, "previous_score": 0.3},
        roi_decision={"continue_refinement": False},
        accepted=True,
        acceptance_reason="candidate_improves_best_score",
    )

    assert trace["loop_name"] == "expert_model_loop"
    assert "boundary_or_crown_size_residual" in trace["residual_error_profile"]["residual_types"]
    assert trace["expert_route_plan"]["preferred_expert_family"] == "boundary_calibration"
    assert trace["accept_reject"]["accepted"] is True
