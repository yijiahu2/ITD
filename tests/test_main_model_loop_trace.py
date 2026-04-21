from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.orchestration.main_model_loop import build_main_model_loop_trace, save_main_model_loop_trace


def test_main_model_loop_trace_records_ordered_loop_stages(tmp_path: Path) -> None:
    trace = build_main_model_loop_trace(
        run_name="demo",
        online_scene_state={"scene_id": "demo"},
        input_assessment={"readiness_score": 1.0, "modality_status": {"image": True}},
        main_plan={"generated_config_path": "/tmp/cfg.yaml", "parameter_updates": {"tile": 2048}, "pilot_search_result": {"best": "pilot_base"}},
        semantic_prior_info={"m_sem_tif": "/tmp/M_sem.tif"},
        main_model_info={"y_inst_shp": "/tmp/Y_inst.shp"},
        main_eval_info={"assessment_mode": "online_only", "quality_score": 0.2, "online_quality": {"metrics": {}}},
    )

    assert trace["stages"] == [
        "input_assessment",
        "constrained_parameter_hypothesis",
        "pilot_validation",
        "global_execution",
        "online_quality_evaluation",
    ]
    assert trace["pilot_validation"]["enabled"] is True
    out = tmp_path / "trace.json"
    save_main_model_loop_trace(trace, out)
    assert json.loads(out.read_text(encoding="utf-8"))["loop_name"] == "main_model_loop"
