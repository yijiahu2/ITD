from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_main_model_loop_trace(
    *,
    run_name: str,
    online_scene_state: dict[str, Any] | None,
    input_assessment: dict[str, Any],
    main_plan: dict[str, Any],
    semantic_prior_info: dict[str, Any],
    main_model_info: dict[str, Any],
    main_eval_info: dict[str, Any],
) -> dict[str, Any]:
    pilot_search = main_plan.get("pilot_search_result") or {}
    return {
        "loop_name": "main_model_loop",
        "run_name": run_name,
        "stages": [
            "input_assessment",
            "constrained_parameter_hypothesis",
            "pilot_validation",
            "global_execution",
            "online_quality_evaluation",
        ],
        "online_scene_state": online_scene_state or {},
        "input_assessment": {
            "readiness_score": input_assessment.get("readiness_score"),
            "modality_status": input_assessment.get("modality_status") or {},
            "issues": input_assessment.get("issues") or [],
            "recommended_actions": input_assessment.get("recommended_actions") or [],
        },
        "parameter_hypothesis": {
            "generated_config_path": main_plan.get("generated_config_path"),
            "parameter_updates": main_plan.get("parameter_updates") or {},
            "runtime_plan": main_plan.get("runtime_plan") or {},
            "planner_source": "planning_scheduler",
        },
        "pilot_validation": {
            "enabled": bool(pilot_search),
            "result": pilot_search,
        },
        "global_execution": {
            "semantic_prior": {
                "m_sem_tif": semantic_prior_info.get("m_sem_tif"),
                "m_sem_png": semantic_prior_info.get("m_sem_png"),
            },
            "main_model": {
                "y_inst_shp": main_model_info.get("y_inst_shp"),
                "y_inst_tif": main_model_info.get("y_inst_tif"),
                "execution_result": main_model_info.get("execution_result") or {},
            },
        },
        "quality_evaluation": {
            "assessment_mode": main_eval_info.get("assessment_mode"),
            "quality_score": main_eval_info.get("quality_score"),
            "online_quality": main_eval_info.get("online_quality") or {},
            "reference_inventory_metrics_available": bool(main_eval_info.get("metrics")),
        },
    }


def save_main_model_loop_trace(trace: dict[str, Any], output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    return str(path)
