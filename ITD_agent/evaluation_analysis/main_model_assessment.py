from __future__ import annotations

from typing import Any

from .reference_quality_engine import evaluate_reference_quality


def evaluate_main_model_assessment(
    cfg: dict[str, Any],
    *,
    inst_shp: str,
    terrain_info: dict[str, Any],
    metrics_json: str | None = None,
    details_csv: str | None = None,
    command_runner=None,
) -> dict[str, Any]:
    return evaluate_reference_quality(
        cfg,
        inst_shp=inst_shp,
        terrain_info=terrain_info,
        assessment_phase="main_model",
        metrics_json=metrics_json,
        details_csv=details_csv,
        command_runner=command_runner,
    )
