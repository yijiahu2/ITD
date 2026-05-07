from __future__ import annotations

from typing import Any

from .decision_flags import build_decision_flags
from .flow_decisions import build_main_model_flow_decision
from .online_quality_engine import evaluate_online_quality
from .reference_quality_engine import evaluate_reference_quality


def _resolve_online_quality_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    evaluation_cfg = cfg.get("evaluation") or {}
    analysis_cfg = evaluation_cfg.get("analysis") or {}
    online_cfg = analysis_cfg.get("online_quality") or {}
    return dict(online_cfg if isinstance(online_cfg, dict) else {})


def _has_reference_inventory(cfg: dict[str, Any]) -> bool:
    return bool(
        cfg.get("reference_vector_path")
        or cfg.get("inventory_vector_path")
        or cfg.get("xiaoban_shp")
    )


def evaluate_main_model_assessment(
    cfg: dict[str, Any],
    *,
    inst_shp: str,
    terrain_info: dict[str, Any],
    metrics_json: str | None = None,
    details_csv: str | None = None,
    command_runner=None,
) -> dict[str, Any]:
    online_eval = evaluate_online_quality(
        inst_shp=inst_shp,
        m_sem_tif=str(cfg.get("output_dir") and (cfg["output_dir"] + "/M_sem.tif")) if cfg.get("output_dir") else None,
        chm_tif=cfg.get("chm_tif"),
        patch_raster=cfg.get("input_image"),
        quality_cfg=_resolve_online_quality_cfg(cfg),
    )
    if not _has_reference_inventory(cfg):
        result = {
            "assessment_phase": "main_model",
            "assessment_mode": "online_only",
            "metrics_json": metrics_json,
            "details_csv": details_csv,
            "metrics": {},
            "detail_summary": {},
            "quality_score": online_eval.get("quality_score"),
            "online_quality": online_eval,
            "terrain_info": terrain_info,
        }
        result["decision_flags"] = build_decision_flags(result, runtime_cfg=cfg)
        result["flow_decision"] = build_main_model_flow_decision(result)
        return result

    reference_eval = evaluate_reference_quality(
        cfg,
        inst_shp=inst_shp,
        terrain_info=terrain_info,
        assessment_phase="main_model",
        metrics_json=metrics_json,
        details_csv=details_csv,
        command_runner=command_runner,
    )
    reference_eval["assessment_mode"] = "online_plus_reference"
    reference_eval["online_quality"] = online_eval
    reference_eval["decision_flags"] = build_decision_flags(reference_eval, runtime_cfg=cfg)
    reference_eval["flow_decision"] = build_main_model_flow_decision(reference_eval)
    return reference_eval
