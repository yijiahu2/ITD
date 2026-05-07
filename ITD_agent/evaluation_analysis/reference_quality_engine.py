from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .artifact_store import get_phase_dir
from .contracts import ReferenceQualityResult
from .detail_ranker import summarize_details_csv
from .metrics_catalog import REFERENCE_METRIC_CATALOG


def _reference_vector_path(cfg: dict[str, Any]) -> str | None:
    return (
        cfg.get("reference_vector_path")
        or cfg.get("inventory_vector_path")
        or cfg.get("xiaoban_shp")
    )


def _reference_id_field(cfg: dict[str, Any]) -> str | None:
    return (
        cfg.get("reference_id_field")
        or cfg.get("inventory_id_field")
        or cfg.get("xiaoban_id_field")
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


DEFAULT_SCORE_WEIGHTS = {
    "tree_count_error_ratio": 0.30,
    "mean_crown_width_error_ratio": 0.40,
    "closure_error_abs": 0.20,
    "density_error_ratio": 0.10,
}

BOUNDARY_FOCUSED_SCORE_WEIGHTS = {
    "tree_count_error_ratio": 0.15,
    "mean_crown_width_error_ratio": 0.55,
    "closure_error_abs": 0.20,
    "density_error_ratio": 0.10,
}

METRIC_CATEGORIES = REFERENCE_METRIC_CATALOG


def _get_roi_refine_block(cfg: dict[str, Any] | None) -> dict[str, Any]:
    planning_cfg = (((cfg or {}).get("ITD_agent") or {}).get("planning") or {})
    roi_cfg = planning_cfg.get("roi_extraction")
    if isinstance(roi_cfg, dict):
        return roi_cfg
    roi_cfg = planning_cfg.get("roi_refine")
    return roi_cfg if isinstance(roi_cfg, dict) else {}


def _normalize_weight_map(raw: dict[str, Any], defaults: dict[str, float]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for key, default_value in defaults.items():
        value = _safe_float((raw or {}).get(key))
        weights[key] = max(float(value if value is not None else default_value), 0.0)
    total = sum(weights.values())
    if total <= 0:
        return dict(defaults)
    return {key: value / total for key, value in weights.items()}


def build_reference_score_breakdown(
    metrics: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tree = _safe_float(metrics.get("tree_count_error_ratio"))
    crown = _safe_float(metrics.get("mean_crown_width_error_ratio"))
    closure = _safe_float(metrics.get("closure_error_abs"))
    density_abs = _safe_float(metrics.get("density_error_abs"))
    expected_density = _safe_float(metrics.get("expected_density"))
    if tree is None or crown is None or closure is None:
        return {
            "score": None,
            "weights": dict(DEFAULT_SCORE_WEIGHTS),
            "normalized_metrics": {},
            "focus_mode": "incomplete_metrics",
        }

    density_ratio = None
    if density_abs is not None:
        if expected_density is not None and expected_density > 0:
            density_ratio = density_abs / expected_density
        else:
            density_ratio = density_abs / 1000.0

    focus_mode = "balanced"
    default_weights = DEFAULT_SCORE_WEIGHTS
    if tree <= 0.12:
        focus_mode = "boundary_priority"
        default_weights = BOUNDARY_FOCUSED_SCORE_WEIGHTS

    roi_cfg = _get_roi_refine_block(cfg)
    weights = _normalize_weight_map(roi_cfg.get("score_weights") or {}, default_weights)
    normalized_metrics = {
        "tree_count_error_ratio": float(tree),
        "mean_crown_width_error_ratio": float(crown),
        "closure_error_abs": float(closure),
        "density_error_ratio": float(max(density_ratio or 0.0, 0.0)),
    }
    weighted_terms = {
        key: {
            **METRIC_CATEGORIES[key],
            "value": normalized_metrics[key],
            "weight": weights[key],
            "contribution": normalized_metrics[key] * weights[key],
        }
        for key in weights
    }
    metric_groups: dict[str, dict[str, Any]] = {}
    for key, term in weighted_terms.items():
        category = str(term["category"])
        group = metric_groups.setdefault(
            category,
            {
                "category": category,
                "label": term["label"],
                "direction": term["direction"],
                "metrics": [],
                "contribution": 0.0,
            },
        )
        group["metrics"].append(
            {
                "metric": key,
                "value": term["value"],
                "weight": term["weight"],
                "contribution": term["contribution"],
            }
        )
        group["contribution"] = float(group["contribution"]) + float(term["contribution"])
    score = sum(float(term["contribution"]) for term in weighted_terms.values())
    return {
        "score": float(score),
        "weights": weights,
        "normalized_metrics": normalized_metrics,
        "weighted_terms": weighted_terms,
        "metric_groups": metric_groups,
        "focus_mode": focus_mode,
    }


def score_reference_metrics(metrics: dict[str, Any], *, cfg: dict[str, Any] | None = None) -> float | None:
    breakdown = build_reference_score_breakdown(metrics, cfg=cfg)
    return breakdown.get("score")


def evaluate_reference_quality(
    cfg: dict[str, Any],
    *,
    inst_shp: str,
    terrain_info: dict[str, Any],
    assessment_phase: str,
    metrics_json: str | None = None,
    details_csv: str | None = None,
    command_runner=None,
) -> dict[str, Any]:
    phase_root = get_phase_dir(cfg, assessment_phase)
    metrics_json = metrics_json or str(phase_root / "evaluation_metrics.json")
    details_csv = details_csv or str(phase_root / "evaluation_details.csv")
    Path(metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(details_csv).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "scripts.evaluate_reference_quality",
        "--tree_crowns_shp",
        str(inst_shp),
        "--patch_raster",
        str(cfg["input_image"]),
        "--reference_vector",
        str(_reference_vector_path(cfg)),
        "--evaluation_metrics_json",
        str(metrics_json),
        "--evaluation_details_csv",
        str(details_csv),
        "--id_field",
        str(_reference_id_field(cfg)),
        "--tree_count_field",
        str(cfg["tree_count_field"]),
        "--crown_field",
        str(cfg["crown_field"]),
        "--closure_field",
        str(cfg["closure_field"]),
        "--area_ha_field",
        str(cfg["area_ha_field"]),
    ]
    if cfg.get("density_field"):
        cmd.extend(["--density_field", str(cfg["density_field"])])
    if terrain_info.get("dem_tif"):
        cmd.extend(["--dem_tif", str(terrain_info["dem_tif"])])
    if terrain_info.get("slope_tif"):
        cmd.extend(["--slope_tif", str(terrain_info["slope_tif"])])
    if terrain_info.get("aspect_tif"):
        cmd.extend(["--aspect_tif", str(terrain_info["aspect_tif"])])
    cmd.extend(
        [
            "--flat_slope_threshold_deg",
            str(cfg.get("flat_slope_threshold_deg", 5.0)),
            "--plain_relief_threshold_m",
            str(cfg.get("plain_relief_threshold_m", 30.0)),
        ]
    )

    runner = command_runner or subprocess.run
    result = runner(cmd)
    returncode = getattr(result, "returncode", 0)
    if returncode != 0:
        stderr = getattr(result, "stderr", "") or getattr(result, "stdout", "")
        raise RuntimeError(f"Reference quality evaluation failed:\n{stderr}")

    metrics = _load_json(metrics_json)
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"Evaluation metrics json is empty or invalid: {metrics_json}")

    detail_summary = summarize_details_csv(details_csv, top_k=5, cfg=cfg)
    payload = ReferenceQualityResult(
        assessment_phase=assessment_phase,
        metrics_json=str(metrics_json),
        details_csv=str(details_csv),
        metrics=metrics,
        detail_summary=detail_summary,
        quality_score=score_reference_metrics(metrics, cfg=cfg),
        terrain_error_summary=metrics.get("terrain_stratified_error_summary") or {},
    )
    result_dict = payload.to_dict()
    result_dict["cmd"] = cmd
    result_dict["terrain_info"] = terrain_info
    result_dict["score_breakdown"] = build_reference_score_breakdown(metrics, cfg=cfg)
    return result_dict
