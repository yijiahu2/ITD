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


def _clamp01(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(min(max(value, 0.0), 1.0))


DEFAULT_ERROR_TOLERANCES = {
    "tree_count_error_ratio": 0.20,
    "mean_crown_width_error_ratio": 0.15,
    "closure_error_abs": 0.10,
    "density_error_ratio": 0.20,
}


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


def _normalize_fraction(value: float | None) -> float | None:
    if value is None:
        return None
    if value > 1.0 and value <= 100.0:
        return float(value / 100.0)
    return float(value)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) <= 1.0e-6:
        return None
    return float(numerator / denominator)


def _resolve_reference_tolerances(cfg: dict[str, Any] | None) -> dict[str, float]:
    evaluation_cfg = ((cfg or {}).get("evaluation") or {}).get("analysis") or {}
    raw = (
        evaluation_cfg.get("reference_error_tolerances")
        or evaluation_cfg.get("reference_tolerances")
        or _get_roi_refine_block(cfg).get("reference_error_tolerances")
        or {}
    )
    tolerances: dict[str, float] = {}
    for key, default_value in DEFAULT_ERROR_TOLERANCES.items():
        value = _safe_float((raw or {}).get(key))
        tolerances[key] = float(value if value is not None and value > 0 else default_value)
    return tolerances


def build_reference_score_breakdown(
    metrics: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tree = _safe_float(metrics.get("tree_count_error_ratio"))
    if tree is None:
        tree = _safe_ratio(
            _safe_float(metrics.get("tree_count_error_abs")),
            _safe_float(metrics.get("expected_tree_count")),
        )

    crown = _safe_float(metrics.get("mean_crown_width_error_ratio"))
    if crown is None:
        crown = _safe_ratio(
            _safe_float(metrics.get("mean_crown_width_error_abs")),
            _safe_float(metrics.get("expected_mean_crown_width")),
        )

    closure = _safe_float(metrics.get("closure_error_abs"))
    pred_cover_ratio = _normalize_fraction(_safe_float(metrics.get("pred_cover_ratio")))
    expected_closure = _normalize_fraction(_safe_float(metrics.get("expected_closure")))
    if pred_cover_ratio is not None and expected_closure is not None:
        closure = abs(float(pred_cover_ratio) - float(expected_closure))
    elif closure is not None:
        closure = abs(float(_normalize_fraction(closure) or 0.0))

    density_abs = _safe_float(metrics.get("density_error_abs"))
    expected_density = _safe_float(metrics.get("expected_density"))
    density_ratio = None
    if density_abs is not None and expected_density is not None and expected_density > 0:
        density_ratio = density_abs / expected_density

    focus_mode = "balanced"
    default_weights = DEFAULT_SCORE_WEIGHTS
    if tree is not None and tree <= 0.12:
        focus_mode = "boundary_priority"
        default_weights = BOUNDARY_FOCUSED_SCORE_WEIGHTS

    roi_cfg = _get_roi_refine_block(cfg)
    weights = _normalize_weight_map(roi_cfg.get("score_weights") or {}, default_weights)
    tolerances = _resolve_reference_tolerances(cfg)

    raw_metrics: dict[str, float] = {}
    if tree is not None:
        raw_metrics["tree_count_error_ratio"] = max(float(tree), 0.0)
    if crown is not None:
        raw_metrics["mean_crown_width_error_ratio"] = max(float(crown), 0.0)
    if closure is not None:
        raw_metrics["closure_error_abs"] = max(float(closure), 0.0)
    if density_ratio is not None:
        raw_metrics["density_error_ratio"] = max(float(density_ratio), 0.0)

    if not raw_metrics:
        return {
            "score": None,
            "weights": dict(DEFAULT_SCORE_WEIGHTS),
            "normalized_metrics": {},
            "raw_metrics": {},
            "tolerances": tolerances,
            "focus_mode": "incomplete_metrics",
        }

    normalized_metrics = {
        key: _clamp01(float(value) / max(float(tolerances[key]), 1.0e-6))
        for key, value in raw_metrics.items()
    }
    dynamic_weights = {key: weights[key] for key in normalized_metrics.keys()}
    weight_total = sum(dynamic_weights.values())
    if weight_total > 0:
        dynamic_weights = {key: value / weight_total for key, value in dynamic_weights.items()}
    else:
        dynamic_weights = {}
    weighted_terms = {
        key: {
            **METRIC_CATEGORIES[key],
            "raw_value": raw_metrics[key],
            "tolerance": tolerances[key],
            "value": normalized_metrics[key],
            "weight": dynamic_weights[key],
            "contribution": normalized_metrics[key] * dynamic_weights[key],
        }
        for key in dynamic_weights
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
                "raw_value": term["raw_value"],
                "tolerance": term["tolerance"],
                "value": term["value"],
                "weight": term["weight"],
                "contribution": term["contribution"],
            }
        )
        group["contribution"] = float(group["contribution"]) + float(term["contribution"])
    score = sum(float(term["contribution"]) for term in weighted_terms.values())
    return {
        "score": float(score),
        "weights": dynamic_weights,
        "raw_metrics": raw_metrics,
        "tolerances": tolerances,
        "normalized_metrics": normalized_metrics,
        "weighted_terms": weighted_terms,
        "metric_groups": metric_groups,
        "focus_mode": focus_mode,
        "reference_error_score": float(score),
        "reference_quality_score": float(1.0 - _clamp01(score)),
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
    result_dict["reference_error_score"] = result_dict.get("quality_score")
    reference_error_score = result_dict.get("reference_error_score")
    result_dict["reference_quality_score"] = None if reference_error_score is None else float(1.0 - _clamp01(reference_error_score))
    return result_dict
