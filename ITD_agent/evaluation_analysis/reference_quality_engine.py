from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .artifact_store import get_phase_dir
from .contracts import ReferenceQualityResult
from .detail_ranker import summarize_details_csv


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


def score_reference_metrics(metrics: dict[str, Any]) -> float | None:
    tree = _safe_float(metrics.get("tree_count_error_ratio"))
    crown = _safe_float(metrics.get("mean_crown_width_error_ratio"))
    closure = _safe_float(metrics.get("closure_error_abs"))
    density = _safe_float(metrics.get("density_error_abs"))
    if tree is None or crown is None or closure is None:
        return None
    return tree + crown + closure + (density or 0.0) / 1000.0


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
        "--xiaoban_shp",
        str(cfg["xiaoban_shp"]),
        "--evaluation_metrics_json",
        str(metrics_json),
        "--evaluation_details_csv",
        str(details_csv),
        "--id_field",
        str(cfg["xiaoban_id_field"]),
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
        quality_score=score_reference_metrics(metrics),
        terrain_error_summary=metrics.get("terrain_stratified_error_summary") or {},
    )
    result_dict = payload.to_dict()
    result_dict["cmd"] = cmd
    result_dict["terrain_info"] = terrain_info
    return result_dict
