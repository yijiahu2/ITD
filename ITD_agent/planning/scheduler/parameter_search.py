from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

from ITD_agent.common.config_refs import reference_id_field as cfg_reference_id_field
from ITD_agent.common.config_refs import reference_vector_path as cfg_reference_vector_path
from ITD_agent.common.values import safe_float as _safe_float
from ITD_agent.data_processing.vector import crop_raster_to_geometry
from ITD_agent.data_processing.roi.extractor import clip_xiaoban_to_geometry_with_fields, crop_roi_terrain_bundle
from ITD_agent.evaluation_analysis.reference_quality_engine import evaluate_reference_quality, score_reference_metrics
from tools.cached_stage_runners import run_segmentation_cached, run_semantic_prior_cached


CORE_KEYS = ("diam_list", "tile", "overlap", "tile_overlap", "augment", "iou_merge_thr", "bsize")
SAFE_BSIZE = 256


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalize_diam_list(value: Any) -> str:
    raw = str(value or "").strip()
    values: list[int] = []
    for part in raw.split(","):
        try:
            item = int(round(float(part.strip())))
        except Exception:
            continue
        item = max(96, min(384, int(round(item / 32.0) * 32)))
        if item not in values:
            values.append(item)
    if not values:
        return "160,256,384"
    return ",".join(str(item) for item in values[:3])


def _shift_diam_list(value: Any, delta: int) -> str:
    shifted: list[int] = []
    for part in _normalize_diam_list(value).split(","):
        item = int(part)
        item = max(96, min(384, int(round((item + delta) / 32.0) * 32)))
        if item not in shifted:
            shifted.append(item)
    return ",".join(str(item) for item in shifted)


def _round_overlap(value: int) -> int:
    allowed = [128, 192, 256, 384]
    return min(allowed, key=lambda item: abs(item - int(value)))


def _round_tile_overlap(value: float) -> float:
    allowed = [0.25, 0.30, 0.35, 0.40, 0.45]
    return min(allowed, key=lambda item: abs(item - float(value)))


def _round_iou(value: float) -> float:
    allowed = [0.22, 0.28, 0.35, 0.40, 0.50]
    return min(allowed, key=lambda item: abs(item - float(value)))


def _get_search_block(runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    planning_cfg = ((runtime_cfg.get("ITD_agent") or {}).get("planning") or {})
    adaptive = planning_cfg.get("adaptive_generation") or {}
    search_cfg = adaptive.get("parameter_search")
    return search_cfg if isinstance(search_cfg, dict) else {}


def _pilot_window_px(runtime_cfg: dict[str, Any], search_cfg: dict[str, Any]) -> int:
    value = int(search_cfg.get("pilot_window_px") or 1280)
    return max(768, min(2048, value))


def _build_candidate_pool(
    *,
    runtime_cfg: dict[str, Any],
    preliminary_updates: dict[str, Any],
    scheduler_context: dict[str, Any],
    max_candidates: int,
) -> list[dict[str, Any]]:
    current = {
        key: runtime_cfg.get(key)
        for key in CORE_KEYS
        if runtime_cfg.get(key) not in (None, "")
    }
    base = deepcopy(current)
    base.update(deepcopy(preliminary_updates or {}))
    base["diam_list"] = _normalize_diam_list(base.get("diam_list"))
    base["tile"] = int(base.get("tile") or 2048)
    base["overlap"] = _round_overlap(int(base.get("overlap") or 128))
    base["tile_overlap"] = _round_tile_overlap(float(base.get("tile_overlap") or 0.35))
    base["augment"] = _normalize_bool(base.get("augment", True))
    base["iou_merge_thr"] = _round_iou(float(base.get("iou_merge_thr") or 0.35))
    base["bsize"] = SAFE_BSIZE

    scene = scheduler_context.get("scene_profile") or {}
    quality = scene.get("image_quality_levels") or {}
    texture = scene.get("image_texture_levels") or {}
    candidates = [
        {"candidate_id": "pilot_base", "params": deepcopy(base), "reason": "当前综合推荐参数"},
        {
            "candidate_id": "pilot_split_bias",
            "params": {
                **deepcopy(base),
                "diam_list": _shift_diam_list(base["diam_list"], -32),
                "overlap": _round_overlap(int(base["overlap"]) + 64),
                "tile_overlap": _round_tile_overlap(float(base["tile_overlap"]) + 0.05),
                "iou_merge_thr": _round_iou(float(base["iou_merge_thr"]) - 0.18),
                "augment": True,
            },
            "reason": "偏向抑制欠分割，增加上下文并降低合并阈值",
        },
        {
            "candidate_id": "pilot_merge_bias",
            "params": {
                **deepcopy(base),
                "diam_list": _shift_diam_list(base["diam_list"], 32),
                "iou_merge_thr": _round_iou(float(base["iou_merge_thr"]) + 0.10),
                "tile_overlap": _round_tile_overlap(float(base["tile_overlap"])),
                "augment": True,
            },
            "reason": "偏向抑制过分裂，扩大候选冠幅并提高合并阈值",
        },
        {
            "candidate_id": "pilot_context_bias",
            "params": {
                **deepcopy(base),
                "overlap": _round_overlap(max(int(base["overlap"]), 192)),
                "tile_overlap": _round_tile_overlap(max(float(base["tile_overlap"]), 0.40)),
                "augment": True,
            },
            "reason": "偏向复杂地形/阴影场景，强化跨窗上下文",
        },
    ]

    if quality.get("blur") in {"high", "severe"} or quality.get("shadow") == "heavy":
        candidates.append(
            {
                "candidate_id": "pilot_blur_shadow_bias",
                "params": {
                    **deepcopy(base),
                    "overlap": _round_overlap(max(int(base["overlap"]), 192)),
                    "tile_overlap": _round_tile_overlap(max(float(base["tile_overlap"]), 0.40)),
                    "augment": True,
                },
                "reason": "影像模糊/重阴影时增强重叠与增强",
            }
        )
    if texture.get("complexity") == "high":
        candidates.append(
            {
                "candidate_id": "pilot_complex_texture_bias",
                "params": {
                    **deepcopy(base),
                    "diam_list": _shift_diam_list(base["diam_list"], 0),
                    "iou_merge_thr": _round_iou(max(float(base["iou_merge_thr"]), 0.40)),
                    "augment": True,
                },
                "reason": "复杂纹理下提高实例合并约束",
            }
        )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        item["params"]["bsize"] = SAFE_BSIZE
        signature = json.dumps(item["params"], sort_keys=True, ensure_ascii=False)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
        if len(deduped) >= max_candidates:
            break
    return deduped


def _read_preview_gray(input_image: str, max_dim: int = 512) -> tuple[np.ndarray, rasterio.Affine, int, int]:
    with rasterio.open(input_image) as src:
        scale = max(src.width / float(max_dim), src.height / float(max_dim), 1.0)
        out_w = max(64, int(round(src.width / scale)))
        out_h = max(64, int(round(src.height / scale)))
        arr = src.read([1, 2, 3] if src.count >= 3 else [1], out_shape=(min(src.count, 3), out_h, out_w)).astype(np.float32)
        if arr.shape[0] >= 3:
            gray = 0.2989 * arr[0] + 0.5870 * arr[1] + 0.1140 * arr[2]
        else:
            gray = arr[0]
        gray -= np.nanmin(gray)
        denom = np.nanmax(gray)
        if denom > 0:
            gray = gray / denom
        return gray.astype(np.float32), src.transform, src.width, src.height


def _select_pilot_windows(runtime_cfg: dict[str, Any], search_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    input_image = str(runtime_cfg["input_image"])
    gray, _, full_w, full_h = _read_preview_gray(input_image)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    darkness = np.clip(1.0 - gray, 0.0, 1.0).astype(np.float32)

    h, w = gray.shape
    patch = max(64, min(h, w) // 5)
    stride = max(32, patch // 2)
    candidates: list[dict[str, Any]] = []
    for y in range(0, max(1, h - patch + 1), stride):
        for x in range(0, max(1, w - patch + 1), stride):
            sub_grad = grad[y : y + patch, x : x + patch]
            sub_dark = darkness[y : y + patch, x : x + patch]
            center_bias = 1.0 - (abs((x + patch / 2) / max(w, 1) - 0.5) + abs((y + patch / 2) / max(h, 1) - 0.5))
            candidates.append(
                {
                    "x": x,
                    "y": y,
                    "texture_score": float(np.nanmean(sub_grad)),
                    "shadow_score": float(np.nanmean(sub_dark)),
                    "represent_score": float(0.65 * np.nanmean(sub_grad) + 0.20 * np.nanmean(sub_dark) + 0.15 * center_bias),
                }
            )

    def _pick(key: str, used: list[tuple[int, int]]) -> dict[str, Any] | None:
        ordered = sorted(candidates, key=lambda item: float(item[key]), reverse=True)
        for item in ordered:
            if all(abs(item["x"] - ux) >= patch or abs(item["y"] - uy) >= patch for ux, uy in used):
                used.append((int(item["x"]), int(item["y"])))
                return item
        return None

    selected_preview: list[dict[str, Any]] = []
    used: list[tuple[int, int]] = []
    center = {
        "x": max(0, w // 2 - patch // 2),
        "y": max(0, h // 2 - patch // 2),
        "texture_score": float(np.nanmean(grad[max(0, h // 2 - patch // 2) : min(h, h // 2 + patch // 2), max(0, w // 2 - patch // 2) : min(w, w // 2 + patch // 2)])),
        "shadow_score": float(np.nanmean(darkness[max(0, h // 2 - patch // 2) : min(h, h // 2 + patch // 2), max(0, w // 2 - patch // 2) : min(w, w // 2 + patch // 2)])),
        "represent_score": 0.0,
        "window_role": "center_representative",
    }
    used.append((int(center["x"]), int(center["y"])))
    selected_preview.append(center)
    texture_win = _pick("texture_score", used)
    if texture_win:
        texture_win["window_role"] = "high_texture"
        selected_preview.append(texture_win)
    shadow_win = _pick("shadow_score", used)
    if shadow_win:
        shadow_win["window_role"] = "high_shadow"
        selected_preview.append(shadow_win)

    pilot_px = _pilot_window_px(runtime_cfg, search_cfg)
    selected: list[dict[str, Any]] = []
    with rasterio.open(input_image) as src:
        for idx, item in enumerate(selected_preview[: int(search_cfg.get("max_pilots") or 3)], start=1):
            cx = (float(item["x"]) + patch / 2.0) / max(float(w), 1.0)
            cy = (float(item["y"]) + patch / 2.0) / max(float(h), 1.0)
            center_x = int(round(cx * full_w))
            center_y = int(round(cy * full_h))
            half = pilot_px // 2
            x0 = max(0, min(full_w - 1, center_x - half))
            y0 = max(0, min(full_h - 1, center_y - half))
            x1 = min(full_w, x0 + pilot_px)
            y1 = min(full_h, y0 + pilot_px)
            if x1 - x0 < 256 or y1 - y0 < 256:
                continue
            bounds = src.window_bounds(Window(x0, y0, x1 - x0, y1 - y0))
            selected.append(
                {
                    "pilot_id": f"pilot_{idx:02d}",
                    "window_role": item.get("window_role") or "representative",
                    "pixel_window": [int(x0), int(y0), int(x1), int(y1)],
                    "bounds": [float(v) for v in bounds],
                    "geometry": box(*bounds),
                    "crs": src.crs,
                    "texture_score": float(item.get("texture_score") or 0.0),
                    "shadow_score": float(item.get("shadow_score") or 0.0),
                }
            )
    return selected


def _expected_density_and_crown(scheduler_context: dict[str, Any]) -> tuple[float | None, float | None]:
    scene = ((scheduler_context.get("input_assessment") or {}).get("scene_analysis") or {})
    stats = scene.get("inventory_scene_stats") or {}
    return _safe_float(stats.get("density_mean")), _safe_float(stats.get("crown_width_mean"))


def _score_proxy(
    *,
    y_inst_tif: str,
    y_inst_shp: str,
    m_sem_tif: str,
    scheduler_context: dict[str, Any],
) -> dict[str, Any]:
    with rasterio.open(y_inst_tif) as pred_src, rasterio.open(m_sem_tif) as sem_src:
        pred = pred_src.read(1) > 0
        canopy = sem_src.read(1) > 0
    inter = float(np.logical_and(pred, canopy).sum())
    pred_sum = float(pred.sum())
    canopy_sum = float(canopy.sum())
    recall = inter / canopy_sum if canopy_sum > 0 else 0.0
    precision = inter / pred_sum if pred_sum > 0 else 0.0
    fn_ratio = 1.0 - recall if canopy_sum > 0 else 1.0
    fp_ratio = 1.0 - precision if pred_sum > 0 else 1.0

    gdf = gpd.read_file(y_inst_shp)
    if gdf.empty:
        return {
            "proxy_score": 999.0,
            "proxy_metrics": {
                "fn_ratio": 1.0,
                "fp_ratio": 1.0,
                "boundary_touch_ratio": 1.0,
                "tiny_fragment_ratio": 1.0,
                "diameter_error_ratio": None,
                "density_error_ratio": None,
            },
        }
    areas = gdf.geometry.area.astype(float)
    eq_diam = np.sqrt(4.0 * areas / math.pi)
    bounds_geom = box(*gdf.total_bounds)
    boundary_line = bounds_geom.boundary.buffer(max(bounds_geom.length * 0.0005, 0.2))
    boundary_touch_ratio = float(np.mean(gdf.geometry.intersects(boundary_line)))
    tiny_threshold = max(float(np.nanmedian(areas)) * 0.15, 0.5)
    tiny_fragment_ratio = float(np.mean(areas <= tiny_threshold))

    expected_density, expected_crown = _expected_density_and_crown(scheduler_context)
    area_ha = max(bounds_geom.area / 10000.0, 1e-6)
    density_error_ratio = None
    if expected_density and expected_density > 0:
        density_error_ratio = abs((len(gdf) / area_ha) - expected_density) / expected_density
    diameter_error_ratio = None
    if expected_crown and expected_crown > 0:
        diameter_error_ratio = abs(float(np.nanmedian(eq_diam)) - expected_crown) / expected_crown

    score = 1.30 * fn_ratio + 1.10 * fp_ratio + 0.40 * boundary_touch_ratio + 0.35 * tiny_fragment_ratio
    if density_error_ratio is not None:
        score += 0.75 * float(density_error_ratio)
    if diameter_error_ratio is not None:
        score += 0.90 * float(diameter_error_ratio)
    return {
        "proxy_score": float(score),
        "proxy_metrics": {
            "fn_ratio": float(fn_ratio),
            "fp_ratio": float(fp_ratio),
            "boundary_touch_ratio": float(boundary_touch_ratio),
            "tiny_fragment_ratio": float(tiny_fragment_ratio),
            "diameter_error_ratio": None if diameter_error_ratio is None else float(diameter_error_ratio),
            "density_error_ratio": None if density_error_ratio is None else float(density_error_ratio),
        },
    }


def _evaluate_exact_if_possible(
    *,
    runtime_cfg: dict[str, Any],
    scheduler_context: dict[str, Any],
    pilot_dir: Path,
    pilot_geom_gdf: gpd.GeoDataFrame,
    pilot_image: str,
    y_inst_shp: str,
) -> dict[str, Any]:
    reference_vector = cfg_reference_vector_path(runtime_cfg)
    reference_id = cfg_reference_id_field(runtime_cfg)
    if not reference_vector or not Path(str(reference_vector)).exists():
        return {}

    try:
        clipped_xiaoban = clip_xiaoban_to_geometry_with_fields(
            src_vector=str(reference_vector),
            geom_gdf=pilot_geom_gdf,
            out_vector=str(pilot_dir / "pilot_xiaoban.gpkg"),
            xiaoban_id_field=str(reference_id),
            tree_count_field=runtime_cfg.get("tree_count_field"),
            crown_field=runtime_cfg.get("crown_field"),
            closure_field=runtime_cfg.get("closure_field"),
            area_ha_field=runtime_cfg.get("area_ha_field"),
            density_field=runtime_cfg.get("density_field"),
        )
    except Exception as exc:
        return {"exact_error": f"clip_xiaoban_failed:{exc}"}

    terrain_summary = (scheduler_context.get("terrain_summary") or {})
    terrain_info = crop_roi_terrain_bundle(
        roi_geom_gdf=pilot_geom_gdf,
        roi_dir=pilot_dir / "terrain",
        dem_tif=terrain_summary.get("dem_tif"),
        slope_tif=terrain_summary.get("slope_tif"),
        aspect_tif=terrain_summary.get("aspect_tif"),
        landform_tif=terrain_summary.get("landform_tif"),
        slope_position_tif=terrain_summary.get("slope_position_tif"),
    )
    eval_cfg = dict(runtime_cfg)
    eval_cfg["input_image"] = str(pilot_image)
    eval_cfg["reference_vector_path"] = str(clipped_xiaoban)
    eval_cfg["inventory_vector_path"] = str(clipped_xiaoban)
    eval_cfg["xiaoban_shp"] = str(clipped_xiaoban)
    eval_cfg["output_dir"] = str(pilot_dir / "evaluation")
    metrics_json = str(pilot_dir / "evaluation" / "pilot_metrics.json")
    details_csv = str(pilot_dir / "evaluation" / "pilot_details.csv")
    eval_result = evaluate_reference_quality(
        eval_cfg,
        inst_shp=y_inst_shp,
        terrain_info=terrain_info,
        assessment_phase="pilot_parameter_search",
        metrics_json=metrics_json,
        details_csv=details_csv,
    )
    return {
        "exact_quality_score": score_reference_metrics(eval_result.get("metrics") or {}),
        "exact_metrics": eval_result.get("metrics") or {},
        "exact_metrics_json": eval_result.get("metrics_json"),
        "exact_details_csv": eval_result.get("details_csv"),
    }


def _run_candidate_on_pilot(
    *,
    runtime_cfg: dict[str, Any],
    scheduler_context: dict[str, Any],
    pilot: dict[str, Any],
    candidate: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    pilot_dir = output_root / pilot["pilot_id"] / candidate["candidate_id"]
    pilot_dir.mkdir(parents=True, exist_ok=True)
    pilot_geom_gdf = gpd.GeoDataFrame({"pilot_id": [pilot["pilot_id"]]}, geometry=[pilot["geometry"]], crs=pilot["crs"])
    pilot_image = crop_raster_to_geometry(str(runtime_cfg["input_image"]), pilot_geom_gdf, pilot_dir / "pilot_image.tif")

    sem_cfg = dict(runtime_cfg)
    sem_cfg["input_image"] = str(pilot_image)
    sem_cfg["output_dir"] = str(pilot_dir / "semantic_prior")
    semantic_info = run_semantic_prior_cached(sem_cfg)

    seg_cfg = dict(runtime_cfg)
    seg_cfg["input_image"] = str(pilot_image)
    seg_cfg["output_dir"] = str(pilot_dir / "segmentation")
    for key, value in candidate["params"].items():
        seg_cfg[key] = value
    seg_result = run_segmentation_cached(seg_cfg, semantic_info["m_sem_tif"])

    exact = _evaluate_exact_if_possible(
        runtime_cfg=runtime_cfg,
        scheduler_context=scheduler_context,
        pilot_dir=pilot_dir,
        pilot_geom_gdf=pilot_geom_gdf,
        pilot_image=str(pilot_image),
        y_inst_shp=seg_result["y_inst_shp"],
    )
    proxy = _score_proxy(
        y_inst_tif=seg_result["y_inst_tif"],
        y_inst_shp=seg_result["y_inst_shp"],
        m_sem_tif=semantic_info["m_sem_tif"],
        scheduler_context=scheduler_context,
    )
    final_score = exact.get("exact_quality_score")
    score_source = "exact_reference" if final_score is not None else "proxy_quality"
    if final_score is None:
        final_score = proxy["proxy_score"]
    return {
        "pilot_id": pilot["pilot_id"],
        "window_role": pilot["window_role"],
        "candidate_id": candidate["candidate_id"],
        "candidate_reason": candidate["reason"],
        "parameter_updates": deepcopy(candidate["params"]),
        "score_source": score_source,
        "score": float(final_score),
        **exact,
        **proxy,
        "outputs": {
            "pilot_image": str(pilot_image),
            "m_sem_tif": semantic_info["m_sem_tif"],
            "y_inst_tif": seg_result["y_inst_tif"],
            "y_inst_shp": seg_result["y_inst_shp"],
        },
    }


def run_main_model_parameter_search(
    *,
    runtime_cfg: dict[str, Any],
    scheduler_context: dict[str, Any],
    preliminary_updates: dict[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    search_cfg = _get_search_block(runtime_cfg)
    if runtime_cfg.get("_planning_stage") != "main_model":
        return {"enabled": False, "reason": "not_main_model_stage"}
    if not bool(search_cfg.get("enabled", True)):
        return {"enabled": False, "reason": "disabled_by_config"}
    if str(runtime_cfg.get("segmentation_algorithm") or runtime_cfg.get("selected_model_name") or "legacy_cellpose_sam").strip().lower() not in {"", "legacy_cellpose_sam"}:
        return {"enabled": False, "reason": "unsupported_algorithm"}

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    max_candidates = max(2, min(6, int(search_cfg.get("max_candidates") or 4)))
    candidates = _build_candidate_pool(
        runtime_cfg=runtime_cfg,
        preliminary_updates=preliminary_updates,
        scheduler_context=scheduler_context,
        max_candidates=max_candidates,
    )
    pilots = _select_pilot_windows(runtime_cfg, search_cfg)
    if not candidates or not pilots:
        return {
            "enabled": False,
            "reason": "no_candidates_or_pilots",
            "candidate_count": len(candidates),
            "pilot_count": len(pilots),
        }

    candidate_runs: list[dict[str, Any]] = []
    aggregated: list[dict[str, Any]] = []
    incumbent_mean_score: float | None = None
    early_stop_margin = float(search_cfg.get("early_stop_margin") or 0.12)
    early_stop_min_completed = max(1, int(search_cfg.get("early_stop_min_completed_pilots") or 1))
    for candidate in candidates:
        runs: list[dict[str, Any]] = []
        pruned_early = False
        prune_reason = ""
        for pilot_idx, pilot in enumerate(pilots, start=1):
            try:
                runs.append(
                    _run_candidate_on_pilot(
                        runtime_cfg=runtime_cfg,
                        scheduler_context=scheduler_context,
                        pilot=pilot,
                        candidate=candidate,
                        output_root=output_root,
                    )
                )
            except Exception as exc:
                runs.append(
                    {
                        "pilot_id": pilot["pilot_id"],
                        "candidate_id": candidate["candidate_id"],
                        "score": 999.0,
                        "score_source": "error_penalty",
                        "error": str(exc),
                        "parameter_updates": deepcopy(candidate["params"]),
                    }
                )
            scores_so_far = [float(item.get("score") or 999.0) for item in runs]
            if (
                incumbent_mean_score is not None
                and pilot_idx >= early_stop_min_completed
                and scores_so_far
            ):
                optimistic_mean = float(np.mean(scores_so_far))
                if optimistic_mean > incumbent_mean_score + early_stop_margin:
                    pruned_early = True
                    prune_reason = (
                        f"完成 {pilot_idx} 个 pilot 后，当前均分 {optimistic_mean:.4f} "
                        f"已比最优候选 {incumbent_mean_score:.4f} 差超过 {early_stop_margin:.4f}，提前停止。"
                    )
                    break

        if pruned_early and len(runs) < len(pilots):
            carried_score = float(np.mean([float(item.get("score") or 999.0) for item in runs])) if runs else 999.0
            for pilot in pilots[len(runs) :]:
                runs.append(
                    {
                        "pilot_id": pilot["pilot_id"],
                        "candidate_id": candidate["candidate_id"],
                        "score": carried_score,
                        "score_source": "early_stop_proxy",
                        "skipped": True,
                        "reason": prune_reason,
                        "parameter_updates": deepcopy(candidate["params"]),
                    }
                )
        scores = [float(item.get("score") or 999.0) for item in runs]
        mean_score = float(np.mean(scores)) if scores else 999.0
        aggregated.append(
            {
                "candidate_id": candidate["candidate_id"],
                "reason": candidate["reason"],
                "parameter_updates": deepcopy(candidate["params"]),
                "pilot_scores": scores,
                "mean_score": mean_score,
                "max_score": float(np.max(scores)) if scores else 999.0,
                "score_source_mix": sorted({str(item.get("score_source") or "") for item in runs}),
                "completed_pilot_count": sum(1 for item in runs if not item.get("skipped")),
                "pruned_early": pruned_early,
                "prune_reason": prune_reason,
            }
        )
        candidate_runs.extend(runs)
        if incumbent_mean_score is None or mean_score < incumbent_mean_score:
            incumbent_mean_score = mean_score

    ranked = sorted(aggregated, key=lambda item: (float(item["mean_score"]), float(item["max_score"])))
    selected = ranked[0] if ranked else None
    result = {
        "enabled": True,
        "pilot_count": len(pilots),
        "candidate_count": len(candidates),
        "pilots": [
            {
                key: (str(value) if key == "crs" else value)
                for key, value in pilot.items()
                if key != "geometry"
            }
            for pilot in pilots
        ],
        "candidates": ranked,
        "candidate_runs": candidate_runs,
        "selected_candidate_id": selected.get("candidate_id") if selected else None,
        "selected_parameter_updates": deepcopy(selected.get("parameter_updates") or {}),
    }
    (output_root / "pilot_parameter_search_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
