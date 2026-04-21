from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from input_layer.contracts import InputManifest


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _load_height_profile(path: str | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {"available": False, "path": path}

    with rasterio.open(path) as src:
        out_h = min(src.height, 512)
        out_w = min(src.width, 512)
        arr = src.read(1, out_shape=(out_h, out_w)).astype(np.float32)
        nodata = src.nodata
        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= ~np.isclose(arr, float(nodata))
        if not np.any(valid):
            return {
                "available": False,
                "path": path,
                "resolution_m": abs(float(src.transform.a)),
                "reason": "no_valid_pixels",
            }

        vals = arr[valid]
        gy, gx = np.gradient(np.where(valid, arr, np.nanmean(vals)))
        grad = np.sqrt(gx * gx + gy * gy)
        peak_mask = (
            (arr > np.nanpercentile(vals, 85))
            & valid
        )
        peak_density = float(peak_mask.mean())
        return {
            "available": True,
            "path": path,
            "resolution_m": abs(float(src.transform.a)),
            "valid_ratio": float(valid.mean()),
            "height_mean": float(np.mean(vals)),
            "height_p50": float(np.percentile(vals, 50)),
            "height_p95": float(np.percentile(vals, 95)),
            "height_max": float(np.max(vals)),
            "height_std": float(np.std(vals)),
            "peak_density": peak_density,
            "height_edge_strength": float(np.nanmean(grad)),
            "pixel_count": int(vals.size),
        }


def _build_public_dataset_prior(public_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    if not public_profiles:
        return {
            "available": False,
            "nearest_domains": [],
            "recommended_model_families": [],
            "known_failure_modes": [],
        }
    domains = []
    families = []
    for item in public_profiles:
        for value in item.get("forest_types") or []:
            if value and value not in domains:
                domains.append(str(value))
        for value in item.get("target_expert_families") or []:
            if value and value not in families:
                families.append(str(value))
    return {
        "available": True,
        "dataset_count": len(public_profiles),
        "nearest_domains": domains[:6],
        "recommended_model_families": families[:6],
        "known_failure_modes": [],
    }


def build_online_scene_state(
    *,
    runtime_cfg: dict[str, Any],
    input_manifest: InputManifest,
    terrain_info: dict[str, Any],
    data_processing_summary: dict[str, Any],
) -> dict[str, Any]:
    image_profiles = data_processing_summary.get("image_profiles") or []
    image_profile = image_profiles[0] if image_profiles else {}
    texture = image_profile.get("texture_summary") or {}
    quality = (image_profile.get("quality_summary") or {}).get("quality_metrics") or {}
    public_profiles = data_processing_summary.get("public_dataset_profiles") or []

    height_profiles = data_processing_summary.get("height_raster_profiles") or []
    chm_profile = next((item for item in height_profiles if item.get("role") == "chm"), None)
    dsm_profile = next((item for item in height_profiles if item.get("role") == "dsm"), None)
    chm_path = (
        (chm_profile or {}).get("normalization", {}).get("normalized_path")
        or runtime_cfg.get("chm_tif")
        or (input_manifest.chm_paths[0] if input_manifest.chm_paths else None)
    )
    dsm_path = (
        (dsm_profile or {}).get("normalization", {}).get("normalized_path")
        or runtime_cfg.get("dsm_tif")
        or (input_manifest.dsm_paths[0] if input_manifest.dsm_paths else None)
    )

    return {
        "scene_id": str(runtime_cfg.get("run_name") or "unknown_scene"),
        "input_availability": {
            "has_dom": bool(input_manifest.remote_sensing),
            "has_dem": bool(input_manifest.terrain_dem),
            "has_chm": bool(chm_path),
            "has_dsm": bool(dsm_path),
            "has_public_dataset_prior": bool(public_profiles),
            "has_xiaoban_optional": bool(input_manifest.industry_vectors),
        },
        "dom_profile": {
            "source_id": image_profile.get("source_id"),
            "resolution_m": image_profile.get("resolution_x_m") or image_profile.get("resolution_y_m"),
            "width": image_profile.get("width"),
            "height": image_profile.get("height"),
            "area_ha": image_profile.get("area_ha"),
            "texture_metrics": {
                "contrast": _safe_float(texture.get("contrast")),
                "entropy": _safe_float(texture.get("entropy")),
                "energy": _safe_float(texture.get("energy")),
                "correlation": _safe_float(texture.get("correlation")),
                "homogeneity": _safe_float(texture.get("homogeneity")),
                "gradient_mean": _safe_float(texture.get("gradient_mean")),
            },
            "quality_metrics": {
                "laplacian_variance": _safe_float(quality.get("laplacian_variance")),
                "shadow_ratio_estimate": _safe_float(quality.get("shadow_ratio_estimate")),
                "stripe_noise_score": _safe_float(quality.get("stripe_noise_score")),
                "color_cast_score": _safe_float(quality.get("color_cast_score")),
            },
        },
        "dem_profile": {
            "path": runtime_cfg.get("dem_tif"),
            "resolution_m": (
                ((data_processing_summary.get("dem_profiles") or [{}])[0].get("resolution_x_m"))
                if (data_processing_summary.get("dem_profiles") or [])
                else None
            ),
            "global_terrain_background": terrain_info.get("global_terrain_background") or {},
            "dom_terrain_context": terrain_info.get("dom_terrain_context") or {},
            "terrain_layer_policy": terrain_info.get("terrain_layer_policy") or {},
        },
        "chm_profile": chm_profile or _load_height_profile(chm_path),
        "dsm_profile": dsm_profile or _load_height_profile(dsm_path),
        "semantic_prior_profile": {
            "status": "pending",
            "reason": "semantic prior is generated after online_scene_state in the current staged rollout",
        },
        "public_dataset_prior": _build_public_dataset_prior(public_profiles),
        "optional_xiaoban_profile": {
            "enabled": bool(input_manifest.industry_vectors),
            "field_mapping": (input_manifest.industry_vectors[0].field_mapping if input_manifest.industry_vectors else {}),
            "reference_ready": bool(input_manifest.industry_vectors),
        },
    }
